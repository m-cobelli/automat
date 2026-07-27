from __future__ import annotations

import math
from collections.abc import Callable

from pymatgen.core import Composition, Element


CORE_PROPERTIES: tuple[str, ...] = (
    "Z",
    "X",
    "row",
    "group",
    "atomic_mass",
    "atomic_radius",
    "mendeleev_no",
    "electron_affinity",
    "ionization_energy",
)
OXIDATION_GUESS_MAX_REDUCED_ATOMS = 40.0


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _element_property(element: Element, property_name: str) -> float | None:
    try:
        value = getattr(element, property_name)
    except Exception:
        return None
    return _as_float(value)


def _weighted_stats(
    elements: list[Element],
    fractions: list[float],
    property_name: str,
) -> list[float]:
    values: list[tuple[float, float]] = []
    missing_fraction = 0.0
    for element, fraction in zip(elements, fractions, strict=True):
        value = _element_property(element, property_name)
        if value is None:
            missing_fraction += fraction
        else:
            values.append((value, fraction))

    if not values:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    weight_sum = sum(weight for _, weight in values)
    normalized = [(value, weight / weight_sum) for value, weight in values]
    mean = sum(value * weight for value, weight in normalized)
    variance = sum(weight * (value - mean) ** 2 for value, weight in normalized)
    std = math.sqrt(max(variance, 0.0))
    minimum = min(value for value, _ in normalized)
    maximum = max(value for value, _ in normalized)
    mean_abs_dev = sum(weight * abs(value - mean) for value, weight in normalized)
    return [
        mean,
        std,
        minimum,
        maximum,
        maximum - minimum,
        mean_abs_dev,
        missing_fraction,
    ]


def _class_fraction(
    elements: list[Element],
    fractions: list[float],
    predicate: Callable[[Element], bool],
) -> float:
    total = 0.0
    for element, fraction in zip(elements, fractions, strict=True):
        try:
            if predicate(element):
                total += fraction
        except Exception:
            continue
    return total


def _is_nonmetal(element: Element) -> bool:
    return not element.is_metal and not element.is_metalloid


def _is_chalcogen(element: Element) -> bool:
    return element.group == 16


def _subset_weighted_mean(
    elements: list[Element],
    fractions: list[float],
    include: Callable[[Element], bool],
    property_name: str,
) -> float:
    values: list[tuple[float, float]] = []
    for element, fraction in zip(elements, fractions, strict=True):
        if not include(element):
            continue
        value = _element_property(element, property_name)
        if value is not None:
            values.append((value, fraction))
    weight_sum = sum(weight for _, weight in values)
    if weight_sum <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in values) / weight_sum


def baseline(composition: Composition) -> list[float]:
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    elements = [element for element, _ in element_amounts]
    fractions = [float(amount) / total_atoms for _, amount in element_amounts]

    max_fraction = max(fractions)
    min_fraction = min(fractions)
    entropy = -sum(fraction * math.log(fraction) for fraction in fractions if fraction > 0.0)
    concentration = sum(fraction * fraction for fraction in fractions)

    features = [
        float(len(elements)),
        float(composition.reduced_composition.num_atoms),
        max_fraction,
        min_fraction,
        max_fraction - min_fraction,
        concentration,
        entropy,
    ]

    for property_name in CORE_PROPERTIES:
        features.extend(_weighted_stats(elements, fractions, property_name))

    features.extend(
        [
            _class_fraction(elements, fractions, lambda element: bool(element.is_metal)),
            _class_fraction(elements, fractions, lambda element: bool(element.is_metalloid)),
            _class_fraction(elements, fractions, _is_nonmetal),
            _class_fraction(elements, fractions, lambda element: bool(element.is_noble_gas)),
            _class_fraction(elements, fractions, lambda element: bool(element.is_transition_metal)),
            _class_fraction(elements, fractions, lambda element: bool(element.is_alkali)),
            _class_fraction(elements, fractions, lambda element: bool(element.is_alkaline)),
            _class_fraction(elements, fractions, lambda element: element.group == 17),
            _class_fraction(elements, fractions, _is_chalcogen),
            _class_fraction(elements, fractions, lambda element: bool(element.is_rare_earth_metal)),
        ]
    )

    return [float(value) for value in features]


def _first_oxidation_guess(composition: Composition) -> dict[str, float]:
    if composition.reduced_composition.num_atoms > OXIDATION_GUESS_MAX_REDUCED_ATOMS:
        return {}
    try:
        guesses = composition.oxi_state_guesses(max_sites=-1)
    except Exception:
        return {}
    if not guesses:
        return {}
    return {str(symbol): float(oxi_state) for symbol, oxi_state in guesses[0].items()}


def _pairwise_ionic_features(elements: list[Element], fractions: list[float]) -> list[float]:
    weighted_diff = 0.0
    weighted_ionic = 0.0
    metal_nonmetal_weight = 0.0
    max_diff = 0.0
    pair_weight_sum = 0.0

    for i, element_i in enumerate(elements):
        x_i = _element_property(element_i, "X")
        if x_i is None:
            continue
        for j in range(i + 1, len(elements)):
            element_j = elements[j]
            x_j = _element_property(element_j, "X")
            if x_j is None:
                continue
            weight = fractions[i] * fractions[j]
            diff = abs(x_i - x_j)
            pair_weight_sum += weight
            weighted_diff += weight * diff
            weighted_ionic += weight * (1.0 - math.exp(-0.25 * diff * diff))
            max_diff = max(max_diff, diff)
            if element_i.is_metal != element_j.is_metal:
                metal_nonmetal_weight += weight

    if pair_weight_sum <= 0.0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        weighted_diff / pair_weight_sum,
        max_diff,
        weighted_ionic / pair_weight_sum,
        metal_nonmetal_weight / pair_weight_sum,
    ]


def _weighted_numeric_stats(values: list[float], fractions: list[float]) -> list[float]:
    mean = sum(value * fraction for value, fraction in zip(values, fractions, strict=True))
    variance = sum(
        fraction * (value - mean) ** 2
        for value, fraction in zip(values, fractions, strict=True)
    )
    std = math.sqrt(max(variance, 0.0))
    minimum = min(values, default=0.0)
    maximum = max(values, default=0.0)
    return [mean, std, minimum, maximum, maximum - minimum]


def _valence_orbital_counts(element: Element) -> dict[str, float]:
    shell_occupancy = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}
    try:
        structure = element.full_electronic_structure
    except Exception:
        structure = []
    if not structure:
        return {
            "s": 0.0,
            "p": 0.0,
            "d": 0.0,
            "f": 0.0,
            "total": 0.0,
            "open": 0.0,
            "open_fraction": 0.0,
        }

    outer_n = max(int(n) for n, _, _ in structure)
    for n, orbital, electrons in structure:
        if orbital in {"s", "p"} and int(n) == outer_n:
            shell_occupancy[orbital] += float(electrons)
        elif orbital == "d" and int(n) == outer_n - 1:
            shell_occupancy["d"] += float(electrons)
        elif orbital == "f" and int(n) == outer_n - 2:
            shell_occupancy["f"] += float(electrons)

    capacities = {"s": 2.0, "p": 6.0, "d": 10.0, "f": 14.0}
    open_capacity = 0.0
    active_capacity = 0.0
    for orbital, capacity in capacities.items():
        occupied = shell_occupancy[orbital]
        if occupied > 0.0:
            active_capacity += capacity
            open_capacity += min(occupied, capacity - occupied)
    total = sum(shell_occupancy.values())
    open_fraction = open_capacity / active_capacity if active_capacity > 0.0 else 0.0
    return {
        **shell_occupancy,
        "total": total,
        "open": open_capacity,
        "open_fraction": open_fraction,
    }


def oxidation_ionic_baseline(composition: Composition) -> list[float]:
    features = baseline(composition)
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    elements = [element for element, _ in element_amounts]
    fractions = [float(amount) / total_atoms for _, amount in element_amounts]

    oxidation_guess = _first_oxidation_guess(composition)
    oxidation_states = [oxidation_guess.get(element.symbol, 0.0) for element in elements]
    has_guess = bool(oxidation_guess)

    positive_fraction = sum(
        fraction for fraction, oxi_state in zip(fractions, oxidation_states, strict=True)
        if oxi_state > 0.0
    )
    negative_fraction = sum(
        fraction for fraction, oxi_state in zip(fractions, oxidation_states, strict=True)
        if oxi_state < 0.0
    )
    neutral_fraction = sum(
        fraction for fraction, oxi_state in zip(fractions, oxidation_states, strict=True)
        if oxi_state == 0.0
    )
    charge_balance = abs(
        sum(fraction * oxi_state for fraction, oxi_state in zip(fractions, oxidation_states, strict=True))
    )
    mean_abs_oxidation = sum(
        fraction * abs(oxi_state)
        for fraction, oxi_state in zip(fractions, oxidation_states, strict=True)
    )
    positive_states = [oxi_state for oxi_state in oxidation_states if oxi_state > 0.0]
    negative_states = [oxi_state for oxi_state in oxidation_states if oxi_state < 0.0]
    max_positive = max(positive_states, default=0.0)
    min_negative = min(negative_states, default=0.0)
    oxidation_range = max(oxidation_states, default=0.0) - min(oxidation_states, default=0.0)
    positive_charge_density = sum(
        fraction * max(oxi_state, 0.0)
        for fraction, oxi_state in zip(fractions, oxidation_states, strict=True)
    )
    negative_charge_density = sum(
        fraction * abs(min(oxi_state, 0.0))
        for fraction, oxi_state in zip(fractions, oxidation_states, strict=True)
    )

    is_cation = lambda element: oxidation_guess.get(element.symbol, 0.0) > 0.0
    is_anion = lambda element: oxidation_guess.get(element.symbol, 0.0) < 0.0
    cation_x = _subset_weighted_mean(elements, fractions, is_cation, "X")
    anion_x = _subset_weighted_mean(elements, fractions, is_anion, "X")
    cation_z = _subset_weighted_mean(elements, fractions, is_cation, "Z")
    anion_z = _subset_weighted_mean(elements, fractions, is_anion, "Z")
    cation_group = _subset_weighted_mean(elements, fractions, is_cation, "group")
    anion_group = _subset_weighted_mean(elements, fractions, is_anion, "group")
    cation_row = _subset_weighted_mean(elements, fractions, is_cation, "row")
    anion_row = _subset_weighted_mean(elements, fractions, is_anion, "row")
    cation_radius = _subset_weighted_mean(elements, fractions, is_cation, "atomic_radius")
    anion_radius = _subset_weighted_mean(elements, fractions, is_anion, "atomic_radius")

    features.extend(
        [
            0.0 if has_guess else 1.0,
            positive_fraction,
            negative_fraction,
            neutral_fraction,
            charge_balance,
            mean_abs_oxidation,
            max_positive,
            min_negative,
            oxidation_range,
            positive_charge_density,
            negative_charge_density,
            cation_x,
            anion_x,
            anion_x - cation_x,
            cation_z,
            anion_z,
            anion_z - cation_z,
            cation_group,
            anion_group,
            cation_row,
            anion_row,
            cation_radius,
            anion_radius,
            anion_radius - cation_radius,
        ]
    )
    features.extend(_pairwise_ionic_features(elements, fractions))

    features.extend(
        [
            _class_fraction(elements, fractions, lambda element: is_anion(element) and element.symbol == "O"),
            _class_fraction(elements, fractions, lambda element: is_anion(element) and element.group == 17),
            _class_fraction(elements, fractions, lambda element: is_anion(element) and element.group == 16),
            _class_fraction(elements, fractions, lambda element: is_anion(element) and element.group == 15),
            _class_fraction(
                elements,
                fractions,
                lambda element: is_anion(element) and element.symbol in {"C", "N"},
            ),
        ]
    )

    return [float(value) for value in features]


def oxidation_valence_orbital(composition: Composition) -> list[float]:
    features = oxidation_ionic_baseline(composition)
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    elements = [element for element, _ in element_amounts]
    fractions = [float(amount) / total_atoms for _, amount in element_amounts]
    orbital_counts = [_valence_orbital_counts(element) for element in elements]

    for key in ("s", "p", "d", "f", "total", "open", "open_fraction"):
        values = [counts[key] for counts in orbital_counts]
        features.extend(_weighted_numeric_stats(values, fractions))

    features.extend(
        [
            _class_fraction(elements, fractions, lambda element: element.block == "s"),
            _class_fraction(elements, fractions, lambda element: element.block == "p"),
            _class_fraction(elements, fractions, lambda element: element.block == "d"),
            _class_fraction(elements, fractions, lambda element: element.block == "f"),
        ]
    )

    main_group_values = [
        float(element.group if element.group <= 2 else element.group - 10)
        if element.block in {"s", "p"}
        else 0.0
        for element in elements
    ]
    features.extend(_weighted_numeric_stats(main_group_values, fractions))

    partial_d_fraction = 0.0
    partial_f_fraction = 0.0
    for element, fraction, counts in zip(elements, fractions, orbital_counts, strict=True):
        if element.block == "d" and 0.0 < counts["d"] < 10.0:
            partial_d_fraction += fraction
        if element.block == "f" and 0.0 < counts["f"] < 14.0:
            partial_f_fraction += fraction
    features.extend([partial_d_fraction, partial_f_fraction])

    oxidation_guess = _first_oxidation_guess(composition)
    anion_p_values = [
        counts["p"] if oxidation_guess.get(element.symbol, 0.0) < 0.0 else 0.0
        for element, counts in zip(elements, orbital_counts, strict=True)
    ]
    features.extend(_weighted_numeric_stats(anion_p_values, fractions))

    return [float(value) for value in features]


def _subset_orbital_mean(
    elements: list[Element],
    fractions: list[float],
    orbital_counts: list[dict[str, float]],
    include: Callable[[Element], bool],
    key: str,
) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for element, fraction, counts in zip(elements, fractions, orbital_counts, strict=True):
        if include(element):
            weighted_sum += fraction * counts[key]
            weight_sum += fraction
    if weight_sum <= 0.0:
        return 0.0
    return weighted_sum / weight_sum


def charge_orbital_interactions(composition: Composition) -> list[float]:
    features = oxidation_valence_orbital(composition)
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    elements = [element for element, _ in element_amounts]
    fractions = [float(amount) / total_atoms for _, amount in element_amounts]
    orbital_counts = [_valence_orbital_counts(element) for element in elements]
    oxidation_guess = _first_oxidation_guess(composition)

    is_cation = lambda element: oxidation_guess.get(element.symbol, 0.0) > 0.0
    is_anion = lambda element: oxidation_guess.get(element.symbol, 0.0) < 0.0
    cation_fraction = _class_fraction(elements, fractions, is_cation)
    anion_fraction = _class_fraction(elements, fractions, is_anion)

    cation_means = {
        key: _subset_orbital_mean(elements, fractions, orbital_counts, is_cation, key)
        for key in ("s", "p", "d", "f", "total", "open", "open_fraction")
    }
    anion_means = {
        key: _subset_orbital_mean(elements, fractions, orbital_counts, is_anion, key)
        for key in ("s", "p", "d", "f", "total", "open", "open_fraction")
    }
    cation_radius = _subset_weighted_mean(elements, fractions, is_cation, "atomic_radius")
    anion_radius = _subset_weighted_mean(elements, fractions, is_anion, "atomic_radius")
    cation_x = _subset_weighted_mean(elements, fractions, is_cation, "X")
    anion_x = _subset_weighted_mean(elements, fractions, is_anion, "X")
    x_separation = max(anion_x - cation_x, 0.0)

    for key in ("s", "p", "d", "f", "total", "open", "open_fraction"):
        features.extend([cation_means[key], anion_means[key], anion_means[key] - cation_means[key]])

    partial_d_cation_fraction = 0.0
    filled_d_cation_fraction = 0.0
    high_p_anion_fraction = 0.0
    low_p_anion_fraction = 0.0
    positive_ionic_potential = 0.0
    negative_ionic_potential = 0.0
    for element, fraction, counts in zip(elements, fractions, orbital_counts, strict=True):
        radius = _element_property(element, "atomic_radius") or 0.0
        oxi_state = oxidation_guess.get(element.symbol, 0.0)
        if oxi_state > 0.0:
            if 0.0 < counts["d"] < 10.0:
                partial_d_cation_fraction += fraction
            if counts["d"] >= 9.5:
                filled_d_cation_fraction += fraction
            if radius > 0.0:
                positive_ionic_potential += fraction * oxi_state / radius
        elif oxi_state < 0.0:
            if counts["p"] >= 4.0:
                high_p_anion_fraction += fraction
            if 0.0 < counts["p"] < 4.0:
                low_p_anion_fraction += fraction
            if radius > 0.0:
                negative_ionic_potential += fraction * abs(oxi_state) / radius

    radius_ratio = cation_radius / anion_radius if anion_radius > 0.0 else 0.0
    pair_features = _pairwise_ionic_features(elements, fractions)
    metal_nonmetal_pair_weight = pair_features[3]

    features.extend(
        [
            cation_fraction,
            anion_fraction,
            partial_d_cation_fraction,
            filled_d_cation_fraction,
            high_p_anion_fraction,
            low_p_anion_fraction,
            anion_means["p"] * cation_means["d"],
            anion_means["p"] * cation_means["s"],
            anion_means["p"] * cation_means["open_fraction"],
            cation_means["d"] * x_separation,
            cation_means["open_fraction"] * metal_nonmetal_pair_weight,
            radius_ratio,
            positive_ionic_potential,
            negative_ionic_potential,
        ]
    )

    return [float(value) for value in features]


def _element_identity_features(composition: Composition) -> list[float]:
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    identity = [0.0] * 118
    for element, amount in element_amounts:
        atomic_number = int(element.Z)
        if 1 <= atomic_number <= 118:
            identity[atomic_number - 1] = float(amount) / total_atoms
    return [
        *identity,
        max(identity, default=0.0),
        float(sum(1 for fraction in identity if fraction > 0.0)),
        sum(fraction * fraction for fraction in identity),
        sum(identity[atomic_number - 1] for atomic_number in range(55, 119)),
    ]


def charge_orbital_identity(composition: Composition) -> list[float]:
    features = charge_orbital_interactions(composition)
    features.extend(_element_identity_features(composition))
    return [float(value) for value in features]


def charge_orbital_identity_family(composition: Composition) -> list[float]:
    features = charge_orbital_identity(composition)
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    elements = [element for element, _ in element_amounts]
    fractions = [float(amount) / total_atoms for _, amount in element_amounts]

    oxygen = _class_fraction(elements, fractions, lambda element: element.symbol == "O")
    nitrogen = _class_fraction(elements, fractions, lambda element: element.symbol == "N")
    carbon = _class_fraction(elements, fractions, lambda element: element.symbol == "C")
    halogen = _class_fraction(elements, fractions, lambda element: element.group == 17)
    light_halogen = _class_fraction(elements, fractions, lambda element: element.symbol in {"F", "Cl"})
    heavy_halogen = _class_fraction(elements, fractions, lambda element: element.symbol in {"Br", "I"})
    heavy_chalcogen = _class_fraction(elements, fractions, lambda element: element.symbol in {"S", "Se", "Te"})
    pnictogen = _class_fraction(elements, fractions, lambda element: element.group == 15)
    transition = _class_fraction(elements, fractions, lambda element: bool(element.is_transition_metal))
    post_transition = _class_fraction(
        elements,
        fractions,
        lambda element: bool(getattr(element, "is_post_transition_metal", False)),
    )
    alkali_alkaline = _class_fraction(
        elements,
        fractions,
        lambda element: bool(element.is_alkali) or bool(element.is_alkaline),
    )
    metalloid = _class_fraction(elements, fractions, lambda element: bool(element.is_metalloid))
    rare = _class_fraction(
        elements,
        fractions,
        lambda element: bool(getattr(element, "is_lanthanoid", False))
        or bool(getattr(element, "is_actinoid", False)),
    )
    heavy_p_block = _class_fraction(
        elements,
        fractions,
        lambda element: element.block == "p" and element.Z >= 49,
    )
    metal = _class_fraction(elements, fractions, lambda element: bool(element.is_metal))
    chalcogen = oxygen + heavy_chalcogen

    features.extend(
        [
            oxygen,
            nitrogen,
            carbon,
            halogen,
            light_halogen,
            heavy_halogen,
            heavy_chalcogen,
            pnictogen,
            transition,
            post_transition,
            alkali_alkaline,
            metalloid,
            rare,
            heavy_p_block,
            transition * chalcogen,
            transition * pnictogen,
            post_transition * chalcogen,
            alkali_alkaline * halogen,
            alkali_alkaline * oxygen,
            metalloid * chalcogen,
            heavy_p_block * chalcogen,
            metal * oxygen,
            metal * nitrogen,
            metal * halogen,
            metal * chalcogen,
            oxygen / heavy_chalcogen if heavy_chalcogen > 0.0 else 0.0,
            halogen / chalcogen if chalcogen > 0.0 else 0.0,
            transition / alkali_alkaline if alkali_alkaline > 0.0 else 0.0,
        ]
    )
    return [float(value) for value in features]


def _charge_weighted_mean(
    elements: list[Element],
    oxidation_guess: dict[str, float],
    include_positive: bool,
    value_fn: Callable[[Element], float | None],
) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for element in elements:
        oxi_state = oxidation_guess.get(element.symbol, 0.0)
        if include_positive and oxi_state <= 0.0:
            continue
        if not include_positive and oxi_state >= 0.0:
            continue
        value = value_fn(element)
        if value is None:
            continue
        weight = abs(oxi_state)
        weighted_sum += weight * value
        weight_sum += weight
    return weighted_sum / weight_sum if weight_sum > 0.0 else 0.0


def _charge_fraction(
    elements: list[Element],
    fractions: list[float],
    oxidation_guess: dict[str, float],
    include_positive: bool,
    predicate: Callable[[Element], bool],
) -> float:
    total_charge = 0.0
    selected_charge = 0.0
    for element, fraction in zip(elements, fractions, strict=True):
        oxi_state = oxidation_guess.get(element.symbol, 0.0)
        if include_positive and oxi_state <= 0.0:
            continue
        if not include_positive and oxi_state >= 0.0:
            continue
        charge = fraction * abs(oxi_state)
        total_charge += charge
        if predicate(element):
            selected_charge += charge
    return selected_charge / total_charge if total_charge > 0.0 else 0.0


def charge_orbital_identity_family_spectrum(composition: Composition) -> list[float]:
    features = charge_orbital_identity_family(composition)
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    elements = [element for element, _ in element_amounts]
    fractions = [float(amount) / total_atoms for _, amount in element_amounts]
    oxidation_guess = _first_oxidation_guess(composition)

    def orbital_value(key: str) -> Callable[[Element], float | None]:
        return lambda element: _valence_orbital_counts(element)[key]

    value_functions: tuple[Callable[[Element], float | None], ...] = (
        lambda element: _element_property(element, "Z"),
        lambda element: _element_property(element, "row"),
        lambda element: _element_property(element, "group"),
        lambda element: _element_property(element, "mendeleev_no"),
        lambda element: _element_property(element, "X"),
        lambda element: _element_property(element, "atomic_radius"),
        orbital_value("p"),
        orbital_value("d"),
    )
    for include_positive in (True, False):
        for value_fn in value_functions:
            features.append(_charge_weighted_mean(elements, oxidation_guess, include_positive, value_fn))

    positive_alkali_alkaline = _charge_fraction(
        elements,
        fractions,
        oxidation_guess,
        True,
        lambda element: bool(element.is_alkali) or bool(element.is_alkaline),
    )
    positive_transition = _charge_fraction(
        elements,
        fractions,
        oxidation_guess,
        True,
        lambda element: bool(element.is_transition_metal),
    )
    positive_heavy_p = _charge_fraction(
        elements,
        fractions,
        oxidation_guess,
        True,
        lambda element: element.block == "p" and element.Z >= 49,
    )
    positive_rare = _charge_fraction(
        elements,
        fractions,
        oxidation_guess,
        True,
        lambda element: bool(getattr(element, "is_lanthanoid", False))
        or bool(getattr(element, "is_actinoid", False)),
    )
    negative_oxygen = _charge_fraction(
        elements, fractions, oxidation_guess, False, lambda element: element.symbol == "O"
    )
    negative_nitrogen = _charge_fraction(
        elements, fractions, oxidation_guess, False, lambda element: element.symbol == "N"
    )
    negative_halogen = _charge_fraction(
        elements, fractions, oxidation_guess, False, lambda element: element.group == 17
    )
    negative_heavy_chalcogen = _charge_fraction(
        elements,
        fractions,
        oxidation_guess,
        False,
        lambda element: element.symbol in {"S", "Se", "Te"},
    )
    negative_heavy_halogen = _charge_fraction(
        elements,
        fractions,
        oxidation_guess,
        False,
        lambda element: element.symbol in {"Br", "I"},
    )
    features.extend(
        [
            0.0 if oxidation_guess else 1.0,
            positive_alkali_alkaline,
            positive_transition,
            positive_heavy_p,
            positive_rare,
            negative_oxygen,
            negative_nitrogen,
            negative_halogen,
            negative_heavy_chalcogen,
            negative_heavy_halogen,
            positive_heavy_p * (negative_heavy_chalcogen + negative_heavy_halogen),
            positive_transition * (negative_oxygen + negative_heavy_chalcogen),
        ]
    )
    return [float(value) for value in features]


TARGETED_ELEMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Ti", "O"),
    ("Zn", "O"),
    ("Cu", "O"),
    ("Fe", "O"),
    ("Bi", "O"),
    ("Pb", "O"),
    ("Cd", "S"),
    ("Cd", "Se"),
    ("Cd", "Te"),
    ("Pb", "S"),
    ("Pb", "Se"),
    ("Pb", "Te"),
    ("Bi", "S"),
    ("Bi", "Se"),
    ("Bi", "Te"),
    ("Ag", "Te"),
    ("Cu", "S"),
    ("Cu", "Se"),
    ("Pb", "I"),
    ("Sn", "I"),
)


def charge_orbital_identity_family_spectrum_pairs(composition: Composition) -> list[float]:
    features = charge_orbital_identity_family_spectrum(composition)
    element_amounts = composition.element_composition.get_el_amt_dict()
    total_atoms = float(sum(element_amounts.values()))
    fractions = {
        symbol: float(amount) / total_atoms
        for symbol, amount in element_amounts.items()
    }
    for left, right in TARGETED_ELEMENT_PAIRS:
        features.append(fractions.get(left, 0.0) * fractions.get(right, 0.0))
    return [float(value) for value in features]


def charge_orbital_identity_family_spectrum_pair_summary(composition: Composition) -> list[float]:
    features = charge_orbital_identity_family_spectrum_pairs(composition)
    element_amounts = composition.element_composition.get_el_amt_dict()
    total_atoms = float(sum(element_amounts.values()))
    fractions = {
        symbol: float(amount) / total_atoms
        for symbol, amount in element_amounts.items()
    }
    pair_products = [
        fractions.get(left, 0.0) * fractions.get(right, 0.0)
        for left, right in TARGETED_ELEMENT_PAIRS
    ]
    features.extend(
        [
            sum(pair_products),
            max(pair_products, default=0.0),
            float(sum(1 for value in pair_products if value > 0.0)),
        ]
    )
    return [float(value) for value in features]


def _simple_valence_proxy(element: Element) -> float:
    if element.group <= 2:
        return float(element.group)
    if element.group >= 13:
        return float(element.group - 10)
    return 0.0


def _valence_concentration_features(composition: Composition) -> list[float]:
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    elements = [element for element, _ in element_amounts]
    fractions = [float(amount) / total_atoms for _, amount in element_amounts]
    values = [_simple_valence_proxy(element) for element in elements]
    features = _weighted_numeric_stats(values, fractions)
    reduced_amounts = list(composition.reduced_composition.element_composition.items())
    total_proxy = sum(_simple_valence_proxy(element) * float(amount) for element, amount in reduced_amounts)
    reduced_atoms = float(composition.reduced_composition.num_atoms)
    per_atom = total_proxy / reduced_atoms if reduced_atoms > 0.0 else 0.0
    rounded_total = int(round(total_proxy))
    features.extend(
        [
            total_proxy,
            per_atom,
            abs(per_atom - 4.0),
            abs(per_atom - 6.0),
            abs(per_atom - 8.0),
            sum(fraction for value, fraction in zip(values, fractions, strict=True) if value == 0.0),
            sum(fraction for value, fraction in zip(values, fractions, strict=True) if 1.0 <= value <= 2.0),
            sum(fraction for value, fraction in zip(values, fractions, strict=True) if 3.0 <= value <= 5.0),
            sum(fraction for value, fraction in zip(values, fractions, strict=True) if 6.0 <= value <= 8.0),
            1.0 if rounded_total % 2 == 0 else 0.0,
            1.0 if rounded_total % 8 == 0 else 0.0,
        ]
    )
    return [float(value) for value in features]


def charge_orbital_identity_family_spectrum_valence(composition: Composition) -> list[float]:
    features = charge_orbital_identity_family_spectrum(composition)
    features.extend(_valence_concentration_features(composition))
    return [float(value) for value in features]


def _common_oxidation_features(element: Element) -> dict[str, float] | None:
    states = tuple(float(state) for state in getattr(element, "common_oxidation_states", ()))
    if not states:
        return None
    has_positive = any(state > 0.0 for state in states)
    has_negative = any(state < 0.0 for state in states)
    return {
        "count": float(len(states)),
        "range": max(states) - min(states),
        "max_abs": max(abs(state) for state in states),
        "has_positive": 1.0 if has_positive else 0.0,
        "has_negative": 1.0 if has_negative else 0.0,
        "mixed_sign": 1.0 if has_positive and has_negative else 0.0,
    }


def _compact_oxidation_flex_features(composition: Composition) -> list[float]:
    element_amounts = list(composition.element_composition.items())
    total_atoms = float(sum(amount for _, amount in element_amounts))
    collected = []
    weights = []
    missing_fraction = 0.0
    for element, amount in element_amounts:
        fraction = float(amount) / total_atoms
        values = _common_oxidation_features(element)
        if values is None:
            missing_fraction += fraction
        else:
            collected.append(values)
            weights.append(fraction)
    if not collected:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    return [
        sum(weight * values["count"] for values, weight in zip(collected, weights, strict=True)),
        sum(weight * values["range"] for values, weight in zip(collected, weights, strict=True)),
        sum(weight * values["max_abs"] for values, weight in zip(collected, weights, strict=True)),
        sum(weight * values["has_positive"] for values, weight in zip(collected, weights, strict=True)),
        sum(weight * values["has_negative"] for values, weight in zip(collected, weights, strict=True)),
        sum(weight * values["mixed_sign"] for values, weight in zip(collected, weights, strict=True)),
        missing_fraction,
    ]


def charge_orbital_identity_family_spectrum_valence_oxidation_flex(
    composition: Composition,
) -> list[float]:
    features = charge_orbital_identity_family_spectrum_valence(composition)
    features.extend(_compact_oxidation_flex_features(composition))
    return [float(value) for value in features]
