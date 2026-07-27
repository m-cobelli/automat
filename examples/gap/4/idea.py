from __future__ import annotations

import math
import warnings
from itertools import product

import numpy as np
from pymatgen.core import Composition, Element


def _finite_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _safe_property(getter, element: Element) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return _finite_float(getter(element))
        except Exception:
            return 0.0


def _weighted_stats(values: list[float], weights: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.dot(weights, arr))
    variance = float(np.dot(weights, (arr - mean) ** 2))
    std = math.sqrt(max(variance, 0.0))
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    return [mean, std, min_value, max_value, max_value - min_value]


def _basic_composition(composition: Composition) -> tuple[list[Element], np.ndarray]:
    element_amounts = composition.element_composition.get_el_amt_dict()
    elements = [Element(symbol) for symbol in element_amounts]
    amounts = np.asarray([element_amounts[element.symbol] for element in elements], dtype=float)
    return elements, amounts / float(np.sum(amounts))


def _common_oxidation_features(element: Element) -> tuple[float, float, float, float]:
    states = tuple(_finite_float(state) for state in element.common_oxidation_states)
    if not states:
        return 0.0, 0.0, 0.0, 0.0
    positives = [state for state in states if state > 0]
    negatives = [state for state in states if state < 0]
    max_positive = max(positives) if positives else 0.0
    min_negative = min(negatives) if negatives else 0.0
    return max_positive, min_negative, max(states) - min(states), float(len(states))


def baseline(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)

    features: list[float] = [
        float(len(elements)),
        _finite_float(composition.num_atoms),
        float(-np.sum(fractions * np.log(fractions))),
        float(np.sum(fractions**2)),
        float(np.sum(fractions**3)),
        float(np.max(fractions)),
        float(np.min(fractions)),
    ]

    property_getters = [
        lambda element: element.Z,
        lambda element: element.atomic_mass,
        lambda element: element.X,
        lambda element: element.row,
        lambda element: element.group,
        lambda element: element.atomic_radius,
        lambda element: element.atomic_radius_calculated,
        lambda element: element.mendeleev_no,
        lambda element: element.average_ionic_radius,
    ]
    for getter in property_getters:
        values = [_safe_property(getter, element) for element in elements]
        features.extend(_weighted_stats(values, fractions))

    for block in ("s", "p", "d", "f"):
        features.append(
            float(sum(weight for element, weight in zip(elements, fractions) if element.block == block))
        )

    metal_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.is_metal)
    )
    metalloid_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.is_metalloid)
    )
    features.extend([metal_fraction, metalloid_fraction, 1.0 - metal_fraction - metalloid_fraction])

    oxidation_columns = list(zip(*[_common_oxidation_features(element) for element in elements]))
    for values in oxidation_columns:
        features.extend(_weighted_stats(list(values), fractions))

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite baseline features for composition {composition}")
    return features


def _pairwise_difference_features(
    values: list[float],
    elements: list[Element],
    fractions: np.ndarray,
) -> list[float]:
    del elements
    if len(values) < 2:
        return [0.0, 0.0, 0.0]

    weighted_sum = 0.0
    weighted_square_sum = 0.0
    weight_total = 0.0
    max_difference = 0.0
    for i, first in enumerate(values):
        for j in range(i + 1, len(values)):
            difference = abs(first - values[j])
            weight = float(fractions[i] * fractions[j])
            weighted_sum += weight * difference
            weighted_square_sum += weight * difference**2
            weight_total += weight
            max_difference = max(max_difference, difference)
    if weight_total == 0.0:
        return [0.0, 0.0, max_difference]
    return [weighted_sum / weight_total, math.sqrt(weighted_square_sum / weight_total), max_difference]


def _family_fractions(elements: list[Element], fractions: np.ndarray) -> list[float]:
    families = [
        lambda element: element.group == 1 and element.symbol != "H",
        lambda element: element.group == 2,
        lambda element: 3 <= element.group <= 12,
        lambda element: element.block == "p" and element.is_metal,
        lambda element: element.is_metalloid,
        lambda element: element.group == 15,
        lambda element: element.group == 16,
        lambda element: element.group == 17,
        lambda element: element.group == 18,
        lambda element: 57 <= element.Z <= 71,
        lambda element: 89 <= element.Z <= 103,
    ]
    return [
        float(sum(weight for element, weight in zip(elements, fractions) if family(element)))
        for family in families
    ]


def gap_contrast_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = baseline(composition)

    property_getters = [
        lambda element: element.X,
        lambda element: element.atomic_radius,
        lambda element: element.atomic_radius_calculated,
        lambda element: element.row,
        lambda element: element.group,
        lambda element: element.mendeleev_no,
        lambda element: element.Z,
    ]
    property_columns = [
        [_safe_property(getter, element) for element in elements] for getter in property_getters
    ]
    for values in property_columns:
        features.extend(_pairwise_difference_features(values, elements, fractions))

    electronegativities = property_columns[0]
    max_en = max(electronegativities)
    anion_mask = np.asarray([value == max_en for value in electronegativities], dtype=bool)
    anion_fraction = float(np.sum(fractions[anion_mask]))
    cation_fraction = 1.0 - anion_fraction
    features.append(anion_fraction)

    for values in property_columns[:6]:
        arr = np.asarray(values, dtype=float)
        anion_average = float(np.dot(fractions[anion_mask], arr[anion_mask]) / anion_fraction)
        if cation_fraction > 0.0:
            cation_average = float(np.dot(fractions[~anion_mask], arr[~anion_mask]) / cation_fraction)
        else:
            cation_average = anion_average
        features.append(anion_average - cation_average)

    family_values = _family_fractions(elements, fractions)
    features.extend(family_values)

    metal_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.is_metal)
    )
    metalloid_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.is_metalloid)
    )
    nonmetal_fraction = 1.0 - metal_fraction - metalloid_fraction
    en_mean_contrast, en_rms_contrast, en_max_contrast = _pairwise_difference_features(
        electronegativities, elements, fractions
    )
    radius_mean_contrast, radius_rms_contrast, radius_max_contrast = _pairwise_difference_features(
        property_columns[1], elements, fractions
    )
    features.extend(
        [
            metal_fraction * nonmetal_fraction,
            metal_fraction * en_mean_contrast,
            nonmetal_fraction * en_mean_contrast,
            anion_fraction * cation_fraction * en_max_contrast,
            anion_fraction * cation_fraction * radius_max_contrast,
            en_rms_contrast * radius_rms_contrast,
            family_values[6] * metal_fraction,
            family_values[7] * metal_fraction,
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_contrast_v1 features for composition {composition}")
    return features


def _limited_common_states(element: Element) -> tuple[float, ...]:
    states = tuple(_finite_float(state) for state in element.common_oxidation_states)
    if not states:
        return (0.0,)
    ordered = sorted(states, key=lambda state: (abs(state), state))
    return tuple(ordered[:5])


def _charge_balance_features(elements: list[Element], fractions: np.ndarray) -> list[float]:
    state_options = [_limited_common_states(element) for element in elements]
    best_abs_charge = float("inf")
    near_neutral = 0.0
    exact_neutral = 0.0
    checked = 0

    for state_combo in product(*state_options):
        checked += 1
        if checked > 400:
            break
        charge = abs(float(np.dot(fractions, np.asarray(state_combo, dtype=float))))
        best_abs_charge = min(best_abs_charge, charge)
        if charge <= 0.25:
            near_neutral = 1.0
        if charge <= 1e-8:
            exact_neutral = 1.0
            near_neutral = 1.0

    if not math.isfinite(best_abs_charge):
        best_abs_charge = 0.0
    return [best_abs_charge, near_neutral, exact_neutral, float(checked)]


def gap_oxidation_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_contrast_v1(composition)

    oxidation_values = [_common_oxidation_features(element) for element in elements]
    positive_states, negative_states, spans, state_counts = zip(*oxidation_values)
    for values in (positive_states, negative_states, spans, state_counts):
        features.extend(_weighted_stats(list(values), fractions))

    features.extend(_charge_balance_features(elements, fractions))

    positive_capacity = float(np.dot(fractions, np.asarray(positive_states, dtype=float)))
    negative_capacity = abs(float(np.dot(fractions, np.asarray(negative_states, dtype=float))))
    charge_ratio = positive_capacity / (negative_capacity + 1.0e-6)

    electronegativities = [_safe_property(lambda element: element.X, element) for element in elements]
    en_mean_contrast, en_rms_contrast, en_max_contrast = _pairwise_difference_features(
        electronegativities, elements, fractions
    )

    metal_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.is_metal)
    )
    transition_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if 3 <= element.group <= 12)
    )

    family_sets = [
        {"O", "S", "Se", "Te"},
        {"N", "P", "As", "Sb"},
        {"F", "Cl", "Br", "I"},
        {"C", "B", "Si"},
    ]
    family_fractions = [
        float(sum(weight for element, weight in zip(elements, fractions) if element.symbol in family))
        for family in family_sets
    ]

    features.extend(
        [
            positive_capacity,
            negative_capacity,
            charge_ratio,
            positive_capacity * negative_capacity,
            positive_capacity * en_mean_contrast,
            negative_capacity * en_mean_contrast,
            en_max_contrast * (1.0 - metal_fraction),
            en_rms_contrast * transition_fraction,
        ]
    )
    features.extend(family_fractions)
    for value in family_fractions:
        features.extend([value * metal_fraction, value * transition_fraction, value * en_max_contrast])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_oxidation_v1 features for composition {composition}")
    return features


def gap_oxidation_compact_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = baseline(composition)

    oxidation_values = [_common_oxidation_features(element) for element in elements]
    positive_states, negative_states, spans, state_counts = zip(*oxidation_values)
    for values in (positive_states, negative_states, spans, state_counts):
        features.extend(_weighted_stats(list(values), fractions))

    features.extend(_charge_balance_features(elements, fractions))

    positive_capacity = float(np.dot(fractions, np.asarray(positive_states, dtype=float)))
    negative_capacity = abs(float(np.dot(fractions, np.asarray(negative_states, dtype=float))))
    charge_ratio = positive_capacity / (negative_capacity + 1.0e-6)

    metal_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.is_metal)
    )
    transition_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if 3 <= element.group <= 12)
    )
    family_sets = [
        {"O", "S", "Se", "Te"},
        {"N", "P", "As", "Sb"},
        {"F", "Cl", "Br", "I"},
        {"C", "B", "Si"},
    ]
    family_fractions = [
        float(sum(weight for element, weight in zip(elements, fractions) if element.symbol in family))
        for family in family_sets
    ]

    features.extend(
        [
            positive_capacity,
            negative_capacity,
            charge_ratio,
            positive_capacity * negative_capacity,
            metal_fraction,
            transition_fraction,
        ]
    )
    features.extend(family_fractions)
    for value in family_fractions:
        features.extend([value * metal_fraction, value * transition_fraction])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_oxidation_compact_v1 features for composition {composition}")
    return features


def gap_element_fraction_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_oxidation_compact_v1(composition)

    element_fraction_by_z = [0.0] * 118
    for element, fraction in zip(elements, fractions):
        if 1 <= element.Z <= 118:
            element_fraction_by_z[element.Z - 1] = float(fraction)
    features.extend(element_fraction_by_z)

    oxygen_fraction = element_fraction_by_z[7]
    sulfur_fraction = element_fraction_by_z[15]
    nitrogen_fraction = element_fraction_by_z[6]
    halogen_fraction = sum(element_fraction_by_z[z - 1] for z in (9, 17, 35, 53))
    transition_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if 3 <= element.group <= 12)
    )
    concentration_l2 = float(np.sum(fractions**2))
    max_fraction = float(np.max(fractions))

    features.extend(
        [
            oxygen_fraction,
            sulfur_fraction,
            nitrogen_fraction,
            halogen_fraction,
            transition_fraction,
            concentration_l2,
            max_fraction,
            oxygen_fraction * transition_fraction,
            sulfur_fraction * transition_fraction,
            nitrogen_fraction * transition_fraction,
            halogen_fraction * transition_fraction,
            concentration_l2 * transition_fraction,
            max_fraction * (oxygen_fraction + sulfur_fraction + nitrogen_fraction + halogen_fraction),
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_element_fraction_v1 features for composition {composition}")
    return features


def _padded(values: list[float], length: int) -> list[float]:
    return values[:length] + [0.0] * max(0, length - len(values))


def _normalized_integer_pattern(composition: Composition) -> list[int]:
    reduced = composition.reduced_composition
    amounts = sorted((int(round(amount)) for amount in reduced.get_el_amt_dict().values()), reverse=True)
    return amounts


def gap_stoich_pattern_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_element_fraction_v1(composition)

    sorted_fractions = sorted((float(value) for value in fractions), reverse=True)
    features.extend(_padded(sorted_fractions, 8))

    integer_pattern = _normalized_integer_pattern(composition)
    features.extend(_padded([float(value) for value in integer_pattern], 8))
    reduced_total = float(sum(integer_pattern))
    max_reduced = float(max(integer_pattern))
    min_reduced = float(min(integer_pattern))
    reduced_l2 = math.sqrt(float(sum(value * value for value in integer_pattern)))
    features.extend([reduced_total, max_reduced, min_reduced, reduced_l2])

    ascending = sorted(integer_pattern)
    pattern_targets = [
        [1],
        [1, 1],
        [1, 2],
        [2, 3],
        [1, 3],
        [1, 1, 1],
        [1, 1, 2],
        [1, 1, 3],
        [1, 2, 4],
        [1, 1, 1, 1],
    ]
    features.extend([1.0 if ascending == target else 0.0 for target in pattern_targets])

    element_count = len(elements)
    features.extend([1.0 if element_count == count else 0.0 for count in range(1, 6)])
    features.append(1.0 if element_count >= 6 else 0.0)

    charge_residual = _charge_balance_features(elements, fractions)[0]
    element_fraction_by_z = [0.0] * 118
    for element, fraction in zip(elements, fractions):
        if 1 <= element.Z <= 118:
            element_fraction_by_z[element.Z - 1] = float(fraction)
    oxygen_fraction = element_fraction_by_z[7]
    halogen_fraction = sum(element_fraction_by_z[z - 1] for z in (9, 17, 35, 53))
    transition_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if 3 <= element.group <= 12)
    )
    features.extend(
        [
            reduced_total * oxygen_fraction,
            reduced_total * halogen_fraction,
            reduced_total * transition_fraction,
            charge_residual * reduced_total,
            charge_residual * oxygen_fraction,
            charge_residual * halogen_fraction,
            max_reduced / (reduced_total + 1.0e-6),
            min_reduced / (reduced_total + 1.0e-6),
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_stoich_pattern_v1 features for composition {composition}")
    return features


def _electron_configuration_features(element: Element) -> list[float]:
    structure = element.full_electronic_structure
    outer_shell = max(shell for shell, _, _ in structure)
    shell_totals = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}
    outer_totals = {"s": 0.0, "p": 0.0, "d": 0.0, "f": 0.0}
    highest_d = 0.0
    highest_f = 0.0
    d_shell = -1
    f_shell = -1
    for shell, orbital, electrons in structure:
        count = float(electrons)
        if orbital in shell_totals:
            shell_totals[orbital] += count
        if shell == outer_shell and orbital in outer_totals:
            outer_totals[orbital] += count
        if orbital == "d" and shell >= d_shell:
            d_shell = shell
            highest_d = count
        if orbital == "f" and shell >= f_shell:
            f_shell = shell
            highest_f = count
    outer_electrons = sum(outer_totals.values())
    open_d = min(highest_d, max(0.0, 10.0 - highest_d))
    open_f = min(highest_f, max(0.0, 14.0 - highest_f))
    return [
        float(outer_shell),
        outer_electrons,
        shell_totals["s"],
        shell_totals["p"],
        shell_totals["d"],
        shell_totals["f"],
        outer_totals["s"],
        outer_totals["p"],
        highest_d,
        highest_f,
        open_d,
        open_f,
    ]


def gap_valence_orbital_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_stoich_pattern_v1(composition)

    columns = list(zip(*[_electron_configuration_features(element) for element in elements]))
    for values in columns:
        features.extend(_weighted_stats(list(values), fractions))

    element_fraction_by_z = [0.0] * 118
    for element, fraction in zip(elements, fractions):
        if 1 <= element.Z <= 118:
            element_fraction_by_z[element.Z - 1] = float(fraction)
    oxygen_fraction = element_fraction_by_z[7]
    halogen_fraction = sum(element_fraction_by_z[z - 1] for z in (9, 17, 35, 53))
    transition_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if 3 <= element.group <= 12)
    )
    charge_residual = _charge_balance_features(elements, fractions)[0]
    electron_features = [_electron_configuration_features(element) for element in elements]
    outer_electron_mean = float(np.dot(fractions, np.asarray([row[1] for row in electron_features])))
    d_open_mean = float(np.dot(fractions, np.asarray([row[10] for row in electron_features])))
    f_open_mean = float(np.dot(fractions, np.asarray([row[11] for row in electron_features])))
    outer_p_mean = float(np.dot(fractions, np.asarray([row[7] for row in electron_features])))
    features.extend(
        [
            outer_electron_mean * oxygen_fraction,
            outer_electron_mean * halogen_fraction,
            d_open_mean * transition_fraction,
            f_open_mean,
            outer_p_mean * (oxygen_fraction + halogen_fraction),
            d_open_mean * charge_residual,
            transition_fraction * charge_residual,
            (d_open_mean + f_open_mean) * (1.0 - oxygen_fraction),
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_valence_orbital_v1 features for composition {composition}")
    return features


def gap_periodic_grid_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_valence_orbital_v1(composition)

    group_fractions = [0.0] * 18
    period_fractions = [0.0] * 7
    for element, fraction in zip(elements, fractions):
        if 1 <= element.group <= 18:
            group_fractions[element.group - 1] += float(fraction)
        if 1 <= element.row <= 7:
            period_fractions[element.row - 1] += float(fraction)
    features.extend(group_fractions)
    features.extend(period_fractions)

    p_block_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.block == "p")
    )
    d_block_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.block == "d")
    )
    transition_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if 3 <= element.group <= 12)
    )
    group16_fraction = group_fractions[15]
    group17_fraction = group_fractions[16]
    period2_fraction = period_fractions[1]
    period3_fraction = period_fractions[2]
    heavy_period_fraction = sum(period_fractions[4:])
    features.extend(
        [
            p_block_fraction * d_block_fraction,
            transition_fraction * group16_fraction,
            transition_fraction * group17_fraction,
            period2_fraction * group16_fraction,
            period3_fraction * group16_fraction,
            heavy_period_fraction * group17_fraction,
            heavy_period_fraction * transition_fraction,
            (group16_fraction + group17_fraction) * (period2_fraction + period3_fraction),
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_periodic_grid_v1 features for composition {composition}")
    return features


def gap_charge_anion_focus_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_periodic_grid_v1(composition)

    group16_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 16)
    )
    group17_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 17)
    )
    charge_residual = _charge_balance_features(elements, fractions)[0]
    features.extend([charge_residual * group16_fraction, charge_residual * group17_fraction])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_charge_anion_focus_v1 features for composition {composition}")
    return features


def gap_charge_residual_sqrt_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_periodic_grid_v1(composition)

    group16_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 16)
    )
    group17_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 17)
    )
    charge_residual = _charge_balance_features(elements, fractions)[0]
    residual_sqrt = math.sqrt(max(charge_residual, 0.0))
    features.extend([residual_sqrt * group16_fraction, residual_sqrt * group17_fraction])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_charge_residual_sqrt_v1 features for composition {composition}")
    return features


def gap_charge_residual_fourthroot_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_periodic_grid_v1(composition)

    group16_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 16)
    )
    group17_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 17)
    )
    charge_residual = _charge_balance_features(elements, fractions)[0]
    residual_root = math.sqrt(math.sqrt(max(charge_residual, 0.0)))
    features.extend([residual_root * group16_fraction, residual_root * group17_fraction])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_charge_residual_fourthroot_v1 features for composition {composition}")
    return features


def gap_charge_residual_cuberoot_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_periodic_grid_v1(composition)

    group16_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 16)
    )
    group17_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 17)
    )
    charge_residual = _charge_balance_features(elements, fractions)[0]
    residual_root = max(charge_residual, 0.0) ** (1.0 / 3.0)
    features.extend([residual_root * group16_fraction, residual_root * group17_fraction])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_charge_residual_cuberoot_v1 features for composition {composition}")
    return features


def gap_charge_oxygen_halogen_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_periodic_grid_v1(composition)

    oxygen_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.symbol == "O")
    )
    halogen_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.symbol in {"F", "Cl", "Br", "I"})
    )
    charge_residual = _charge_balance_features(elements, fractions)[0]
    residual_root = max(charge_residual, 0.0) ** (1.0 / 3.0)
    features.extend([residual_root * oxygen_fraction, residual_root * halogen_fraction])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_charge_oxygen_halogen_v1 features for composition {composition}")
    return features


def gap_charge_oxygen_pnictogen_v1(composition: Composition) -> list[float]:
    elements, fractions = _basic_composition(composition)
    features = gap_periodic_grid_v1(composition)

    oxygen_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.symbol == "O")
    )
    pnictogen_fraction = float(
        sum(weight for element, weight in zip(elements, fractions) if element.group == 15)
    )
    charge_residual = _charge_balance_features(elements, fractions)[0]
    residual_root = max(charge_residual, 0.0) ** (1.0 / 3.0)
    features.extend([residual_root * oxygen_fraction, residual_root * pnictogen_fraction])

    if not all(math.isfinite(value) for value in features):
        raise ValueError(f"Non-finite gap_charge_oxygen_pnictogen_v1 features for composition {composition}")
    return features
