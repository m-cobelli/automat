from __future__ import annotations

import math

from pymatgen.core import Composition, Element


_RARE_EARTHS = {
    "Sc",
    "Y",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
}

_HEAVY_RARE_EARTHS = {"Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
_THREE_D_TRANSITION = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"}
_CANONICAL_FERROMAGNETS = {"Fe", "Co", "Ni", "Gd"}


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _weighted_stats(values: list[float], weights: list[float]) -> list[float]:
    mean = sum(w * v for w, v in zip(weights, values))
    variance = sum(w * (v - mean) ** 2 for w, v in zip(weights, values))
    std = math.sqrt(max(variance, 0.0))
    min_value = min(values)
    max_value = max(values)
    min_fraction = sum(w for w, v in zip(weights, values) if v == min_value)
    max_fraction = sum(w for w, v in zip(weights, values) if v == max_value)
    return [
        mean,
        std,
        min_value,
        max_value,
        max_value - min_value,
        min_fraction,
        max_fraction,
    ]


def _fraction(elements: list[Element], weights: list[float], predicate) -> float:
    return sum(weight for element, weight in zip(elements, weights) if predicate(element))


def baseline_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    weights = [amount / total_atoms for amount in element_amounts.values()]

    entropy = -sum(weight * math.log(weight) for weight in weights if weight > 0.0)
    herfindahl = sum(weight**2 for weight in weights)

    features = [
        float(len(elements)),
        float(total_atoms),
        max(weights),
        entropy,
        herfindahl,
    ]

    numeric_properties = [
        [float(element.Z) for element in elements],
        [_as_float(element.atomic_mass) for element in elements],
        [_as_float(element.mendeleev_no) for element in elements],
        [float(element.row) for element in elements],
        [float(element.group) for element in elements],
        [_as_float(element.X) for element in elements],
        [_as_float(element.atomic_radius) for element in elements],
    ]
    for values in numeric_properties:
        features.extend(_weighted_stats(values, weights))

    block_fractions = [
        _fraction(elements, weights, lambda element, block=block: element.block == block)
        for block in ("s", "p", "d", "f")
    ]
    chemical_family_fractions = [
        _fraction(elements, weights, lambda element: bool(element.is_metal)),
        _fraction(elements, weights, lambda element: bool(element.is_metalloid)),
        _fraction(elements, weights, lambda element: not bool(element.is_metal) and not bool(element.is_metalloid)),
        _fraction(elements, weights, lambda element: bool(element.is_transition_metal)),
        _fraction(elements, weights, lambda element: bool(element.is_alkali)),
        _fraction(elements, weights, lambda element: bool(element.is_alkaline)),
        _fraction(elements, weights, lambda element: bool(element.is_halogen)),
        _fraction(elements, weights, lambda element: bool(element.is_chalcogen)),
    ]
    features.extend(block_fractions)
    features.extend(chemical_family_fractions)

    symbol_fractions = {
        element.symbol: weight for element, weight in zip(elements, weights)
    }
    fe_fraction = symbol_fractions.get("Fe", 0.0)
    co_fraction = symbol_fractions.get("Co", 0.0)
    ni_fraction = symbol_fractions.get("Ni", 0.0)
    mn_fraction = symbol_fractions.get("Mn", 0.0)
    cr_fraction = symbol_fractions.get("Cr", 0.0)
    rare_earth_fraction = _fraction(elements, weights, lambda element: element.symbol in _RARE_EARTHS)
    heavy_rare_earth_fraction = _fraction(
        elements, weights, lambda element: element.symbol in _HEAVY_RARE_EARTHS
    )
    three_d_fraction = _fraction(
        elements, weights, lambda element: element.symbol in _THREE_D_TRANSITION
    )
    canonical_ferromagnet_fraction = _fraction(
        elements, weights, lambda element: element.symbol in _CANONICAL_FERROMAGNETS
    )

    magnetic_fractions = [
        fe_fraction,
        co_fraction,
        ni_fraction,
        mn_fraction,
        cr_fraction,
        rare_earth_fraction,
        heavy_rare_earth_fraction,
        three_d_fraction,
        canonical_ferromagnet_fraction,
    ]
    features.extend(magnetic_fractions)

    mean_atomic_number = numeric_properties[0] and _weighted_stats(numeric_properties[0], weights)[0]
    mean_electronegativity = numeric_properties[5] and _weighted_stats(numeric_properties[5], weights)[0]
    magnetic_total = three_d_fraction + rare_earth_fraction
    features.extend(
        [
            magnetic_total * entropy,
            magnetic_total * mean_atomic_number,
            magnetic_total * mean_electronegativity,
            canonical_ferromagnet_fraction * entropy,
            canonical_ferromagnet_fraction * mean_atomic_number,
            canonical_ferromagnet_fraction * mean_electronegativity,
            three_d_fraction * rare_earth_fraction,
            fe_fraction + co_fraction + ni_fraction,
            (fe_fraction + co_fraction + ni_fraction) * (1.0 - herfindahl),
        ]
    )

    return [float(value) for value in features]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / (denominator + 1.0e-6)


def magnetic_sublattice_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    weights = [amount / total_atoms for amount in element_amounts.values()]
    symbol_fractions = {
        element.symbol: weight for element, weight in zip(elements, weights)
    }

    fe = symbol_fractions.get("Fe", 0.0)
    co = symbol_fractions.get("Co", 0.0)
    ni = symbol_fractions.get("Ni", 0.0)
    mn = symbol_fractions.get("Mn", 0.0)
    cr = symbol_fractions.get("Cr", 0.0)
    gd = symbol_fractions.get("Gd", 0.0)
    fecn = fe + co + ni
    mncr = mn + cr
    three_d_magnetic = fecn + mncr
    rare_earth = _fraction(elements, weights, lambda element: element.symbol in _RARE_EARTHS)
    heavy_rare_earth = _fraction(elements, weights, lambda element: element.symbol in _HEAVY_RARE_EARTHS)
    three_d_transition = _fraction(
        elements, weights, lambda element: element.symbol in _THREE_D_TRANSITION
    )
    transition = _fraction(elements, weights, lambda element: bool(element.is_transition_metal))
    metalloid = _fraction(elements, weights, lambda element: bool(element.is_metalloid))
    nonmetal = _fraction(
        elements,
        weights,
        lambda element: not bool(element.is_metal) and not bool(element.is_metalloid),
    )
    chalcogen = _fraction(elements, weights, lambda element: bool(element.is_chalcogen))
    halogen = _fraction(elements, weights, lambda element: bool(element.is_halogen))
    candidate_magnetic = three_d_magnetic + rare_earth
    nonmagnetic = max(0.0, 1.0 - candidate_magnetic)

    entropy = -sum(weight * math.log(weight) for weight in weights if weight > 0.0)
    herfindahl = sum(weight**2 for weight in weights)
    electronegativities = [_as_float(element.X) for element in elements]
    radii = [_as_float(element.atomic_radius) for element in elements]
    en_range = max(electronegativities) - min(electronegativities)
    radius_range = max(radii) - min(radii)

    features = baseline_descriptor(composition)
    features.extend(
        [
            fe,
            co,
            ni,
            mn,
            cr,
            gd,
            fecn,
            mncr,
            three_d_magnetic,
            rare_earth,
            heavy_rare_earth,
            three_d_transition,
            transition,
            metalloid,
            nonmetal,
            chalcogen,
            halogen,
            candidate_magnetic,
            nonmagnetic,
            three_d_magnetic + rare_earth,
            _safe_ratio(candidate_magnetic, nonmagnetic),
            _safe_ratio(fecn, 1.0 - fecn),
            _safe_ratio(three_d_magnetic, rare_earth),
            _safe_ratio(rare_earth, three_d_magnetic),
            candidate_magnetic * max(weights),
            candidate_magnetic * entropy,
            candidate_magnetic * herfindahl,
            fecn * nonmetal,
            fecn * chalcogen,
            three_d_magnetic * nonmetal,
            three_d_magnetic * chalcogen,
            rare_earth * nonmetal,
            rare_earth * chalcogen,
            fe * co,
            fe * ni,
            co * ni,
            fe * mn,
            mn * rare_earth,
            three_d_magnetic * rare_earth,
            fecn * rare_earth,
            candidate_magnetic * en_range,
            candidate_magnetic * radius_range,
            fecn * en_range,
            three_d_magnetic * en_range,
            rare_earth * radius_range,
            nonmetal * en_range,
            chalcogen * en_range,
        ]
    )

    return [float(value) for value in features]


def _element_data_float(element: Element, key: str) -> tuple[float, float]:
    value = _as_float(element.data.get(key), default=float("nan"))
    if math.isnan(value):
        return 0.0, 0.0
    return value, 1.0


def _log1p_property(value: float, available: float) -> float:
    if available <= 0.0:
        return 0.0
    return math.log1p(max(value, 0.0))


def thermal_metallicity_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    weights = [amount / total_atoms for amount in element_amounts.values()]
    symbol_fractions = {
        element.symbol: weight for element, weight in zip(elements, weights)
    }

    property_specs = [
        ("Melting point", False),
        ("Boiling point", False),
        ("Thermal conductivity", True),
        ("Electrical resistivity", True),
        ("Bulk modulus", False),
        ("Brinell hardness", False),
        ("Coefficient of linear thermal expansion", True),
    ]

    features = magnetic_sublattice_descriptor(composition)
    property_means: dict[str, float] = {}
    for key, use_log in property_specs:
        values = []
        availability = []
        for element in elements:
            value, is_available = _element_data_float(element, key)
            availability.append(is_available)
            values.append(_log1p_property(value, is_available) if use_log else value)
        features.extend(_weighted_stats(values, weights))
        available_fraction = sum(weight * ok for weight, ok in zip(weights, availability))
        features.append(available_fraction)
        property_means[key] = sum(weight * value for weight, value in zip(weights, values))

    fe_co_ni = sum(
        weight for element, weight in zip(elements, weights) if element.symbol in {"Fe", "Co", "Ni"}
    )
    candidate_magnetic = sum(
        weight
        for element, weight in zip(elements, weights)
        if element.symbol in {"Fe", "Co", "Ni", "Mn", "Cr"} or element.symbol in _RARE_EARTHS
    )
    metal_fraction = _fraction(elements, weights, lambda element: bool(element.is_metal))
    nonmetal_fraction = _fraction(
        elements,
        weights,
        lambda element: not bool(element.is_metal) and not bool(element.is_metalloid),
    )
    melting = property_means["Melting point"]
    conductivity = property_means["Thermal conductivity"]
    resistivity = property_means["Electrical resistivity"]
    bulk = property_means["Bulk modulus"]

    features.extend(
        [
            candidate_magnetic * melting,
            candidate_magnetic * conductivity,
            candidate_magnetic * resistivity,
            candidate_magnetic * bulk,
            fe_co_ni * melting,
            fe_co_ni * conductivity,
            fe_co_ni * resistivity,
            fe_co_ni * bulk,
            metal_fraction * conductivity,
            metal_fraction * bulk,
            nonmetal_fraction * resistivity,
            symbol_fractions.get("Gd", 0.0) * bulk,
        ]
    )

    return [float(value) for value in features]


def targeted_identity_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    weights = [amount / total_atoms for amount in element_amounts.values()]
    symbol_fractions = {
        element.symbol: weight for element, weight in zip(elements, weights)
    }

    selected_symbols = [
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Yb",
        "Lu",
        "B",
        "C",
        "N",
        "O",
        "P",
        "S",
        "Si",
        "Ge",
    ]

    features = thermal_metallicity_descriptor(composition)
    features.extend(symbol_fractions.get(symbol, 0.0) for symbol in selected_symbols)

    early_3d = sum(symbol_fractions.get(symbol, 0.0) for symbol in ("Ti", "V", "Cr", "Mn"))
    late_3d = sum(symbol_fractions.get(symbol, 0.0) for symbol in ("Fe", "Co", "Ni", "Cu"))
    light_rare_earth = sum(
        symbol_fractions.get(symbol, 0.0) for symbol in ("La", "Ce", "Pr", "Nd", "Sm", "Eu")
    )
    heavy_rare_earth = sum(
        symbol_fractions.get(symbol, 0.0) for symbol in ("Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu")
    )
    pnictogen_chalcogen = sum(
        symbol_fractions.get(symbol, 0.0) for symbol in ("N", "P", "O", "S")
    )
    fecn = sum(symbol_fractions.get(symbol, 0.0) for symbol in ("Fe", "Co", "Ni"))
    gd_tb_dy = sum(symbol_fractions.get(symbol, 0.0) for symbol in ("Gd", "Tb", "Dy"))
    oxygen_boron = symbol_fractions.get("O", 0.0) + symbol_fractions.get("B", 0.0)
    transition_magnetic = sum(
        symbol_fractions.get(symbol, 0.0) for symbol in ("Cr", "Mn", "Fe", "Co", "Ni")
    )

    features.extend(
        [
            early_3d,
            late_3d,
            light_rare_earth,
            heavy_rare_earth,
            pnictogen_chalcogen,
            early_3d * late_3d,
            light_rare_earth * late_3d,
            heavy_rare_earth * late_3d,
            gd_tb_dy * fecn,
            symbol_fractions.get("Fe", 0.0) * symbol_fractions.get("Co", 0.0),
            symbol_fractions.get("Fe", 0.0) * symbol_fractions.get("Ni", 0.0),
            symbol_fractions.get("Co", 0.0) * symbol_fractions.get("Ni", 0.0),
            symbol_fractions.get("Mn", 0.0) * late_3d,
            symbol_fractions.get("Cr", 0.0) * late_3d,
            oxygen_boron * transition_magnetic,
            pnictogen_chalcogen * fecn,
            symbol_fractions.get("Gd", 0.0) * symbol_fractions.get("Fe", 0.0),
            symbol_fractions.get("Nd", 0.0) * fecn,
        ]
    )

    return [float(value) for value in features]


def stoichiometric_motif_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    amount = {symbol: float(value) for symbol, value in element_amounts.items()}

    selected_symbols = [
        "Fe",
        "Co",
        "Ni",
        "Mn",
        "Cr",
        "Gd",
        "Tb",
        "Dy",
        "Nd",
        "Sm",
        "B",
        "C",
        "N",
        "O",
        "P",
        "S",
        "Si",
    ]

    features = targeted_identity_descriptor(composition)
    for symbol in selected_symbols:
        value = amount.get(symbol, 0.0)
        features.extend([value, math.log1p(value), value / total_atoms])

    fecn_atoms = sum(amount.get(symbol, 0.0) for symbol in ("Fe", "Co", "Ni"))
    mncr_atoms = sum(amount.get(symbol, 0.0) for symbol in ("Mn", "Cr"))
    transition_magnetic_atoms = fecn_atoms + mncr_atoms
    rare_earth_atoms = sum(amount.get(symbol, 0.0) for symbol in _RARE_EARTHS)
    candidate_magnetic_atoms = transition_magnetic_atoms + rare_earth_atoms
    exchange_partner_atoms = sum(amount.get(symbol, 0.0) for symbol in ("B", "C", "N", "O", "P", "S", "Si"))
    nonmagnetic_atoms = max(0.0, total_atoms - candidate_magnetic_atoms)
    element_count = len(element_amounts)
    largest_amount = max(amount.values())
    smallest_amount = min(amount.values())

    features.extend(
        [
            fecn_atoms,
            mncr_atoms,
            transition_magnetic_atoms,
            rare_earth_atoms,
            candidate_magnetic_atoms,
            exchange_partner_atoms,
            nonmagnetic_atoms,
            _safe_ratio(candidate_magnetic_atoms, nonmagnetic_atoms),
            _safe_ratio(fecn_atoms, exchange_partner_atoms),
            _safe_ratio(rare_earth_atoms, transition_magnetic_atoms),
            _safe_ratio(amount.get("O", 0.0), transition_magnetic_atoms),
            _safe_ratio(largest_amount, smallest_amount),
            1.0 if element_count == 2 else 0.0,
            1.0 if element_count == 3 else 0.0,
            1.0 if element_count >= 4 else 0.0,
            1.0 if largest_amount / total_atoms >= 0.75 else 0.0,
            1.0 if transition_magnetic_atoms > 0.0 and rare_earth_atoms > 0.0 else 0.0,
            math.log1p(total_atoms),
            math.log1p(candidate_magnetic_atoms),
            math.log1p(exchange_partner_atoms),
        ]
    )

    return [float(value) for value in features]


def composition_class_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    weights = [amount / total_atoms for amount in element_amounts.values()]
    fractions = {element.symbol: weight for element, weight in zip(elements, weights)}

    transition = _fraction(elements, weights, lambda element: bool(element.is_transition_metal))
    rare_earth = _fraction(elements, weights, lambda element: element.symbol in _RARE_EARTHS)
    fecn = sum(fractions.get(symbol, 0.0) for symbol in ("Fe", "Co", "Ni"))
    mn = fractions.get("Mn", 0.0)
    cr = fractions.get("Cr", 0.0)
    candidate_magnetic = fecn + mn + cr + rare_earth
    oxygen = fractions.get("O", 0.0)
    boron = fractions.get("B", 0.0)
    carbon = fractions.get("C", 0.0)
    nitrogen = fractions.get("N", 0.0)
    pnictogen = sum(fractions.get(symbol, 0.0) for symbol in ("N", "P", "As", "Sb", "Bi"))
    chalcogen = sum(fractions.get(symbol, 0.0) for symbol in ("O", "S", "Se", "Te"))
    halogen = sum(fractions.get(symbol, 0.0) for symbol in ("F", "Cl", "Br", "I"))
    silicide_germanide = fractions.get("Si", 0.0) + fractions.get("Ge", 0.0)
    nonmetal_light = oxygen + boron + carbon + nitrogen
    metal_fraction = _fraction(elements, weights, lambda element: bool(element.is_metal))

    oxide = 1.0 if oxygen > 0.0 else 0.0
    boride = 1.0 if boron > 0.0 else 0.0
    carbide = 1.0 if carbon > 0.0 else 0.0
    nitride = 1.0 if nitrogen > 0.0 else 0.0
    pnictide = 1.0 if pnictogen > 0.0 else 0.0
    chalcogenide = 1.0 if chalcogen > oxygen else 0.0
    halide = 1.0 if halogen > 0.0 else 0.0
    silicide = 1.0 if silicide_germanide > 0.0 else 0.0
    re_tm = 1.0 if rare_earth > 0.0 and transition > 0.0 else 0.0
    alloy_like = 1.0 if metal_fraction >= 0.9 and fecn > 0.0 else 0.0
    mn_rich = 1.0 if mn >= 0.25 else 0.0
    cr_rich = 1.0 if cr >= 0.25 else 0.0
    flags = [
        oxide,
        boride,
        carbide,
        nitride,
        pnictide,
        chalcogenide,
        halide,
        silicide,
        re_tm,
        alloy_like,
        mn_rich,
        cr_rich,
        1.0 if transition > 0.0 and oxygen > 0.0 else 0.0,
        1.0 if transition > 0.0 and boron > 0.0 else 0.0,
        1.0 if transition > 0.0 and carbon + nitrogen > 0.0 else 0.0,
        1.0 if rare_earth > 0.0 and oxygen > 0.0 else 0.0,
    ]

    features = stoichiometric_motif_descriptor(composition)
    features.extend(flags)
    features.extend(
        [
            oxygen,
            boron,
            carbon,
            nitrogen,
            pnictogen,
            chalcogen,
            halogen,
            silicide_germanide,
            nonmetal_light,
            oxide * candidate_magnetic,
            boride * candidate_magnetic,
            carbide * candidate_magnetic,
            nitride * candidate_magnetic,
            pnictide * candidate_magnetic,
            chalcogenide * candidate_magnetic,
            halide * candidate_magnetic,
            silicide * fecn,
            re_tm * fecn,
            re_tm * rare_earth,
            alloy_like * fecn,
            oxide * rare_earth,
            oxide * fecn,
            boride * fecn,
            pnictide * transition,
        ]
    )

    return [float(value) for value in features]


def magnetic_threshold_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    weights = [amount / total_atoms for amount in element_amounts.values()]
    fractions = {element.symbol: weight for element, weight in zip(elements, weights)}

    fecn = sum(fractions.get(symbol, 0.0) for symbol in ("Fe", "Co", "Ni"))
    mncr = fractions.get("Mn", 0.0) + fractions.get("Cr", 0.0)
    rare_earth = _fraction(elements, weights, lambda element: element.symbol in _RARE_EARTHS)
    transition = _fraction(elements, weights, lambda element: bool(element.is_transition_metal))
    candidate_magnetic = fecn + mncr + rare_earth
    nonmetal = _fraction(
        elements,
        weights,
        lambda element: not bool(element.is_metal) and not bool(element.is_metalloid),
    )
    oxygen = fractions.get("O", 0.0)
    boron = fractions.get("B", 0.0)
    pnictogen_chalcogen = sum(
        fractions.get(symbol, 0.0) for symbol in ("N", "P", "As", "Sb", "Bi", "O", "S", "Se", "Te")
    )
    metal_fraction = _fraction(elements, weights, lambda element: bool(element.is_metal))

    low_mag = 1.0 if 0.0 < candidate_magnetic < 0.25 else 0.0
    mid_mag = 1.0 if 0.25 <= candidate_magnetic < 0.6 else 0.0
    high_mag = 1.0 if candidate_magnetic >= 0.6 else 0.0
    high_fecn = 1.0 if fecn >= 0.5 else 0.0
    high_rare = 1.0 if rare_earth >= 0.25 else 0.0
    coexist_re_tm = 1.0 if rare_earth > 0.0 and transition > 0.0 else 0.0
    oxide = 1.0 if oxygen > 0.0 else 0.0
    boride = 1.0 if boron > 0.0 else 0.0
    alloy_like = 1.0 if metal_fraction >= 0.9 and fecn > 0.0 else 0.0
    nonmetal_diluted = 1.0 if candidate_magnetic > 0.0 and nonmetal >= 0.3 else 0.0

    flags = [
        low_mag,
        mid_mag,
        high_mag,
        high_fecn,
        high_rare,
        coexist_re_tm,
        oxide * low_mag,
        oxide * mid_mag,
        oxide * high_mag,
        boride * mid_mag,
        boride * high_mag,
        alloy_like * high_fecn,
        nonmetal_diluted,
        1.0 if pnictogen_chalcogen > 0.0 and transition > 0.0 else 0.0,
    ]

    features = composition_class_descriptor(composition)
    features.extend(flags)
    features.extend(
        [
            low_mag * candidate_magnetic,
            mid_mag * candidate_magnetic,
            high_mag * candidate_magnetic,
            high_fecn * fecn,
            high_rare * rare_earth,
            coexist_re_tm * candidate_magnetic,
            oxide * candidate_magnetic,
            boride * candidate_magnetic,
            nonmetal_diluted * candidate_magnetic,
            _safe_ratio(candidate_magnetic, nonmetal),
            _safe_ratio(fecn, nonmetal),
            _safe_ratio(rare_earth, nonmetal),
            _safe_ratio(transition, pnictogen_chalcogen),
            _safe_ratio(fecn, oxygen + boron),
        ]
    )

    return [float(value) for value in features]


def drop_sublattice_block_descriptor(composition: Composition) -> list[float]:
    features = magnetic_threshold_descriptor(composition)
    return features[:84] + features[131:]


def thirds_after_sublattice_descriptor(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_amounts = reduced.get_el_amt_dict()
    total_atoms = sum(element_amounts.values())
    elements = [Element(symbol) for symbol in element_amounts]
    weights = [amount / total_atoms for amount in element_amounts.values()]
    fractions = {element.symbol: weight for element, weight in zip(elements, weights)}
    fecn = sum(fractions.get(symbol, 0.0) for symbol in ("Fe", "Co", "Ni"))
    mncr = fractions.get("Mn", 0.0) + fractions.get("Cr", 0.0)
    rare_earth = _fraction(elements, weights, lambda element: element.symbol in _RARE_EARTHS)
    transition = _fraction(elements, weights, lambda element: bool(element.is_transition_metal))
    candidate_magnetic = fecn + mncr + rare_earth
    nonmetal = _fraction(elements, weights, lambda element: not bool(element.is_metal) and not bool(element.is_metalloid))
    oxygen = fractions.get("O", 0.0)
    boron = fractions.get("B", 0.0)
    pnictogen_chalcogen = sum(fractions.get(symbol, 0.0) for symbol in ("N", "P", "As", "Sb", "Bi", "O", "S", "Se", "Te"))
    metal_fraction = _fraction(elements, weights, lambda element: bool(element.is_metal))
    low_mag = 1.0 if 0.0 < candidate_magnetic < (1.0 / 3.0) else 0.0
    mid_mag = 1.0 if (1.0 / 3.0) <= candidate_magnetic < (2.0 / 3.0) else 0.0
    high_mag = 1.0 if candidate_magnetic >= (2.0 / 3.0) else 0.0
    high_fecn = 1.0 if fecn >= (2.0 / 3.0) else 0.0
    high_rare = 1.0 if rare_earth >= (1.0 / 3.0) else 0.0
    oxide = 1.0 if oxygen > 0.0 else 0.0
    boride = 1.0 if boron > 0.0 else 0.0
    alloy_like = 1.0 if metal_fraction >= 0.9 and fecn > 0.0 else 0.0
    nonmetal_diluted = 1.0 if candidate_magnetic > 0.0 and nonmetal >= 0.3 else 0.0
    flags = [low_mag, mid_mag, high_mag, high_fecn, high_rare, rare_earth * transition, oxide * low_mag, oxide * mid_mag, oxide * high_mag, boride * mid_mag, boride * high_mag, alloy_like * high_fecn, nonmetal_diluted, 1.0 if pnictogen_chalcogen > 0.0 and transition > 0.0 else 0.0]
    threshold = flags + [low_mag * candidate_magnetic, mid_mag * candidate_magnetic, high_mag * candidate_magnetic, high_fecn * fecn, high_rare * rare_earth, rare_earth * transition * candidate_magnetic, oxide * candidate_magnetic, boride * candidate_magnetic, nonmetal_diluted * candidate_magnetic, _safe_ratio(candidate_magnetic, nonmetal), _safe_ratio(fecn, nonmetal), _safe_ratio(rare_earth, nonmetal), _safe_ratio(transition, pnictogen_chalcogen), _safe_ratio(fecn, oxygen + boron)]
    return drop_sublattice_block_descriptor(composition)[:-28] + [float(value) for value in threshold]


def thirds_drop_thermal_descriptor(composition: Composition) -> list[float]:
    features = thirds_after_sublattice_descriptor(composition)
    return features[:84] + features[152:]
