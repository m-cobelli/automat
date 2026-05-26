from __future__ import annotations

import math

import numpy as np
from pymatgen.core import Composition, Element

MAGNETIC_3D = {"Cr", "Mn", "Fe", "Co", "Ni"}
LIGHT_INTERSTITIALS = {"B", "C", "N", "H"}
CHALCOGENS = {"O", "S", "Se", "Te"}
HALOGENS = {"F", "Cl", "Br", "I"}
ANIONS = {"B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "Te", "I"}
FAMILY_ELEMENTS = [
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Y",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "U",
    "B",
    "C",
    "N",
    "O",
    "P",
    "Si",
    "Al",
    "Ga",
    "Ge",
    "Sn",
]
LIGHT_RARE_EARTHS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd"}
HEAVY_RARE_EARTHS = {"Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}
YTTRIUM_LANTHANIDES = LIGHT_RARE_EARTHS | HEAVY_RARE_EARTHS | {"Y"}
ACTINIDES = {"Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}
GROUP_13 = {"B", "Al", "Ga", "In", "Tl"}
GROUP_14 = {"C", "Si", "Ge", "Sn", "Pb"}


def _clean(value: object) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _weighted_stats(values: list[float], weights: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.dot(weights, arr))
    var = float(np.dot(weights, (arr - mean) ** 2))
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    return [mean, math.sqrt(max(var, 0.0)), min_value, max_value, max_value - min_value]


def _fraction(symbols: list[str], fractions: np.ndarray, selected: set[str]) -> float:
    return float(sum(frac for symbol, frac in zip(symbols, fractions) if symbol in selected))


def baseline_descriptor(composition: Composition) -> list[float]:
    frac_comp = composition.fractional_composition
    amounts = frac_comp.get_el_amt_dict()
    symbols = list(amounts)
    fractions = np.asarray([amounts[symbol] for symbol in symbols], dtype=float)
    elements = [Element(symbol) for symbol in symbols]

    n_elements = float(len(elements))
    entropy = float(-np.sum(fractions * np.log(np.clip(fractions, 1e-12, 1.0))))
    max_fraction = float(np.max(fractions))
    l2_fraction = float(np.sum(fractions**2))

    fe_fraction = _fraction(symbols, fractions, {"Fe"})
    co_fraction = _fraction(symbols, fractions, {"Co"})
    ni_fraction = _fraction(symbols, fractions, {"Ni"})
    mn_fraction = _fraction(symbols, fractions, {"Mn"})
    cr_fraction = _fraction(symbols, fractions, {"Cr"})
    magnetic_3d_fraction = _fraction(symbols, fractions, MAGNETIC_3D)
    light_interstitial_fraction = _fraction(symbols, fractions, LIGHT_INTERSTITIALS)
    oxygen_fraction = _fraction(symbols, fractions, {"O"})
    chalcogen_fraction = _fraction(symbols, fractions, CHALCOGENS)
    halogen_fraction = _fraction(symbols, fractions, HALOGENS)

    transition_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_transition_metal)
    )
    lanthanoid_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_lanthanoid)
    )
    actinoid_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_actinoid)
    )
    metal_fraction = float(sum(frac for element, frac in zip(elements, fractions) if element.is_metal))
    metalloid_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_metalloid)
    )
    heavy_fraction = float(sum(frac for element, frac in zip(elements, fractions) if element.Z >= 57))

    total_atoms = _clean(composition.num_atoms)
    stoichiometry_features = [
        n_elements,
        math.log1p(total_atoms),
        entropy,
        max_fraction,
        l2_fraction,
    ]

    targeted_features = [
        fe_fraction,
        co_fraction,
        ni_fraction,
        mn_fraction,
        cr_fraction,
        fe_fraction + co_fraction,
        magnetic_3d_fraction,
        transition_fraction,
        lanthanoid_fraction,
        actinoid_fraction,
        metal_fraction,
        metalloid_fraction,
        light_interstitial_fraction,
        oxygen_fraction,
        chalcogen_fraction,
        halogen_fraction,
        heavy_fraction,
        magnetic_3d_fraction * (1.0 - oxygen_fraction),
        transition_fraction * (lanthanoid_fraction + actinoid_fraction),
        transition_fraction * oxygen_fraction,
        (fe_fraction + co_fraction) / max(transition_fraction, 1e-8),
    ]

    property_values = [
        [element.Z for element in elements],
        [_clean(element.atomic_mass) for element in elements],
        [element.row for element in elements],
        [element.group for element in elements],
        [_clean(element.X) for element in elements],
        [_clean(element.atomic_radius) for element in elements],
        [_clean(getattr(element, "mendeleev_no", 0.0)) for element in elements],
    ]
    property_features: list[float] = []
    for values in property_values:
        property_features.extend(_weighted_stats(values, fractions))

    features = stoichiometry_features + targeted_features + property_features
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in features]


def magnetic_sublattice_v1_descriptor(composition: Composition) -> list[float]:
    frac_comp = composition.fractional_composition
    amounts = frac_comp.get_el_amt_dict()
    symbols = list(amounts)
    fractions = np.asarray([amounts[symbol] for symbol in symbols], dtype=float)
    elements = [Element(symbol) for symbol in symbols]

    fe_fraction = _fraction(symbols, fractions, {"Fe"})
    co_fraction = _fraction(symbols, fractions, {"Co"})
    ni_fraction = _fraction(symbols, fractions, {"Ni"})
    mn_fraction = _fraction(symbols, fractions, {"Mn"})
    cr_fraction = _fraction(symbols, fractions, {"Cr"})
    fe_co_fraction = fe_fraction + co_fraction
    mn_cr_fraction = mn_fraction + cr_fraction
    magnetic_3d_fraction = _fraction(symbols, fractions, MAGNETIC_3D)
    light_interstitial_fraction = _fraction(symbols, fractions, LIGHT_INTERSTITIALS)
    oxygen_fraction = _fraction(symbols, fractions, {"O"})
    anion_fraction = _fraction(symbols, fractions, ANIONS)

    transition_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_transition_metal)
    )
    lanthanoid_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_lanthanoid)
    )
    actinoid_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_actinoid)
    )
    metal_fraction = float(sum(frac for element, frac in zip(elements, fractions) if element.is_metal))
    rare_act_fraction = lanthanoid_fraction + actinoid_fraction

    extra_features = [
        fe_fraction,
        co_fraction,
        ni_fraction,
        mn_fraction,
        cr_fraction,
        fe_co_fraction,
        magnetic_3d_fraction,
        transition_fraction,
        lanthanoid_fraction,
        actinoid_fraction,
        light_interstitial_fraction,
        oxygen_fraction,
        anion_fraction,
        fe_co_fraction / max(transition_fraction, 1e-8),
        magnetic_3d_fraction / max(transition_fraction, 1e-8),
        rare_act_fraction / max(transition_fraction + rare_act_fraction, 1e-8),
        light_interstitial_fraction / max(fe_co_fraction, 1e-8),
        fe_co_fraction * rare_act_fraction,
        fe_co_fraction * (1.0 - oxygen_fraction),
        magnetic_3d_fraction * metal_fraction,
        transition_fraction * anion_fraction,
        fe_fraction - co_fraction,
        fe_co_fraction - mn_cr_fraction,
        transition_fraction - anion_fraction,
        oxygen_fraction - light_interstitial_fraction,
    ]

    features = baseline_descriptor(composition) + extra_features
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in features]


def family_identity_v1_descriptor(composition: Composition) -> list[float]:
    frac_comp = composition.fractional_composition
    amounts = frac_comp.get_el_amt_dict()
    symbols = list(amounts)
    fractions = np.asarray([amounts[symbol] for symbol in symbols], dtype=float)
    elements = [Element(symbol) for symbol in symbols]

    fe_co_fraction = _fraction(symbols, fractions, {"Fe", "Co"})
    magnetic_3d_fraction = _fraction(symbols, fractions, MAGNETIC_3D)
    transition_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_transition_metal)
    )
    light_re_fraction = _fraction(symbols, fractions, LIGHT_RARE_EARTHS)
    heavy_re_fraction = _fraction(symbols, fractions, HEAVY_RARE_EARTHS)
    yttrium_lanthanide_fraction = _fraction(symbols, fractions, YTTRIUM_LANTHANIDES)
    actinide_fraction = _fraction(symbols, fractions, ACTINIDES)
    pnictogen_fraction = _fraction(symbols, fractions, PNICTOGENS)
    group_13_fraction = _fraction(symbols, fractions, GROUP_13)
    group_14_fraction = _fraction(symbols, fractions, GROUP_14)
    interstitial_fraction = _fraction(symbols, fractions, LIGHT_INTERSTITIALS)
    oxygen_fraction = _fraction(symbols, fractions, {"O"})
    bcn_fraction = _fraction(symbols, fractions, {"B", "C", "N"})
    nd_sm_y_fraction = _fraction(symbols, fractions, {"Nd", "Sm", "Y"})

    explicit_element_features = [_fraction(symbols, fractions, {symbol}) for symbol in FAMILY_ELEMENTS]
    grouped_features = [
        light_re_fraction,
        heavy_re_fraction,
        yttrium_lanthanide_fraction,
        actinide_fraction,
        pnictogen_fraction,
        group_13_fraction,
        group_14_fraction,
        interstitial_fraction,
    ]
    interaction_features = [
        nd_sm_y_fraction * fe_co_fraction,
        heavy_re_fraction * fe_co_fraction,
        yttrium_lanthanide_fraction * transition_fraction,
        actinide_fraction * transition_fraction,
        bcn_fraction * fe_co_fraction,
        oxygen_fraction * magnetic_3d_fraction,
        pnictogen_fraction * transition_fraction,
        group_13_fraction * fe_co_fraction,
        group_14_fraction * fe_co_fraction,
    ]

    features = (
        magnetic_sublattice_v1_descriptor(composition)
        + explicit_element_features
        + grouped_features
        + interaction_features
    )
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in features]


def _oxidation_priors(element: Element) -> tuple[float, float, float, float, float, float]:
    states = [float(state) for state in element.common_oxidation_states]
    if not states:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    min_state = min(states)
    max_state = max(states)
    abs_state = max(abs(state) for state in states)
    positive_capacity = max(max_state, 0.0)
    negative_capacity = abs(min(min_state, 0.0))
    mean_state = float(np.mean(states))
    return (
        min_state,
        max_state,
        abs_state,
        positive_capacity,
        negative_capacity,
        mean_state,
    )


def valence_balance_v1_descriptor(composition: Composition) -> list[float]:
    frac_comp = composition.fractional_composition
    amounts = frac_comp.get_el_amt_dict()
    symbols = list(amounts)
    fractions = np.asarray([amounts[symbol] for symbol in symbols], dtype=float)
    elements = [Element(symbol) for symbol in symbols]

    priors = np.asarray([_oxidation_priors(element) for element in elements], dtype=float)
    min_states = priors[:, 0]
    max_states = priors[:, 1]
    abs_states = priors[:, 2]
    positive_capacity = priors[:, 3]
    negative_capacity = priors[:, 4]
    mean_states = priors[:, 5]

    summary_features: list[float] = []
    for values in [
        min_states,
        max_states,
        abs_states,
        positive_capacity,
        negative_capacity,
        mean_states,
    ]:
        summary_features.extend(_weighted_stats(list(values), fractions)[:2])

    weighted_positive = float(np.dot(fractions, positive_capacity))
    weighted_negative = float(np.dot(fractions, negative_capacity))
    weighted_abs = float(np.dot(fractions, abs_states))
    weighted_mean_state = float(np.dot(fractions, mean_states))
    state_span = float(np.max(max_states) - np.min(min_states))
    imbalance = weighted_positive - weighted_negative
    anion_fraction = _fraction(symbols, fractions, ANIONS)
    transition_fraction = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_transition_metal)
    )
    electronegativities = np.asarray([_clean(element.X) for element in elements], dtype=float)

    formula_features = [
        weighted_positive,
        weighted_negative,
        weighted_positive + weighted_negative,
        imbalance,
        abs(imbalance),
        weighted_positive / max(weighted_negative, 1e-8),
        weighted_abs,
        weighted_mean_state,
        state_span,
        float(np.dot(fractions, electronegativities * positive_capacity)),
        float(np.dot(fractions, electronegativities * negative_capacity)),
        anion_fraction * weighted_positive,
        transition_fraction * weighted_negative,
    ]

    features = family_identity_v1_descriptor(composition) + summary_features + formula_features
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in features]


def periodic_identity_v1_descriptor(composition: Composition) -> list[float]:
    frac_comp = composition.fractional_composition
    amounts = frac_comp.get_el_amt_dict()
    symbols = list(amounts)
    fractions = np.asarray([amounts[symbol] for symbol in symbols], dtype=float)
    elements = [Element(symbol) for symbol in symbols]

    z_fractions = [0.0] * 96
    for element, fraction in zip(elements, fractions):
        if 1 <= element.Z <= 96:
            z_fractions[element.Z - 1] = float(fraction)

    s_block = float(
        sum(frac for element, frac in zip(elements, fractions) if element.group in {1, 2})
    )
    p_block = float(
        sum(frac for element, frac in zip(elements, fractions) if 13 <= element.group <= 18)
    )
    d_block = float(
        sum(frac for element, frac in zip(elements, fractions) if element.is_transition_metal)
    )
    f_block = _fraction(symbols, fractions, YTTRIUM_LANTHANIDES | ACTINIDES)
    fe_co_fraction = _fraction(symbols, fractions, {"Fe", "Co"})
    magnetic_3d_fraction = _fraction(symbols, fractions, MAGNETIC_3D)
    anion_fraction = _fraction(symbols, fractions, ANIONS)

    block_features = [
        s_block,
        p_block,
        d_block,
        f_block,
        d_block * f_block,
        d_block * anion_fraction,
        f_block * fe_co_fraction,
        p_block * magnetic_3d_fraction,
        fe_co_fraction / max(d_block, 1e-8),
        f_block / max(d_block + f_block, 1e-8),
    ]

    features = valence_balance_v1_descriptor(composition) + z_fractions + block_features
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in features]
