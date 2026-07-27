from __future__ import annotations

import math
import warnings

import numpy as np
from pymatgen.core import Composition, Element


STRONG_3D = ("Fe", "Co", "Ni")
VARIABLE_3D = ("Mn", "Cr")
RARE_EARTH_MAGNETIC = ("Gd", "Tb", "Dy", "Ho", "Er")
CHALCOGENS = ("O", "S", "Se", "Te")
PNICTOGENS = ("N", "P", "As", "Sb", "Bi")
HALOGENS = ("F", "Cl", "Br", "I")
ALKALI_EARTH_DILUTERS = ("Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba")


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _element_property(element: Element, name: str) -> float:
    if name == "atomic_mass":
        return _safe_float(element.atomic_mass)
    if name == "electronegativity":
        return _safe_float(element.X)
    if name == "row":
        return _safe_float(element.row)
    if name == "group":
        return _safe_float(element.group)
    if name == "radius":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            radius = element.atomic_radius_calculated or element.atomic_radius
        return _safe_float(radius)
    if name == "atomic_number":
        return _safe_float(element.Z)
    raise ValueError(f"Unknown element property: {name}")


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> list[float]:
    mean = float(np.dot(values, weights))
    variance = float(np.dot((values - mean) ** 2, weights))
    std = math.sqrt(max(variance, 0.0))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    return [mean, std, min_value, max_value, max_value - min_value]


def _fraction_for_symbols(comp: Composition, symbols: tuple[str, ...]) -> float:
    return sum(_safe_float(comp.get_atomic_fraction(symbol)) for symbol in symbols)


def _common_oxidation_features(element: Element) -> tuple[float, float, float, float]:
    states = tuple(_safe_float(state) for state in element.common_oxidation_states)
    if not states:
        return (0.0, 0.0, 0.0, 0.0)
    state_array = np.asarray(states, dtype=float)
    return (
        float(np.mean(state_array)),
        float(np.max(np.abs(state_array))),
        float(len(states)),
        float(np.max(state_array) - np.min(state_array)),
    )


def _shell_counts(element: Element) -> tuple[float, float, float]:
    d_count = 0.0
    f_count = 0.0
    outer_count = 0.0
    max_n = 0
    for n_value, orbital, electrons in element.full_electronic_structure:
        n_int = int(n_value)
        electron_count = _safe_float(electrons)
        if orbital == "d":
            d_count += electron_count
        elif orbital == "f":
            f_count += electron_count
        if n_int > max_n:
            max_n = n_int
            outer_count = electron_count
        elif n_int == max_n:
            outer_count += electron_count
    return d_count, f_count, outer_count


def no_fe_co_pair_descriptor(comp: Composition) -> list[float]:
    fractional = comp.fractional_composition
    items = sorted(fractional.items(), key=lambda item: item[0].symbol)
    elements = [element for element, _ in items]
    fractions = np.asarray([_safe_float(amount) for _, amount in items], dtype=float)
    fractions = fractions / fractions.sum()

    reduced_atoms = _safe_float(comp.get_reduced_composition_and_factor()[0].num_atoms)
    entropy = -float(np.sum(fractions * np.log(np.clip(fractions, 1e-12, 1.0))))
    max_fraction = float(np.max(fractions))
    min_fraction = float(np.min(fractions))
    concentration = float(np.sum(fractions**2))

    features: list[float] = [
        float(len(elements)),
        reduced_atoms,
        max_fraction,
        min_fraction,
        max_fraction - min_fraction,
        concentration,
        entropy,
        entropy / max(math.log(max(len(elements), 2)), 1e-12),
    ]

    property_arrays: dict[str, np.ndarray] = {}
    for prop in (
        "atomic_number",
        "atomic_mass",
        "electronegativity",
        "row",
        "group",
        "radius",
    ):
        values = np.asarray([_element_property(element, prop) for element in elements], dtype=float)
        property_arrays[prop] = values
        features.extend(_weighted_stats(values, fractions))

    oxidation_matrix = np.asarray(
        [_common_oxidation_features(element) for element in elements],
        dtype=float,
    )
    for idx in range(oxidation_matrix.shape[1]):
        features.extend(_weighted_stats(oxidation_matrix[:, idx], fractions))

    shell_matrix = np.asarray([_shell_counts(element) for element in elements], dtype=float)
    for idx in range(shell_matrix.shape[1]):
        features.extend(_weighted_stats(shell_matrix[:, idx], fractions))

    strong_3d = _fraction_for_symbols(comp, STRONG_3D)
    variable_3d = _fraction_for_symbols(comp, VARIABLE_3D)
    rare_earth_magnetic = _fraction_for_symbols(comp, RARE_EARTH_MAGNETIC)
    magnetic_total = strong_3d + variable_3d + rare_earth_magnetic
    nonmagnetic_balance = max(1.0 - magnetic_total, 0.0)

    magnetic_symbols = set(STRONG_3D + VARIABLE_3D + RARE_EARTH_MAGNETIC)
    magnetic_mask = np.asarray([1.0 if element.symbol in magnetic_symbols else 0.0 for element in elements])
    magnetic_weights = fractions * magnetic_mask
    magnetic_weight_sum = float(np.sum(magnetic_weights))
    if magnetic_weight_sum > 0.0:
        magnetic_weights = magnetic_weights / magnetic_weight_sum
        magnetic_d = float(np.dot(shell_matrix[:, 0], magnetic_weights))
        magnetic_f = float(np.dot(shell_matrix[:, 1], magnetic_weights))
        magnetic_oxidation = float(np.dot(oxidation_matrix[:, 1], magnetic_weights))
    else:
        magnetic_d = 0.0
        magnetic_f = 0.0
        magnetic_oxidation = 0.0

    transition_metal = sum(
        frac for element, frac in zip(elements, fractions) if element.is_transition_metal
    )
    lanthanide = sum(frac for element, frac in zip(elements, fractions) if element.is_lanthanoid)
    actinide = sum(frac for element, frac in zip(elements, fractions) if element.is_actinoid)
    light_element = sum(frac for element, frac in zip(elements, fractions) if element.Z <= 10)
    chalcogen = _fraction_for_symbols(comp, CHALCOGENS)
    pnictogen = _fraction_for_symbols(comp, PNICTOGENS)
    halogen = _fraction_for_symbols(comp, HALOGENS)
    alkali_earth_diluter = _fraction_for_symbols(comp, ALKALI_EARTH_DILUTERS)
    oxygen = _fraction_for_symbols(comp, ("O",))

    en_range = float(np.max(property_arrays["electronegativity"]) - np.min(property_arrays["electronegativity"]))
    radius_range = float(np.max(property_arrays["radius"]) - np.min(property_arrays["radius"]))
    oxidation_magnitude_mean = float(np.dot(oxidation_matrix[:, 1], fractions))
    d_density = float(np.dot(shell_matrix[:, 0], fractions))
    f_density = float(np.dot(shell_matrix[:, 1], fractions))

    row_fractions = [
        sum(frac for element, frac in zip(elements, fractions) if element.row == row)
        for row in range(1, 8)
    ]
    group_fractions = [
        sum(frac for element, frac in zip(elements, fractions) if element.group == group)
        for group in range(1, 19)
    ]
    s_block = sum(frac for element, frac in zip(elements, fractions) if element.block == "s")
    p_block = sum(frac for element, frac in zip(elements, fractions) if element.block == "p")
    d_block = sum(frac for element, frac in zip(elements, fractions) if element.block == "d")
    f_block = sum(frac for element, frac in zip(elements, fractions) if element.block == "f")
    early_tm = sum(
        frac
        for element, frac in zip(elements, fractions)
        if element.is_transition_metal and 3 <= element.group <= 5
    )
    middle_tm = sum(
        frac
        for element, frac in zip(elements, fractions)
        if element.is_transition_metal and 6 <= element.group <= 8
    )
    late_tm = sum(
        frac
        for element, frac in zip(elements, fractions)
        if element.is_transition_metal and 9 <= element.group <= 12
    )
    tm_3d = sum(
        frac
        for element, frac in zip(elements, fractions)
        if element.is_transition_metal and element.row == 4
    )
    tm_4d = sum(
        frac
        for element, frac in zip(elements, fractions)
        if element.is_transition_metal and element.row == 5
    )
    tm_5d = sum(
        frac
        for element, frac in zip(elements, fractions)
        if element.is_transition_metal and element.row == 6
    )
    magnetic_by_row = [
        sum(
            frac
            for element, frac in zip(elements, fractions)
            if element.symbol in magnetic_symbols and element.row == row
        )
        for row in range(4, 7)
    ]
    fe_frac = _fraction_for_symbols(comp, ("Fe",))
    co_frac = _fraction_for_symbols(comp, ("Co",))
    ni_frac = _fraction_for_symbols(comp, ("Ni",))
    mn_frac = _fraction_for_symbols(comp, ("Mn",))
    cr_frac = _fraction_for_symbols(comp, ("Cr",))

    features.extend(
        [
            fe_frac,
            co_frac,
            ni_frac,
            mn_frac,
            cr_frac,
            strong_3d,
            variable_3d,
            rare_earth_magnetic,
            magnetic_total,
            transition_metal,
            lanthanide,
            actinide,
            light_element,
            chalcogen,
            pnictogen,
            halogen,
            oxygen,
            alkali_earth_diluter,
            magnetic_d,
            magnetic_f,
            magnetic_oxidation,
            strong_3d / (nonmagnetic_balance + 1e-6),
            magnetic_total / (nonmagnetic_balance + 1e-6),
            magnetic_total * en_range,
            magnetic_total * oxidation_magnitude_mean,
            magnetic_total * d_density,
            rare_earth_magnetic * f_density,
            transition_metal * radius_range,
            strong_3d * (1.0 - chalcogen - halogen),
            variable_3d * (chalcogen + pnictogen + oxygen),
            alkali_earth_diluter * nonmagnetic_balance,
        ]
    )
    features.extend(float(x) for x in row_fractions)
    features.extend(float(x) for x in group_fractions)
    features.extend(
        [
            s_block,
            p_block,
            d_block,
            f_block,
            early_tm,
            middle_tm,
            late_tm,
            tm_3d,
            tm_4d,
            tm_5d,
            *magnetic_by_row,
            strong_3d / (p_block + 1e-6),
            rare_earth_magnetic / (d_block + 1e-6),
            transition_metal * late_tm,
            magnetic_total * chalcogen,
            magnetic_total * pnictogen,
            magnetic_total * halogen,
            tm_3d * en_range,
            (tm_4d + tm_5d) * radius_range,
        ]
    )
    features.extend(
        [
            fe_frac * ni_frac,
            co_frac * ni_frac,
            fe_frac * mn_frac,
            co_frac * mn_frac,
            ni_frac * mn_frac,
            fe_frac * oxygen,
            co_frac * oxygen,
            ni_frac * oxygen,
            mn_frac * oxygen,
            cr_frac * oxygen,
            fe_frac * chalcogen,
            co_frac * chalcogen,
            ni_frac * chalcogen,
            mn_frac * chalcogen,
            cr_frac * chalcogen,
            strong_3d * late_tm,
            tm_3d * tm_3d,
            transition_metal * transition_metal,
        ]
    )

    return [float(x) if math.isfinite(float(x)) else 0.0 for x in features]
