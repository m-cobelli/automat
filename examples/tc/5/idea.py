from __future__ import annotations

import math
import warnings

import numpy as np
from pymatgen.core import Composition


MAGNETIC_ELEMENTS = ("Fe", "Co", "Ni", "Mn", "Cr", "Gd", "Tb", "Dy", "Ho", "Er", "Sm", "Nd")
FERROMAGNETIC_3D = {"Fe", "Co", "Ni"}
VARIABLE_3D = {"Mn", "Cr"}
LANTHANIDES = {
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
MAGNETIC_RARE_EARTHS = {"Nd", "Sm", "Gd", "Tb", "Dy", "Ho", "Er"}
ELEMENT_FRACTION_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
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
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
)
OBSERVED_TRAIN_SYMBOLS = (
    "H",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Cs",
    "Ba",
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
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Th",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
)
CONTEXT_PAIR_SYMBOLS = (
    "B",
    "C",
    "N",
    "O",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "In",
    "Sn",
    "Sb",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Hf",
    "Ta",
    "W",
)
SPECIFIC_3D_CONTEXT_SYMBOLS = ("B", "C", "O", "Al", "Si", "Mn", "Ga", "Ge", "In", "Sn")
APPROX_MAGNETIC_MOMENTS = {
    "Cr": 2.5,
    "Mn": 3.5,
    "Fe": 2.2,
    "Co": 1.7,
    "Ni": 0.6,
    "Nd": 3.2,
    "Sm": 0.7,
    "Gd": 7.0,
    "Tb": 9.0,
    "Dy": 10.0,
    "Ho": 10.0,
    "Er": 9.0,
}


def _finite_float(value, fallback: float = 0.0) -> float:
    if value is None:
        return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(result):
        return fallback
    return result


def _weighted_stats(values: list[float], weights: list[float]) -> list[float]:
    values_array = np.asarray(values, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    mean = float(np.dot(values_array, weights_array))
    variance = float(np.dot((values_array - mean) ** 2, weights_array))
    minimum = float(np.min(values_array))
    maximum = float(np.max(values_array))
    return [mean, math.sqrt(max(variance, 0.0)), minimum, maximum, maximum - minimum]


def baseline_descriptor(composition: Composition) -> list[float]:
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    by_symbol = {element.symbol: fraction for element, fraction in zip(elements, fractions)}

    features: list[float] = []
    n_elements = len(elements)
    max_fraction = max(fractions)
    sorted_fractions = sorted(fractions, reverse=True)
    entropy = -sum(frac * math.log(frac) for frac in fractions if frac > 0.0)
    features.extend(
        [
            float(n_elements),
            entropy,
            entropy / math.log(n_elements) if n_elements > 1 else 0.0,
            max_fraction,
            sorted_fractions[0] + (sorted_fractions[1] if n_elements > 1 else 0.0),
            sum(frac * frac for frac in fractions),
            min(fractions),
        ]
    )

    property_values = [
        [_finite_float(element.Z) for element in elements],
        [_finite_float(element.atomic_mass) for element in elements],
        [_finite_float(element.X, fallback=0.0) for element in elements],
        [_finite_float(element.row) for element in elements],
        [_finite_float(element.group) for element in elements],
        [_finite_float(element.atomic_radius, fallback=0.0) for element in elements],
        [_finite_float(element.average_ionic_radius, fallback=0.0) for element in elements],
    ]
    for values in property_values:
        features.extend(_weighted_stats(values, fractions))

    metal_fraction = sum(frac for element, frac in zip(elements, fractions) if element.is_metal)
    transition_fraction = sum(
        frac for element, frac in zip(elements, fractions) if element.is_transition_metal
    )
    lanthanide_fraction = sum(
        frac for element, frac in zip(elements, fractions) if element.symbol in LANTHANIDES
    )
    magnetic_re_fraction = sum(
        frac for element, frac in zip(elements, fractions) if element.symbol in MAGNETIC_RARE_EARTHS
    )
    ferromagnetic_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in FERROMAGNETIC_3D)
    variable_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in VARIABLE_3D)
    magnetic_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_ELEMENTS)
    p_block_fraction = sum(
        frac for element, frac in zip(elements, fractions) if 13 <= _finite_float(element.group) <= 18
    )

    features.extend(
        [
            metal_fraction,
            transition_fraction,
            lanthanide_fraction,
            magnetic_re_fraction,
            ferromagnetic_3d_fraction,
            variable_3d_fraction,
            magnetic_fraction,
            sum(frac for element, frac in zip(elements, fractions) if element.is_alkali),
            sum(frac for element, frac in zip(elements, fractions) if element.is_alkaline),
            sum(frac for element, frac in zip(elements, fractions) if element.is_chalcogen),
            sum(frac for element, frac in zip(elements, fractions) if element.is_halogen),
            p_block_fraction,
        ]
    )

    features.extend(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_ELEMENTS)
    features.extend(
        [
            by_symbol.get("Fe", 0.0) * magnetic_re_fraction,
            by_symbol.get("Co", 0.0) * magnetic_re_fraction,
            by_symbol.get("Ni", 0.0) * magnetic_re_fraction,
            ferromagnetic_3d_fraction * magnetic_re_fraction,
            transition_fraction * lanthanide_fraction,
            magnetic_fraction * (1.0 - magnetic_fraction),
            metal_fraction * magnetic_fraction,
        ]
    )

    moment_values = [APPROX_MAGNETIC_MOMENTS.get(element.symbol, 0.0) for element in elements]
    moment_mean = float(np.dot(moment_values, fractions))
    moment_sq_mean = float(np.dot(np.asarray(moment_values, dtype=float) ** 2, fractions))
    features.extend(
        [
            moment_mean,
            moment_sq_mean,
            max(moment_values) if moment_values else 0.0,
            moment_mean * ferromagnetic_3d_fraction,
            moment_mean * magnetic_re_fraction,
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError("baseline_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def element_fraction_map_descriptor(composition: Composition) -> list[float]:
    features = baseline_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    by_symbol = {element.symbol: fraction for element, fraction in zip(elements, fractions)}

    element_fractions = [by_symbol.get(symbol, 0.0) for symbol in ELEMENT_FRACTION_SYMBOLS]
    features.extend(element_fractions)

    ferromagnetic_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in FERROMAGNETIC_3D)
    variable_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in VARIABLE_3D)
    magnetic_re_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_RARE_EARTHS)
    lanthanide_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in LANTHANIDES)
    nonmagnetic_lanthanide_fraction = max(lanthanide_fraction - magnetic_re_fraction, 0.0)
    transition_fraction = sum(
        frac for element, frac in zip(elements, fractions) if element.is_transition_metal
    )
    magnetic_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_ELEMENTS)

    features.extend(
        [
            ferromagnetic_3d_fraction,
            variable_3d_fraction,
            magnetic_re_fraction,
            nonmagnetic_lanthanide_fraction,
            _safe_ratio(by_symbol.get("Fe", 0.0), ferromagnetic_3d_fraction),
            _safe_ratio(by_symbol.get("Co", 0.0), ferromagnetic_3d_fraction),
            _safe_ratio(by_symbol.get("Ni", 0.0), ferromagnetic_3d_fraction),
            _safe_ratio(ferromagnetic_3d_fraction, transition_fraction),
            _safe_ratio(ferromagnetic_3d_fraction, magnetic_fraction),
            _safe_ratio(magnetic_re_fraction, magnetic_fraction),
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError("element_fraction_map_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def magnetic_pair_context_descriptor(composition: Composition) -> list[float]:
    features = element_fraction_map_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    by_symbol = {element.symbol: fraction for element, fraction in zip(elements, fractions)}

    ferromagnetic_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in FERROMAGNETIC_3D)
    variable_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in VARIABLE_3D)
    magnetic_re_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_RARE_EARTHS)
    magnetic_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_ELEMENTS)
    magnetic_families = (
        ferromagnetic_3d_fraction,
        variable_3d_fraction,
        magnetic_re_fraction,
        magnetic_fraction,
    )

    for symbol in CONTEXT_PAIR_SYMBOLS:
        context_fraction = by_symbol.get(symbol, 0.0)
        features.extend(context_fraction * family_fraction for family_fraction in magnetic_families)

    features.extend(
        [
            ferromagnetic_3d_fraction * variable_3d_fraction,
            ferromagnetic_3d_fraction * magnetic_re_fraction,
            variable_3d_fraction * magnetic_re_fraction,
            ferromagnetic_3d_fraction * (1.0 - ferromagnetic_3d_fraction),
            magnetic_re_fraction * (1.0 - magnetic_re_fraction),
            magnetic_fraction * (1.0 - magnetic_fraction),
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError("magnetic_pair_context_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def specific_3d_context_descriptor(composition: Composition) -> list[float]:
    features = magnetic_pair_context_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    by_symbol = {element.symbol: fraction for element, fraction in zip(elements, fractions)}

    for context_symbol in SPECIFIC_3D_CONTEXT_SYMBOLS:
        context_fraction = by_symbol.get(context_symbol, 0.0)
        features.extend(
            [
                by_symbol.get("Fe", 0.0) * context_fraction,
                by_symbol.get("Co", 0.0) * context_fraction,
                by_symbol.get("Ni", 0.0) * context_fraction,
            ]
        )

    if not all(math.isfinite(value) for value in features):
        raise ValueError("specific_3d_context_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def lean_specific_3d_context_descriptor(composition: Composition) -> list[float]:
    features = element_fraction_map_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    by_symbol = {element.symbol: fraction for element, fraction in zip(elements, fractions)}

    ferromagnetic_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in FERROMAGNETIC_3D)
    variable_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in VARIABLE_3D)
    magnetic_re_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_RARE_EARTHS)
    magnetic_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_ELEMENTS)
    magnetic_families = (
        ferromagnetic_3d_fraction,
        variable_3d_fraction,
        magnetic_re_fraction,
        magnetic_fraction,
    )

    for symbol in CONTEXT_PAIR_SYMBOLS:
        context_fraction = by_symbol.get(symbol, 0.0)
        features.extend(context_fraction * family_fraction for family_fraction in magnetic_families)

    for context_symbol in SPECIFIC_3D_CONTEXT_SYMBOLS:
        context_fraction = by_symbol.get(context_symbol, 0.0)
        features.extend(
            [
                by_symbol.get("Fe", 0.0) * context_fraction,
                by_symbol.get("Co", 0.0) * context_fraction,
                by_symbol.get("Ni", 0.0) * context_fraction,
            ]
        )

    if not all(math.isfinite(value) for value in features):
        raise ValueError("lean_specific_3d_context_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def three_d_balance_descriptor(composition: Composition) -> list[float]:
    features = lean_specific_3d_context_descriptor(composition)
    fractional = composition.fractional_composition
    by_symbol = {element.symbol: float(amount) for element, amount in fractional.items()}
    fe_fraction = by_symbol.get("Fe", 0.0)
    co_fraction = by_symbol.get("Co", 0.0)
    ni_fraction = by_symbol.get("Ni", 0.0)
    ferromagnetic_3d_fraction = fe_fraction + co_fraction + ni_fraction

    features.extend(
        [
            _safe_ratio(fe_fraction, ferromagnetic_3d_fraction),
            _safe_ratio(co_fraction, ferromagnetic_3d_fraction),
            _safe_ratio(ni_fraction, ferromagnetic_3d_fraction),
            fe_fraction * co_fraction,
            fe_fraction * ni_fraction,
            co_fraction * ni_fraction,
            max(fe_fraction, co_fraction, ni_fraction),
            fe_fraction * fe_fraction + co_fraction * co_fraction + ni_fraction * ni_fraction,
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError("three_d_balance_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def observed_element_balance_descriptor(composition: Composition) -> list[float]:
    features = baseline_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    by_symbol = {element.symbol: fraction for element, fraction in zip(elements, fractions)}

    features.extend(by_symbol.get(symbol, 0.0) for symbol in OBSERVED_TRAIN_SYMBOLS)

    ferromagnetic_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in FERROMAGNETIC_3D)
    variable_3d_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in VARIABLE_3D)
    magnetic_re_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_RARE_EARTHS)
    lanthanide_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in LANTHANIDES)
    nonmagnetic_lanthanide_fraction = max(lanthanide_fraction - magnetic_re_fraction, 0.0)
    transition_fraction = sum(
        frac for element, frac in zip(elements, fractions) if element.is_transition_metal
    )
    magnetic_fraction = sum(by_symbol.get(symbol, 0.0) for symbol in MAGNETIC_ELEMENTS)

    features.extend(
        [
            ferromagnetic_3d_fraction,
            variable_3d_fraction,
            magnetic_re_fraction,
            nonmagnetic_lanthanide_fraction,
            _safe_ratio(by_symbol.get("Fe", 0.0), ferromagnetic_3d_fraction),
            _safe_ratio(by_symbol.get("Co", 0.0), ferromagnetic_3d_fraction),
            _safe_ratio(by_symbol.get("Ni", 0.0), ferromagnetic_3d_fraction),
            _safe_ratio(ferromagnetic_3d_fraction, transition_fraction),
            _safe_ratio(ferromagnetic_3d_fraction, magnetic_fraction),
            _safe_ratio(magnetic_re_fraction, magnetic_fraction),
        ]
    )

    magnetic_families = (
        ferromagnetic_3d_fraction,
        variable_3d_fraction,
        magnetic_re_fraction,
        magnetic_fraction,
    )
    for symbol in CONTEXT_PAIR_SYMBOLS:
        context_fraction = by_symbol.get(symbol, 0.0)
        features.extend(context_fraction * family_fraction for family_fraction in magnetic_families)

    for context_symbol in SPECIFIC_3D_CONTEXT_SYMBOLS:
        context_fraction = by_symbol.get(context_symbol, 0.0)
        features.extend(
            [
                by_symbol.get("Fe", 0.0) * context_fraction,
                by_symbol.get("Co", 0.0) * context_fraction,
                by_symbol.get("Ni", 0.0) * context_fraction,
            ]
        )

    fe_fraction = by_symbol.get("Fe", 0.0)
    co_fraction = by_symbol.get("Co", 0.0)
    ni_fraction = by_symbol.get("Ni", 0.0)
    features.extend(
        [
            _safe_ratio(fe_fraction, ferromagnetic_3d_fraction),
            _safe_ratio(co_fraction, ferromagnetic_3d_fraction),
            _safe_ratio(ni_fraction, ferromagnetic_3d_fraction),
            fe_fraction * co_fraction,
            fe_fraction * ni_fraction,
            co_fraction * ni_fraction,
            max(fe_fraction, co_fraction, ni_fraction),
            fe_fraction * fe_fraction + co_fraction * co_fraction + ni_fraction * ni_fraction,
        ]
    )

    if not all(math.isfinite(value) for value in features):
        raise ValueError("observed_element_balance_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def thermomechanical_context_descriptor(composition: Composition) -> list[float]:
    features = observed_element_balance_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    property_names = (
        "melting_point",
        "boiling_point",
        "density_of_solid",
        "thermal_conductivity",
        "electrical_resistivity",
        "velocity_of_sound",
        "coefficient_of_linear_thermal_expansion",
        "bulk_modulus",
        "youngs_modulus",
    )

    for property_name in property_names:
        values = []
        for element in elements:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                values.append(_finite_float(getattr(element, property_name, 0.0)))
        features.append(float(np.dot(values, fractions)))

    if not all(math.isfinite(value) for value in features):
        raise ValueError("thermomechanical_context_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def thermomechanical_core_descriptor(composition: Composition) -> list[float]:
    features = observed_element_balance_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    property_names = (
        "melting_point",
        "boiling_point",
        "density_of_solid",
        "thermal_conductivity",
        "velocity_of_sound",
        "bulk_modulus",
        "youngs_modulus",
    )

    for property_name in property_names:
        values = []
        for element in elements:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                values.append(_finite_float(getattr(element, property_name, 0.0)))
        features.append(float(np.dot(values, fractions)))

    if not all(math.isfinite(value) for value in features):
        raise ValueError("thermomechanical_core_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def thermomechanical_no_sound_descriptor(composition: Composition) -> list[float]:
    features = observed_element_balance_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    property_names = (
        "melting_point",
        "boiling_point",
        "density_of_solid",
        "thermal_conductivity",
        "bulk_modulus",
        "youngs_modulus",
    )

    for property_name in property_names:
        values = []
        for element in elements:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                values.append(_finite_float(getattr(element, property_name, 0.0)))
        features.append(float(np.dot(values, fractions)))

    if not all(math.isfinite(value) for value in features):
        raise ValueError("thermomechanical_no_sound_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def mendeleev_thermo_descriptor(composition: Composition) -> list[float]:
    features = thermomechanical_no_sound_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    values = [_finite_float(getattr(element, "mendeleev_no", 0.0)) for element in elements]
    mean, std = _weighted_stats(values, fractions)[:2]
    features.extend([mean, std, max(values) - min(values)])

    if not all(math.isfinite(value) for value in features):
        raise ValueError("mendeleev_thermo_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def mendeleev_no_range_descriptor(composition: Composition) -> list[float]:
    features = thermomechanical_no_sound_descriptor(composition)
    fractional = composition.fractional_composition
    items = sorted(fractional.items(), key=lambda pair: pair[0].symbol)
    elements = [element for element, _ in items]
    fractions = [float(amount) for _, amount in items]
    values = [_finite_float(getattr(element, "mendeleev_no", 0.0)) for element in elements]
    mean, std = _weighted_stats(values, fractions)[:2]
    features.extend([mean, std])

    if not all(math.isfinite(value) for value in features):
        raise ValueError("mendeleev_no_range_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def mendeleev_with_max_descriptor(composition: Composition) -> list[float]:
    features = mendeleev_no_range_descriptor(composition)
    fractional = composition.fractional_composition
    elements = [element for element, _ in sorted(fractional.items(), key=lambda pair: pair[0].symbol)]
    values = [_finite_float(getattr(element, "mendeleev_no", 0.0)) for element in elements]
    features.append(max(values))

    if not all(math.isfinite(value) for value in features):
        raise ValueError("mendeleev_with_max_descriptor produced a non-finite feature")
    return [float(value) for value in features]


def no_total_magnetic_context_products_descriptor(composition: Composition) -> list[float]:
    features = mendeleev_with_max_descriptor(composition)
    keep_indices = set(range(len(features)))
    for start in range(173, 353, 4):
        keep_indices.discard(start + 3)
    pruned = [value for index, value in enumerate(features) if index in keep_indices]
    if not all(math.isfinite(value) for value in pruned):
        raise ValueError("no_total_magnetic_context_products_descriptor produced a non-finite feature")
    return [float(value) for value in pruned]
