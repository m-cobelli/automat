from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from pymatgen.core import Composition, Element

SELECTED_PROPERTIES = ['row', 'group', 'mendeleev_number']
SELECTED_STATS = ['mean']
INCLUDE_EN_BLOCK = False
INCLUDE_ORBITAL_BLOCK = False
INCLUDE_REDUCED_SHAPE = True
OXIDATION_GUESS_MAX_REDUCED_ATOMS = 40.0


PROPERTY_GETTERS = {
    'atomic_number': lambda element: element.Z,
    'electronegativity': lambda element: element.X,
    'row': lambda element: element.row,
    'group': lambda element: element.group,
    'mendeleev_number': lambda element: element.mendeleev_no,
    'atomic_mass': lambda element: element.atomic_mass,
    'atomic_radius': lambda element: element.atomic_radius,
    'atomic_radius_calculated': lambda element: element.atomic_radius_calculated,
    'average_ionic_radius': lambda element: element.average_ionic_radius,
    'melting_point': lambda element: element.melting_point,
    'boiling_point': lambda element: element.boiling_point,
    'density_of_solid': lambda element: element.density_of_solid,
    'electrical_resistivity': lambda element: element.electrical_resistivity,
}


def _as_finite_float(value: object) -> tuple[float, float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if not math.isfinite(numeric):
        return 0.0, 1.0
    return numeric, 0.0


def _selected_stats(values: np.ndarray, weights: np.ndarray) -> list[float]:
    mean = float(np.dot(weights, values))
    variance = float(np.dot(weights, (values - mean) ** 2))
    std = math.sqrt(max(variance, 0.0))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    pairwise = float(np.sum(np.abs(values[:, None] - values[None, :]) * weights[:, None] * weights[None, :]))
    lookup = {
        'mean': mean,
        'std': std,
        'min': min_value,
        'max': max_value,
        'range': max_value - min_value,
        'pairwise': pairwise,
    }
    return [float(lookup[key]) for key in SELECTED_STATS]


def _fraction_for(elements: list[Element], fractions: np.ndarray, predicate: Callable[[Element], bool]) -> float:
    return float(sum(fraction for element, fraction in zip(elements, fractions) if predicate(element)))


def _reduced_shape(composition: Composition) -> list[float]:
    if not INCLUDE_REDUCED_SHAPE:
        return []
    coeffs = np.asarray(list(composition.reduced_composition.get_el_amt_dict().values()), dtype=float)
    return [float(np.sum(coeffs)), float(np.max(coeffs)), float(np.min(coeffs)), float(np.std(coeffs))]


def _en_block(elements: list[Element], fractions: np.ndarray) -> list[float]:
    if not INCLUDE_EN_BLOCK:
        return []
    xs = np.asarray([_as_finite_float(element.X)[0] for element in elements], dtype=float)
    mean_x = float(np.dot(fractions, xs))
    low = xs <= mean_x
    high = xs > mean_x
    low_fraction = float(np.sum(fractions[low]))
    high_fraction = float(np.sum(fractions[high]))
    low_mean = float(np.dot(fractions[low], xs[low]) / low_fraction) if low_fraction else mean_x
    high_mean = float(np.dot(fractions[high], xs[high]) / high_fraction) if high_fraction else mean_x
    delta = max(0.0, high_mean - low_mean)
    pairwise = np.abs(xs[:, None] - xs[None, :])
    return [
        low_fraction,
        high_fraction,
        low_mean,
        high_mean,
        delta,
        float(1.0 - math.exp(-0.25 * delta * delta)),
        float(np.max(pairwise)),
        float(np.sum(pairwise * fractions[:, None] * fractions[None, :])),
    ]


def _anion_shape_block(composition: Composition, elements: list[Element], fractions: np.ndarray) -> list[float]:
    reduced_atoms = float(sum(composition.reduced_composition.get_el_amt_dict().values()))
    metal_fraction = _fraction_for(elements, fractions, lambda element: bool(element.is_metal))
    oxygen_fraction = _fraction_for(elements, fractions, lambda element: element.symbol == "O")
    heavy_chalcogen_fraction = _fraction_for(
        elements,
        fractions,
        lambda element: bool(element.is_chalcogen) and element.symbol != "O",
    )
    halogen_fraction = _fraction_for(elements, fractions, lambda element: bool(element.is_halogen))
    total_chalcogen = oxygen_fraction + heavy_chalcogen_fraction
    oxygen_chalcogen_ratio = oxygen_fraction / total_chalcogen if total_chalcogen else 0.0
    return [
        oxygen_fraction,
        heavy_chalcogen_fraction,
        halogen_fraction,
        oxygen_chalcogen_ratio,
        metal_fraction * oxygen_fraction,
        metal_fraction * heavy_chalcogen_fraction,
        metal_fraction * halogen_fraction,
        reduced_atoms * oxygen_fraction,
        reduced_atoms * halogen_fraction,
    ]


def _anion_ratio_block(composition: Composition, elements: list[Element], fractions: np.ndarray) -> list[float]:
    reduced_atoms = float(sum(composition.reduced_composition.get_el_amt_dict().values()))
    metal_fraction = _fraction_for(elements, fractions, lambda element: bool(element.is_metal))
    oxygen_fraction = _fraction_for(elements, fractions, lambda element: element.symbol == "O")
    heavy_chalcogen_fraction = _fraction_for(
        elements,
        fractions,
        lambda element: bool(element.is_chalcogen) and element.symbol != "O",
    )
    halogen_fraction = _fraction_for(elements, fractions, lambda element: bool(element.is_halogen))
    total_anion_fraction = oxygen_fraction + heavy_chalcogen_fraction + halogen_fraction
    denom = metal_fraction + 1.0e-6
    return [
        total_anion_fraction,
        oxygen_fraction / denom,
        heavy_chalcogen_fraction / denom,
        halogen_fraction / denom,
        total_anion_fraction / denom,
        float(oxygen_fraction >= heavy_chalcogen_fraction and oxygen_fraction >= halogen_fraction and oxygen_fraction > 0.0),
        float(halogen_fraction > oxygen_fraction and halogen_fraction >= heavy_chalcogen_fraction),
        float(heavy_chalcogen_fraction > oxygen_fraction and heavy_chalcogen_fraction > halogen_fraction),
        reduced_atoms * heavy_chalcogen_fraction,
    ]


def _oxidation_block(composition: Composition, elements: list[Element], fractions: np.ndarray) -> list[float]:
    if composition.reduced_composition.num_atoms > OXIDATION_GUESS_MAX_REDUCED_ATOMS:
        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    try:
        guesses = composition.oxi_state_guesses()
    except (TypeError, ValueError):
        guesses = ()
    if not guesses:
        return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    states = np.asarray([float(guesses[0].get(element.symbol, 0.0)) for element in elements], dtype=float)
    mean_state = float(np.dot(fractions, states))
    std_state = math.sqrt(max(float(np.dot(fractions, (states - mean_state) ** 2)), 0.0))
    positive = states > 0.0
    negative = states < 0.0
    pairwise = np.abs(states[:, None] - states[None, :])
    return [
        1.0,
        0.0,
        float(min(len(guesses), 8)),
        float(np.dot(fractions, np.abs(states))),
        std_state,
        float(np.min(states)),
        float(np.max(states)),
        float(np.max(states) - np.min(states)),
        float(np.sum(fractions[positive])),
        float(np.sum(fractions[negative])),
        float(np.sum(fractions[positive] * states[positive])),
        float(np.sum(fractions[negative] * np.abs(states[negative]))),
        float(np.sum(pairwise * fractions[:, None] * fractions[None, :])),
    ]


def _orbital_counts(element: Element) -> dict[str, float]:
    counts = {'s': 0.0, 'p': 0.0, 'd': 0.0, 'f': 0.0}
    outer = {'s': 0.0, 'p': 0.0, 'd': 0.0, 'f': 0.0}
    max_n = 0
    for n, orbital, electrons in element.full_electronic_structure:
        letter = str(orbital)[0]
        if letter in counts:
            counts[letter] += float(electrons)
            max_n = max(max_n, int(n))
    for n, orbital, electrons in element.full_electronic_structure:
        letter = str(orbital)[0]
        if int(n) == max_n and letter in outer:
            outer[letter] += float(electrons)
    return {
        'total_s': counts['s'], 'total_p': counts['p'], 'total_d': counts['d'], 'total_f': counts['f'],
        'outer_s': outer['s'], 'outer_p': outer['p'], 'outer_d': outer['d'], 'outer_f': outer['f'],
        'has_d': float(counts['d'] > 0.0), 'has_f': float(counts['f'] > 0.0),
    }


def _compact_orbital_block(elements: list[Element], fractions: np.ndarray) -> list[float]:
    rows = [_orbital_counts(element) for element in elements]
    features = []
    for key in ("outer_s", "outer_p", "outer_d", "outer_f"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        features.append(float(np.dot(fractions, values)))
    return features


def _orbital_block(elements: list[Element], fractions: np.ndarray) -> list[float]:
    if not INCLUDE_ORBITAL_BLOCK:
        return []
    rows = [_orbital_counts(element) for element in elements]
    features: list[float] = []
    for key in ('total_s', 'total_p', 'total_d', 'total_f', 'outer_s', 'outer_p', 'outer_d', 'outer_f', 'has_d', 'has_f'):
        values = np.asarray([row[key] for row in rows], dtype=float)
        features.extend(_selected_stats(values, fractions))
    return features


def gap_anion_ratio_oxi_outer_orbital_v7(composition: Composition) -> list[float]:
    fractional = composition.fractional_composition
    items = tuple(fractional.get_el_amt_dict().items())
    elements = [Element(symbol) for symbol, _ in items]
    fractions = np.asarray([amount for _, amount in items], dtype=float)
    fractions = fractions / float(np.sum(fractions))
    entropy = -float(np.sum([fraction * math.log(fraction) for fraction in fractions if fraction > 0.0]))
    features: list[float] = [
        float(len(elements)), entropy, float(np.max(fractions)), float(np.min(fractions)), float(np.sum(fractions ** 2)),
        *_reduced_shape(composition),
        _fraction_for(elements, fractions, lambda element: bool(element.is_metal)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_metalloid)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_transition_metal)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_alkali)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_alkaline)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_halogen)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_chalcogen)),
        _fraction_for(elements, fractions, lambda element: int(element.group) == 15),
        _fraction_for(elements, fractions, lambda element: bool(element.is_lanthanoid)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_actinoid)),
        _fraction_for(elements, fractions, lambda element: bool(element.is_noble_gas)),
    ]
    for property_name in SELECTED_PROPERTIES:
        getter = PROPERTY_GETTERS[property_name]
        values = []
        missing = []
        for element in elements:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                value, flag = _as_finite_float(getter(element))
            values.append(value)
            missing.append(flag)
        value_array = np.asarray(values, dtype=float)
        missing_array = np.asarray(missing, dtype=float)
        features.extend(_selected_stats(value_array, fractions))
        features.append(float(np.dot(fractions, missing_array)))
    features.extend(_anion_shape_block(composition, elements, fractions))
    features.extend(_anion_ratio_block(composition, elements, fractions))
    features.extend(_oxidation_block(composition, elements, fractions))
    features.extend(_compact_orbital_block(elements, fractions))
    features.extend(_en_block(elements, fractions))
    features.extend(_orbital_block(elements, fractions))
    return [float(value) for value in features]
