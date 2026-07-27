from __future__ import annotations

import math
from itertools import product
from collections.abc import Iterable

import numpy as np
from pymatgen.core import Composition, Element


def _finite_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def _weighted_stats(values: Iterable[float], weights: np.ndarray) -> list[float]:
    arr = np.asarray(list(values), dtype=float)
    mean = float(np.dot(weights, arr))
    centered = arr - mean
    std = float(np.sqrt(np.dot(weights, centered * centered)))
    min_value = float(np.min(arr))
    max_value = float(np.max(arr))
    return [
        mean,
        std,
        min_value,
        max_value,
        max_value - min_value,
        float(np.dot(weights, np.abs(centered))),
    ]


def _common_oxidation_summary(element: Element) -> tuple[float, float, float]:
    states = tuple(_finite_float(state) for state in element.common_oxidation_states)
    if not states:
        return 0.0, 0.0, 0.0
    return min(states), max(states), max(states) - min(states)


def _fraction(elements: list[Element], weights: np.ndarray, predicate) -> float:
    return float(sum(weight for element, weight in zip(elements, weights) if predicate(element)))


def baseline(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()

    fractions = weights.tolist()
    entropy = -float(np.dot(weights, np.log(np.clip(weights, 1e-12, 1.0))))
    max_entropy = math.log(len(elements)) if len(elements) > 1 else 1.0

    features: list[float] = [
        float(len(elements)),
        _finite_float(reduced.num_atoms),
        float(max(fractions)),
        float(min(fractions)),
        entropy,
        entropy / max_entropy,
    ]

    atomic_numbers = [_finite_float(element.Z) for element in elements]
    atomic_masses = [_finite_float(element.atomic_mass) for element in elements]
    mendeleev_numbers = [_finite_float(element.mendeleev_no) for element in elements]
    rows = [_finite_float(element.row) for element in elements]
    groups = [_finite_float(element.group) for element in elements]
    electronegativities = [_finite_float(element.X, default=0.0) for element in elements]
    atomic_radii = [_finite_float(element.atomic_radius, default=0.0) for element in elements]
    ionic_radii = [_finite_float(element.average_ionic_radius, default=0.0) for element in elements]
    ox_min, ox_max, ox_span = zip(*[_common_oxidation_summary(element) for element in elements])

    for values in (
        atomic_numbers,
        atomic_masses,
        mendeleev_numbers,
        rows,
        groups,
        electronegativities,
        atomic_radii,
        ionic_radii,
        ox_min,
        ox_max,
        ox_span,
    ):
        features.extend(_weighted_stats(values, weights))

    metal_fraction = _fraction(elements, weights, lambda element: element.is_metal)
    metalloid_fraction = _fraction(elements, weights, lambda element: element.is_metalloid)
    nonmetal_fraction = 1.0 - metal_fraction - metalloid_fraction
    chalcogen_fraction = _fraction(elements, weights, lambda element: element.group == 16)
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    pnictogen_fraction = _fraction(elements, weights, lambda element: element.group == 15)
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    light_anion_fraction = _fraction(
        elements,
        weights,
        lambda element: element.symbol in {"B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"},
    )
    transition_fraction = _fraction(elements, weights, lambda element: 3 <= element.group <= 12)
    alkali_fraction = _fraction(elements, weights, lambda element: element.group == 1)
    alkaline_earth_fraction = _fraction(elements, weights, lambda element: element.group == 2)
    p_block_fraction = _fraction(elements, weights, lambda element: 13 <= element.group <= 18)
    high_en_fraction = _fraction(elements, weights, lambda element: _finite_float(element.X) >= 2.5)

    en_mean = features[6 + 5 * 6]
    en_min = min(electronegativities)
    en_max = max(electronegativities)
    features.extend(
        [
            metal_fraction,
            metalloid_fraction,
            nonmetal_fraction,
            transition_fraction,
            alkali_fraction,
            alkaline_earth_fraction,
            p_block_fraction,
            oxygen_fraction,
            chalcogen_fraction,
            halogen_fraction,
            pnictogen_fraction,
            light_anion_fraction,
            high_en_fraction,
            en_max - en_min,
            en_max * max(0.0, 1.0 - metal_fraction),
            en_mean * high_en_fraction,
            (en_max - en_min) * (chalcogen_fraction + halogen_fraction),
            oxygen_fraction * max(0.0, 1.0 - transition_fraction),
        ]
    )

    return [float(value) for value in features]


def lean_gap_core(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()

    fractions = weights.tolist()
    entropy = -float(np.dot(weights, np.log(np.clip(weights, 1e-12, 1.0))))
    max_entropy = math.log(len(elements)) if len(elements) > 1 else 1.0
    features: list[float] = [
        float(len(elements)),
        _finite_float(reduced.num_atoms),
        float(max(fractions)),
        float(min(fractions)),
        entropy,
        entropy / max_entropy,
    ]

    ox_min, ox_max, ox_span = zip(*[_common_oxidation_summary(element) for element in elements])
    property_sets = (
        [_finite_float(element.Z) for element in elements],
        [_finite_float(element.mendeleev_no) for element in elements],
        [_finite_float(element.row) for element in elements],
        [_finite_float(element.group) for element in elements],
        [_finite_float(element.X, default=0.0) for element in elements],
        ox_min,
        ox_max,
        ox_span,
    )
    for values in property_sets:
        features.extend(_weighted_stats(values, weights))

    metal_fraction = _fraction(elements, weights, lambda element: element.is_metal)
    metalloid_fraction = _fraction(elements, weights, lambda element: element.is_metalloid)
    nonmetal_fraction = 1.0 - metal_fraction - metalloid_fraction
    transition_fraction = _fraction(elements, weights, lambda element: 3 <= element.group <= 12)
    alkali_fraction = _fraction(elements, weights, lambda element: element.group == 1)
    alkaline_earth_fraction = _fraction(elements, weights, lambda element: element.group == 2)
    p_block_fraction = _fraction(elements, weights, lambda element: 13 <= element.group <= 18)
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    chalcogen_fraction = _fraction(elements, weights, lambda element: element.group == 16)
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    pnictogen_fraction = _fraction(elements, weights, lambda element: element.group == 15)
    light_anion_fraction = _fraction(
        elements,
        weights,
        lambda element: element.symbol in {"B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I"},
    )
    high_en_fraction = _fraction(elements, weights, lambda element: _finite_float(element.X) >= 2.5)
    electronegativities = [_finite_float(element.X, default=0.0) for element in elements]
    en_range = max(electronegativities) - min(electronegativities)

    features.extend(
        [
            metal_fraction,
            metalloid_fraction,
            nonmetal_fraction,
            transition_fraction,
            alkali_fraction,
            alkaline_earth_fraction,
            p_block_fraction,
            oxygen_fraction,
            chalcogen_fraction,
            halogen_fraction,
            pnictogen_fraction,
            light_anion_fraction,
            high_en_fraction,
            en_range,
            en_range * oxygen_fraction,
            en_range * (chalcogen_fraction + halogen_fraction),
            en_range * max(0.0, 1.0 - transition_fraction),
            en_range * nonmetal_fraction,
        ]
    )
    return [float(value) for value in features]


def lean_periodic_histogram(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()

    group_bins = [
        _fraction(elements, weights, lambda element, group=group: element.group == group)
        for group in range(1, 19)
    ]
    period_bins = [
        _fraction(elements, weights, lambda element, period=period: element.row == period)
        for period in range(1, 8)
    ]
    electronegativities = [_finite_float(element.X, default=0.0) for element in elements]
    en_range = max(electronegativities) - min(electronegativities)
    transition_fraction = sum(group_bins[2:12])
    group_13_14_fraction = group_bins[12] + group_bins[13]
    heavy_fraction = period_bins[4] + period_bins[5]

    extras = [
        *group_bins,
        *period_bins,
        en_range * group_bins[15],
        en_range * group_bins[16],
        en_range * transition_fraction,
        en_range * group_13_14_fraction,
        en_range * heavy_fraction,
        group_bins[15] * max(0.0, 1.0 - transition_fraction),
        group_bins[16] * max(0.0, 1.0 - transition_fraction),
        transition_fraction * heavy_fraction,
    ]
    return [*lean_gap_core(composition), *[float(value) for value in extras]]


def lean_periodic_bins(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()

    group_bins = [
        _fraction(elements, weights, lambda element, group=group: element.group == group)
        for group in range(1, 19)
    ]
    period_bins = [
        _fraction(elements, weights, lambda element, period=period: element.row == period)
        for period in range(1, 8)
    ]
    return [*lean_gap_core(composition), *[float(value) for value in [*group_bins, *period_bins]]]


def lean_group_bins(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()

    group_bins = [
        _fraction(elements, weights, lambda element, group=group: element.group == group)
        for group in range(1, 19)
    ]
    return [*lean_gap_core(composition), *[float(value) for value in group_bins]]


def lean_group_element_fractions(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    element_bins = np.zeros(94, dtype=float)
    for element in reduced.elements:
        atomic_number = int(element.Z)
        if 1 <= atomic_number <= len(element_bins):
            element_bins[atomic_number - 1] = reduced.get_atomic_fraction(element)
    return [*lean_group_bins(composition), *[float(value) for value in element_bins]]


def _oxidation_state_options(element: Element) -> tuple[float, ...]:
    states = tuple(sorted({_finite_float(state) for state in element.common_oxidation_states}))
    return states if states else (0.0,)


def lean_element_charge_balance(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    options = [_oxidation_state_options(element)[:8] for element in elements]
    while math.prod(len(option) for option in options) > 20000:
        widest_index = max(range(len(options)), key=lambda index: len(options[index]))
        option = options[widest_index]
        if len(option) <= 2:
            break
        options[widest_index] = (option[0], option[-1])

    assignments = list(product(*options))
    residuals = np.asarray([float(np.dot(weights, assignment)) for assignment in assignments], dtype=float)
    abs_residuals = np.abs(residuals)
    best_index = int(np.argmin(abs_residuals))
    best_assignment = np.asarray(assignments[best_index], dtype=float)
    positive_sum = float(np.dot(weights, np.clip(best_assignment, 0.0, None)))
    negative_sum = float(np.dot(weights, np.clip(best_assignment, None, 0.0)))
    has_positive = bool(np.any(best_assignment > 0.0))
    has_negative = bool(np.any(best_assignment < 0.0))

    charge_features = [
        float(abs_residuals[best_index]),
        float(residuals[best_index]),
        float(np.sum(abs_residuals <= 0.05)),
        float(np.mean(abs_residuals)),
        float(np.std(abs_residuals)),
        positive_sum,
        negative_sum,
        abs(positive_sum + negative_sum),
        1.0 if has_positive and has_negative else 0.0,
        float(len(assignments)),
    ]
    return [*lean_group_element_fractions(composition), *charge_features]


def lean_element_charge_compact(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    options = [_oxidation_state_options(element)[:8] for element in elements]
    while math.prod(len(option) for option in options) > 20000:
        widest_index = max(range(len(options)), key=lambda index: len(options[index]))
        option = options[widest_index]
        if len(option) <= 2:
            break
        options[widest_index] = (option[0], option[-1])

    assignments = list(product(*options))
    residuals = np.asarray([float(np.dot(weights, assignment)) for assignment in assignments], dtype=float)
    abs_residuals = np.abs(residuals)
    best_index = int(np.argmin(abs_residuals))
    best_assignment = np.asarray(assignments[best_index], dtype=float)
    positive_sum = float(np.dot(weights, np.clip(best_assignment, 0.0, None)))
    negative_sum = float(np.dot(weights, np.clip(best_assignment, None, 0.0)))
    has_positive = bool(np.any(best_assignment > 0.0))
    has_negative = bool(np.any(best_assignment < 0.0))

    charge_features = [
        float(abs_residuals[best_index]),
        float(residuals[best_index]),
        positive_sum,
        negative_sum,
        abs(positive_sum + negative_sum),
        1.0 if has_positive and has_negative else 0.0,
    ]
    return [*lean_group_element_fractions(composition), *charge_features]


def lean_element_charge_oxidation_bins(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    options = [_oxidation_state_options(element)[:8] for element in elements]
    while math.prod(len(option) for option in options) > 20000:
        widest_index = max(range(len(options)), key=lambda index: len(options[index]))
        option = options[widest_index]
        if len(option) <= 2:
            break
        options[widest_index] = (option[0], option[-1])

    assignments = list(product(*options))
    residuals = np.asarray([float(np.dot(weights, assignment)) for assignment in assignments], dtype=float)
    best_assignment = assignments[int(np.argmin(np.abs(residuals)))]
    oxidation_bins = np.zeros(94, dtype=float)
    for element, weight, oxidation_state in zip(elements, weights, best_assignment):
        atomic_number = int(element.Z)
        if 1 <= atomic_number <= len(oxidation_bins):
            oxidation_bins[atomic_number - 1] = float(weight * oxidation_state)
    return [*lean_element_charge_compact(composition), *[float(value) for value in oxidation_bins]]


def lean_element_charge_raw_oxidation_bins(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    options = [_oxidation_state_options(element)[:8] for element in elements]
    while math.prod(len(option) for option in options) > 20000:
        widest_index = max(range(len(options)), key=lambda index: len(options[index]))
        option = options[widest_index]
        if len(option) <= 2:
            break
        options[widest_index] = (option[0], option[-1])

    assignments = list(product(*options))
    residuals = np.asarray([float(np.dot(weights, assignment)) for assignment in assignments], dtype=float)
    best_assignment = assignments[int(np.argmin(np.abs(residuals)))]
    raw_bins = np.zeros(94, dtype=float)
    for element, oxidation_state in zip(elements, best_assignment):
        atomic_number = int(element.Z)
        if 1 <= atomic_number <= len(raw_bins):
            raw_bins[atomic_number - 1] = float(oxidation_state)
    return [*lean_element_charge_oxidation_bins(composition), *[float(value) for value in raw_bins]]


def lean_element_charge_raw_class_flags(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    chalcogen_fraction = _fraction(elements, weights, lambda element: element.group == 16)
    transition_fraction = _fraction(elements, weights, lambda element: 3 <= element.group <= 12)
    metal_fraction = _fraction(elements, weights, lambda element: element.is_metal)
    min_fraction = float(np.min(weights))
    class_features = [
        1.0 if len(elements) == 2 else 0.0,
        1.0 if len(elements) == 3 else 0.0,
        1.0 if len(elements) >= 4 else 0.0,
        1.0 if oxygen_fraction > 0.0 else 0.0,
        1.0 if halogen_fraction > 0.0 else 0.0,
        1.0 if chalcogen_fraction > 0.0 else 0.0,
        1.0 if transition_fraction > 0.0 else 0.0,
        oxygen_fraction * transition_fraction,
        halogen_fraction * metal_fraction,
        chalcogen_fraction * metal_fraction,
        float(np.max(weights) / min_fraction) if min_fraction > 0.0 else 0.0,
        float(np.dot(weights, weights)),
    ]
    return [*lean_element_charge_raw_oxidation_bins(composition), *class_features]


def lean_element_charge_raw_class_only(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    chalcogen_fraction = _fraction(elements, weights, lambda element: element.group == 16)
    transition_fraction = _fraction(elements, weights, lambda element: 3 <= element.group <= 12)
    metal_fraction = _fraction(elements, weights, lambda element: element.is_metal)
    class_features = [
        1.0 if len(elements) == 2 else 0.0,
        1.0 if len(elements) == 3 else 0.0,
        1.0 if len(elements) >= 4 else 0.0,
        1.0 if oxygen_fraction > 0.0 else 0.0,
        1.0 if halogen_fraction > 0.0 else 0.0,
        1.0 if chalcogen_fraction > 0.0 else 0.0,
        1.0 if transition_fraction > 0.0 else 0.0,
        oxygen_fraction * transition_fraction,
        halogen_fraction * metal_fraction,
        chalcogen_fraction * metal_fraction,
    ]
    return [*lean_element_charge_raw_oxidation_bins(composition), *class_features]


def lean_element_charge_raw_anion_balance(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    heavy_chalcogen_fraction = _fraction(
        elements,
        weights,
        lambda element: element.group == 16 and element.symbol != "O",
    )
    high_en_fraction = _fraction(elements, weights, lambda element: _finite_float(element.X) >= 2.5)
    anion_features = [
        oxygen_fraction - halogen_fraction,
        oxygen_fraction - heavy_chalcogen_fraction,
        halogen_fraction + heavy_chalcogen_fraction,
        oxygen_fraction * high_en_fraction,
    ]
    return [*lean_element_charge_raw_class_only(composition), *anion_features]


def lean_element_charge_raw_anion_abs_balance(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    heavy_chalcogen_fraction = _fraction(
        elements,
        weights,
        lambda element: element.group == 16 and element.symbol != "O",
    )
    abs_features = [
        abs(oxygen_fraction - halogen_fraction),
        abs(oxygen_fraction - heavy_chalcogen_fraction),
        abs(halogen_fraction + heavy_chalcogen_fraction),
    ]
    return [*lean_element_charge_raw_anion_balance(composition), *abs_features]


def lean_element_charge_raw_anion_squared_balance(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    heavy_chalcogen_fraction = _fraction(
        elements,
        weights,
        lambda element: element.group == 16 and element.symbol != "O",
    )
    contrasts = [
        oxygen_fraction - halogen_fraction,
        oxygen_fraction - heavy_chalcogen_fraction,
        halogen_fraction + heavy_chalcogen_fraction,
    ]
    squared_features = [float(value * value) for value in contrasts]
    return [*lean_element_charge_raw_anion_abs_balance(composition), *squared_features]


def lean_element_charge_raw_anion_intensity(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    elements = list(reduced.elements)
    weights = np.asarray([reduced.get_atomic_fraction(element) for element in elements], dtype=float)
    weights = weights / weights.sum()
    oxygen_fraction = _fraction(elements, weights, lambda element: element.symbol == "O")
    halogen_fraction = _fraction(elements, weights, lambda element: element.group == 17)
    heavy_chalcogen_fraction = _fraction(
        elements,
        weights,
        lambda element: element.group == 16 and element.symbol != "O",
    )
    non_oxide_anion_fraction = halogen_fraction + heavy_chalcogen_fraction
    intensity_features = [
        oxygen_fraction * oxygen_fraction,
        non_oxide_anion_fraction * non_oxide_anion_fraction,
    ]
    return [*lean_element_charge_raw_anion_squared_balance(composition), *intensity_features]
