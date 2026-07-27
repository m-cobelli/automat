from __future__ import annotations

import math
import warnings
from itertools import product

import numpy as np
from pymatgen.core import Composition, Element


def _finite(value: float | int | None, default: float = 0.0) -> float:
    if value is None:
        return default
    value = float(value)
    if not math.isfinite(value):
        return default
    return value


def _weighted_stats(values: list[float], weights: np.ndarray) -> list[float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.dot(weights, array))
    variance = float(np.dot(weights, (array - mean) ** 2))
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return [mean, minimum, maximum, maximum - minimum, math.sqrt(max(variance, 0.0))]


def baseline(comp: Composition) -> list[float]:
    reduced = comp.reduced_composition
    element_amounts = reduced.element_composition.get_el_amt_dict()
    total_atoms = float(sum(element_amounts.values()))

    elements = [Element(symbol) for symbol in element_amounts]
    fractions = np.asarray(
        [amount / total_atoms for amount in element_amounts.values()],
        dtype=float,
    )

    entropy = -float(np.sum(fractions * np.log(fractions)))
    max_fraction = float(np.max(fractions))
    min_fraction = float(np.min(fractions))

    features = [
        float(len(elements)),
        total_atoms,
        max_fraction,
        min_fraction,
        max_fraction - min_fraction,
        entropy,
    ]

    property_values = [
        [_finite(element.X) for element in elements],
        [_finite(element.Z) for element in elements],
        [_finite(element.row) for element in elements],
        [_finite(element.group) for element in elements],
        [_finite(element.atomic_mass) for element in elements],
    ]
    for values in property_values:
        features.extend(_weighted_stats(values, fractions))

    metal_fraction = float(
        sum(fraction for element, fraction in zip(elements, fractions) if element.is_metal)
    )
    metalloid_fraction = float(
        sum(fraction for element, fraction in zip(elements, fractions) if element.is_metalloid)
    )
    nonmetal_fraction = 1.0 - metal_fraction - metalloid_fraction
    features.extend([metal_fraction, metalloid_fraction, nonmetal_fraction])

    return [_finite(value) for value in features]


def _oxidation_values(element: Element) -> tuple[float, float, float, float, float, float]:
    states = tuple(element.common_oxidation_states) or tuple(element.oxidation_states)
    if not states:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = np.asarray(states, dtype=float)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    has_positive = bool(np.any(values > 0.0))
    has_negative = bool(np.any(values < 0.0))
    return (
        minimum,
        maximum,
        maximum - minimum,
        float(np.max(np.abs(values))),
        float(len(values)),
        float(has_positive and has_negative),
    )


def orbital_oxidation_v1(comp: Composition) -> list[float]:
    reduced = comp.reduced_composition
    element_amounts = reduced.element_composition.get_el_amt_dict()
    total_atoms = float(sum(element_amounts.values()))
    elements = [Element(symbol) for symbol in element_amounts]
    fractions = np.asarray(
        [amount / total_atoms for amount in element_amounts.values()],
        dtype=float,
    )

    features = baseline(comp)

    for block in ("s", "p", "d", "f"):
        features.append(
            float(sum(f for element, f in zip(elements, fractions) if element.block == block))
        )

    families = [
        lambda element: element.is_transition_metal,
        lambda element: element.is_post_transition_metal,
        lambda element: element.is_alkali,
        lambda element: element.is_alkaline,
        lambda element: element.group == 17,
        lambda element: element.group == 16,
        lambda element: element.group == 15,
        lambda element: element.is_lanthanoid,
        lambda element: element.is_actinoid,
    ]
    for family in families:
        features.append(
            float(sum(f for element, f in zip(elements, fractions) if family(element)))
        )

    oxidation_columns = list(zip(*[_oxidation_values(element) for element in elements]))
    for values in oxidation_columns:
        features.extend(_weighted_stats(list(values), fractions))

    return [_finite(value) for value in features]


def _candidate_oxidation_states(element: Element) -> tuple[float, ...]:
    states = tuple(element.common_oxidation_states) or tuple(element.oxidation_states)
    if not states:
        return (0.0,)
    return tuple(float(state) for state in states)


def ionic_balance_v1(comp: Composition) -> list[float]:
    reduced = comp.reduced_composition
    element_amounts = reduced.element_composition.get_el_amt_dict()
    total_atoms = float(sum(element_amounts.values()))
    elements = [Element(symbol) for symbol in element_amounts]
    amounts = np.asarray(list(element_amounts.values()), dtype=float)
    fractions = amounts / total_atoms

    features = orbital_oxidation_v1(comp)

    state_options = [_candidate_oxidation_states(element) for element in elements]
    best_states: tuple[float, ...] | None = None
    best_key: tuple[float, float, float] | None = None
    neutral_count = 0
    total_count = 0

    for states in product(*state_options):
        total_count += 1
        state_array = np.asarray(states, dtype=float)
        charge = float(np.dot(amounts, state_array))
        abs_charge = abs(charge)
        ionic_strength = float(np.dot(amounts, np.abs(state_array)))
        mean_abs_state = float(np.dot(fractions, np.abs(state_array)))
        if abs_charge <= 1e-12:
            neutral_count += 1
        key = (abs_charge, -ionic_strength, mean_abs_state)
        if best_key is None or key < best_key:
            best_key = key
            best_states = states

    if best_states is None:
        best_states = tuple(0.0 for _ in elements)
        total_count = 1

    selected = np.asarray(best_states, dtype=float)
    selected_charge = float(np.dot(amounts, selected))
    residual_per_atom = abs(selected_charge) / total_atoms
    ionic_strength_per_atom = float(np.dot(amounts, np.abs(selected)) / total_atoms)
    exact_neutral = float(residual_per_atom <= 1e-12)
    neutral_fraction = float(neutral_count / max(total_count, 1))

    cation_mask = selected > 0.0
    anion_mask = selected < 0.0
    neutral_mask = selected == 0.0
    cation_fraction = float(np.sum(fractions[cation_mask]))
    anion_fraction = float(np.sum(fractions[anion_mask]))
    neutral_atom_fraction = float(np.sum(fractions[neutral_mask]))

    cation_ox = (
        float(np.dot(fractions[cation_mask], selected[cation_mask]) / cation_fraction)
        if cation_fraction > 0.0
        else 0.0
    )
    anion_ox = (
        float(np.dot(fractions[anion_mask], np.abs(selected[anion_mask])) / anion_fraction)
        if anion_fraction > 0.0
        else 0.0
    )
    electronegativities = np.asarray([_finite(element.X) for element in elements], dtype=float)
    cation_x = (
        float(np.dot(fractions[cation_mask], electronegativities[cation_mask]) / cation_fraction)
        if cation_fraction > 0.0
        else 0.0
    )
    anion_x = (
        float(np.dot(fractions[anion_mask], electronegativities[anion_mask]) / anion_fraction)
        if anion_fraction > 0.0
        else 0.0
    )

    features.extend(
        [
            residual_per_atom,
            exact_neutral,
            neutral_fraction,
            selected_charge / total_atoms,
            ionic_strength_per_atom,
            cation_fraction,
            anion_fraction,
            neutral_atom_fraction,
            cation_ox,
            anion_ox,
            anion_x - cation_x,
        ]
    )

    return [_finite(value) for value in features]


def _positive_values(values: list[float]) -> list[float]:
    return [value for value in values if value > 0.0 and math.isfinite(value)]


def _safe_property(element: Element, name: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _finite(getattr(element, name, None))


def size_thermo_v1(comp: Composition) -> list[float]:
    reduced = comp.reduced_composition
    element_amounts = reduced.element_composition.get_el_amt_dict()
    total_atoms = float(sum(element_amounts.values()))
    elements = [Element(symbol) for symbol in element_amounts]
    fractions = np.asarray(
        [amount / total_atoms for amount in element_amounts.values()],
        dtype=float,
    )

    features = ionic_balance_v1(comp)
    property_names = [
        "atomic_radius",
        "atomic_radius_calculated",
        "van_der_waals_radius",
        "molar_volume",
        "density_of_solid",
        "melting_point",
        "boiling_point",
    ]
    for name in property_names:
        values = [_safe_property(element, name) for element in elements]
        features.extend(_weighted_stats(values, fractions))

    radii = [_safe_property(element, "atomic_radius") for element in elements]
    positive_radii = _positive_values(radii)
    if positive_radii:
        log_radius = np.asarray(
            [math.log(radius) if radius > 0.0 else 0.0 for radius in radii],
            dtype=float,
        )
        features.append(float(math.exp(np.dot(fractions, log_radius))))
        features.append(float(max(positive_radii) / min(positive_radii)))
    else:
        features.extend([0.0, 0.0])

    return [_finite(value) for value in features]


def element_fraction_v1(comp: Composition) -> list[float]:
    reduced = comp.reduced_composition
    element_amounts = reduced.element_composition.get_el_amt_dict()
    total_atoms = float(sum(element_amounts.values()))

    features = size_thermo_v1(comp)
    fractions = [0.0] * 118
    for symbol, amount in element_amounts.items():
        element = Element(symbol)
        if 1 <= element.Z <= 118:
            fractions[element.Z - 1] = float(amount / total_atoms)
    features.extend(fractions)

    return [_finite(value) for value in features]
