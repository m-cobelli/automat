from __future__ import annotations

import math
import warnings

import numpy as np
from pymatgen.core import Composition, Element


MAGNETIC_3D = {"Cr", "Mn", "Fe", "Co", "Ni"}
COMMON_MAGNETIC = MAGNETIC_3D | {"Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Nd", "Sm", "Eu"}
PNICTOGENS = {"N", "P", "As", "Sb", "Bi"}
CHALCOGENS = {"O", "S", "Se", "Te"}
HALOGENS = {"F", "Cl", "Br", "I"}
LIGHT_ATOMS = {"B", "C", "N", "O", "F"}
ELEMENT_BASIS = [Element.from_Z(atomic_number).symbol for atomic_number in range(1, 95)]
MAGNETIC_IDENTITY = ["Fe", "Co", "Ni", "Mn", "Cr", "Gd", "Tb", "Dy", "Nd", "Sm", "Eu"]
MAGNETIC_PAIR_BASIS = ["Fe", "Co", "Ni", "Mn", "Cr", "Gd", "Nd", "Sm"]
FCN = {"Fe", "Co", "Ni"}
MN_CR = {"Mn", "Cr"}
UNPAIRED_PROXY = {
    "Cr": 6.0,
    "Mn": 5.0,
    "Fe": 4.0,
    "Co": 3.0,
    "Ni": 2.0,
    "Gd": 7.0,
    "Tb": 6.0,
    "Dy": 5.0,
    "Ho": 4.0,
    "Er": 3.0,
    "Tm": 2.0,
    "Nd": 3.0,
    "Sm": 5.0,
    "Eu": 7.0,
}


def _safe_float(value) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _safe_attr(element: Element, attr: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(element, attr, None)


def _flag(element: Element, attr: str) -> float:
    return 1.0 if bool(getattr(element, attr, False)) else 0.0


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> list[float]:
    mean = float(np.dot(weights, values))
    centered = values - mean
    std = float(np.sqrt(np.dot(weights, centered * centered)))
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    return [
        mean,
        std,
        min_value,
        max_value,
        max_value - min_value,
        float(np.max(weights * values)),
    ]


def _pairwise_contrasts(values: np.ndarray, weights: np.ndarray) -> list[float]:
    if len(values) < 2:
        return [0.0, 0.0]

    max_delta = 0.0
    weighted_sum = 0.0
    weight_total = 0.0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            delta = abs(float(values[i] - values[j]))
            pair_weight = float(weights[i] * weights[j])
            max_delta = max(max_delta, delta)
            weighted_sum += pair_weight * delta
            weight_total += pair_weight
    mean_delta = weighted_sum / weight_total if weight_total else 0.0
    return [max_delta, mean_delta]


def _element_property(element: Element, name: str) -> float:
    if name == "atomic_number":
        return _safe_float(element.Z)
    if name == "atomic_mass":
        return _safe_float(element.atomic_mass)
    if name == "row":
        return _safe_float(element.row)
    if name == "group":
        return _safe_float(element.group)
    if name == "electronegativity":
        return _safe_float(element.X)
    if name == "mendeleev_no":
        return _safe_float(element.mendeleev_no)
    if name == "atomic_radius":
        return _safe_float(_safe_attr(element, "atomic_radius"))
    if name == "metallic_radius":
        return _safe_float(_safe_attr(element, "metallic_radius"))
    raise KeyError(name)


def baseline(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    items = list(reduced.element_composition.items())
    elements = [Element(str(element)) for element, _ in items]
    amounts = np.asarray([float(amount) for _, amount in items], dtype=float)
    total_amount = float(np.sum(amounts))
    weights = amounts / total_amount

    max_fraction = float(np.max(weights))
    min_fraction = float(np.min(weights))
    entropy = -float(np.sum(weights * np.log(weights)))

    features: list[float] = [
        float(len(elements)),
        total_amount,
        entropy,
        max_fraction,
        min_fraction,
        max_fraction - min_fraction,
    ]

    set_fractions: dict[str, float] = {}
    symbol_sets = [
        ("magnetic_3d", MAGNETIC_3D),
        ("common_magnetic", COMMON_MAGNETIC),
        ("pnictogen", PNICTOGENS),
        ("chalcogen", CHALCOGENS),
        ("halogen", HALOGENS),
        ("light_atom", LIGHT_ATOMS),
        ("oxygen", {"O"}),
    ]
    for label, symbols in symbol_sets:
        mask = np.asarray([element.symbol in symbols for element in elements], dtype=float)
        fraction = float(np.dot(weights, mask))
        set_fractions[label] = fraction
        features.extend([fraction, float(np.sum(mask))])

    flag_fractions: dict[str, float] = {}
    flag_attrs = [
        "is_transition_metal",
        "is_rare_earth_metal",
        "is_alkali",
        "is_alkaline",
        "is_metal",
        "is_metalloid",
        "is_nonmetal",
    ]
    for attr in flag_attrs:
        mask = np.asarray([_flag(element, attr) for element in elements], dtype=float)
        fraction = float(np.dot(weights, mask))
        flag_fractions[attr] = fraction
        features.extend([fraction, float(np.sum(mask))])

    property_names = [
        "atomic_number",
        "atomic_mass",
        "row",
        "group",
        "electronegativity",
        "mendeleev_no",
        "atomic_radius",
        "metallic_radius",
    ]
    property_values: dict[str, np.ndarray] = {}
    for name in property_names:
        values = np.asarray([_element_property(element, name) for element in elements], dtype=float)
        property_values[name] = values
        features.extend(_weighted_stats(values, weights))

    for name in property_names:
        features.extend(_pairwise_contrasts(property_values[name], weights))

    magnetic_fraction = set_fractions["magnetic_3d"]
    common_magnetic_fraction = set_fractions["common_magnetic"]
    chalcogen_fraction = set_fractions["chalcogen"]
    oxygen_fraction = set_fractions["oxygen"]
    transition_fraction = flag_fractions["is_transition_metal"]
    rare_earth_fraction = flag_fractions["is_rare_earth_metal"]
    electronegativity_range = _weighted_stats(property_values["electronegativity"], weights)[4]

    features.extend(
        [
            magnetic_fraction * transition_fraction,
            common_magnetic_fraction * transition_fraction,
            common_magnetic_fraction * rare_earth_fraction,
            common_magnetic_fraction * oxygen_fraction,
            common_magnetic_fraction * chalcogen_fraction,
            common_magnetic_fraction * electronegativity_range,
            transition_fraction * total_amount,
            max_fraction * common_magnetic_fraction,
        ]
    )

    clean = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in clean]


def magnetic_network_v1(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    items = list(reduced.element_composition.items())
    elements = [Element(str(element)) for element, _ in items]
    amounts = np.asarray([float(amount) for _, amount in items], dtype=float)
    total_amount = float(np.sum(amounts))
    weights = amounts / total_amount
    fractions = {element.symbol: float(weight) for element, weight in zip(elements, weights)}

    features = baseline(composition)

    for symbol in MAGNETIC_IDENTITY:
        features.append(fractions.get(symbol, 0.0))

    fcn_fraction = sum(fractions.get(symbol, 0.0) for symbol in FCN)
    mn_cr_fraction = sum(fractions.get(symbol, 0.0) for symbol in MN_CR)
    magnetic_3d_fraction = sum(fractions.get(symbol, 0.0) for symbol in MAGNETIC_3D)
    rare_earth_fraction = float(
        np.dot(weights, np.asarray([_flag(element, "is_rare_earth_metal") for element in elements]))
    )
    transition_fraction = float(
        np.dot(weights, np.asarray([_flag(element, "is_transition_metal") for element in elements]))
    )
    metallic_fraction = float(
        np.dot(weights, np.asarray([_flag(element, "is_metal") for element in elements]))
    )
    oxygen_fraction = fractions.get("O", 0.0)
    pnictogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in PNICTOGENS)
    chalcogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in CHALCOGENS)

    unpaired_values = np.asarray(
        [UNPAIRED_PROXY.get(element.symbol, 0.0) for element in elements],
        dtype=float,
    )
    unpaired_mean = float(np.dot(weights, unpaired_values))
    unpaired_max_contribution = float(np.max(weights * unpaired_values))

    magnetic_fractions = sorted(
        [fractions.get(symbol, 0.0) for symbol in COMMON_MAGNETIC],
        reverse=True,
    )
    largest_magnetic = magnetic_fractions[0] if magnetic_fractions else 0.0
    second_magnetic = magnetic_fractions[1] if len(magnetic_fractions) > 1 else 0.0
    dominance_ratio = largest_magnetic / (second_magnetic + 1.0e-6)
    common_magnetic_fraction = sum(fractions.get(symbol, 0.0) for symbol in COMMON_MAGNETIC)

    pair_features = [
        sum(fractions.get(a, 0.0) * fractions.get(b, 0.0) for a in FCN for b in FCN if a < b),
        sum(
            fractions.get(a, 0.0) * fractions.get(b, 0.0)
            for a in MAGNETIC_3D
            for b in MAGNETIC_3D
            if a < b
        ),
        rare_earth_fraction * magnetic_3d_fraction,
        magnetic_3d_fraction * oxygen_fraction,
        magnetic_3d_fraction * pnictogen_fraction,
        magnetic_3d_fraction * chalcogen_fraction,
    ]

    electronegativities = np.asarray(
        [_element_property(element, "electronegativity") for element in elements],
        dtype=float,
    )
    electronegativity_range = float(np.max(electronegativities) - np.min(electronegativities))

    features.extend(
        [
            fcn_fraction,
            max(fractions.get(symbol, 0.0) for symbol in FCN),
            mn_cr_fraction,
            magnetic_3d_fraction,
            rare_earth_fraction,
            unpaired_mean,
            unpaired_max_contribution,
            unpaired_mean * common_magnetic_fraction,
            largest_magnetic,
            second_magnetic,
            dominance_ratio,
            1.0 - common_magnetic_fraction,
            common_magnetic_fraction * total_amount,
            fcn_fraction * metallic_fraction,
            mn_cr_fraction * oxygen_fraction,
            rare_earth_fraction * transition_fraction,
            common_magnetic_fraction * electronegativity_range,
        ]
    )
    features.extend(pair_features)

    clean = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in clean]


def _common_oxidation_summary(element: Element) -> tuple[float, float, float, float, float]:
    states = [_safe_float(state) for state in element.common_oxidation_states]
    states = [state for state in states if state != 0.0]
    if not states:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    positives = [state for state in states if state > 0.0]
    negatives = [state for state in states if state < 0.0]
    min_state = min(states)
    max_state = max(states)
    positive_mean = float(np.mean(positives)) if positives else 0.0
    negative_mean = float(np.mean(negatives)) if negatives else 0.0
    return (min_state, max_state, positive_mean, negative_mean, max_state - min_state)


def valence_context_v1(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    items = list(reduced.element_composition.items())
    elements = [Element(str(element)) for element, _ in items]
    amounts = np.asarray([float(amount) for _, amount in items], dtype=float)
    total_amount = float(np.sum(amounts))
    weights = amounts / total_amount
    fractions = {element.symbol: float(weight) for element, weight in zip(elements, weights)}

    features = magnetic_network_v1(composition)

    summaries = np.asarray([_common_oxidation_summary(element) for element in elements], dtype=float)
    for column in range(summaries.shape[1]):
        values = summaries[:, column]
        stats = _weighted_stats(values, weights)
        features.extend(stats[:5])

    anion_symbols = PNICTOGENS | CHALCOGENS | HALOGENS | {"B", "C"}
    anion_fraction = sum(fractions.get(symbol, 0.0) for symbol in anion_symbols)
    oxygen_fraction = fractions.get("O", 0.0)
    halogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in HALOGENS)
    pnictogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in PNICTOGENS)
    chalcogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in CHALCOGENS)
    cation_fraction = max(0.0, 1.0 - anion_fraction)

    max_states = summaries[:, 1]
    min_states = summaries[:, 0]
    positive_means = summaries[:, 2]
    negative_means = summaries[:, 3]
    spans = summaries[:, 4]
    optimistic_charge = float(np.dot(weights, max_states))
    reducing_charge = float(np.dot(weights, min_states))
    mixed_charge = float(np.dot(weights, np.where(positive_means > 0.0, positive_means, negative_means)))

    magnetic_fraction = sum(fractions.get(symbol, 0.0) for symbol in COMMON_MAGNETIC)
    fcn_fraction = sum(fractions.get(symbol, 0.0) for symbol in FCN)
    mn_cr_fraction = sum(fractions.get(symbol, 0.0) for symbol in MN_CR)
    rare_earth_fraction = float(
        np.dot(weights, np.asarray([_flag(element, "is_rare_earth_metal") for element in elements]))
    )
    span_mean = float(np.dot(weights, spans))

    features.extend(
        [
            anion_fraction,
            cation_fraction,
            oxygen_fraction,
            halogen_fraction,
            pnictogen_fraction,
            chalcogen_fraction,
            cation_fraction / (anion_fraction + 1.0e-6),
            abs(optimistic_charge),
            abs(reducing_charge),
            abs(mixed_charge),
            optimistic_charge - reducing_charge,
            magnetic_fraction * span_mean,
            fcn_fraction * anion_fraction,
            mn_cr_fraction * oxygen_fraction,
            mn_cr_fraction * halogen_fraction,
            rare_earth_fraction * chalcogen_fraction,
            rare_earth_fraction * pnictogen_fraction,
        ]
    )

    clean = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in clean]


def element_fraction_v1(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    items = list(reduced.element_composition.items())
    elements = [Element(str(element)) for element, _ in items]
    amounts = np.asarray([float(amount) for _, amount in items], dtype=float)
    total_amount = float(np.sum(amounts))
    weights = amounts / total_amount
    fractions = {element.symbol: float(weight) for element, weight in zip(elements, weights)}

    features = valence_context_v1(composition)
    basis_values = np.asarray([fractions.get(symbol, 0.0) for symbol in ELEMENT_BASIS], dtype=float)
    features.extend(float(value) for value in basis_values)

    atomic_numbers = np.asarray([float(Element(symbol).Z) for symbol in ELEMENT_BASIS], dtype=float)
    scaled_z = atomic_numbers / 94.0
    for power in range(1, 5):
        features.append(float(np.dot(basis_values, scaled_z**power)))

    nonzero = basis_values[basis_values > 0.0]
    sorted_nonzero = np.sort(nonzero)[::-1]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) if len(nonzero) else 0.0
    max_fraction = float(sorted_nonzero[0]) if len(sorted_nonzero) else 0.0
    second_fraction = float(sorted_nonzero[1]) if len(sorted_nonzero) > 1 else 0.0
    herfindahl = float(np.sum(basis_values * basis_values))

    fcn_fraction = sum(fractions.get(symbol, 0.0) for symbol in FCN)
    mn_cr_fraction = sum(fractions.get(symbol, 0.0) for symbol in MN_CR)
    rare_earth_fraction = float(
        np.dot(weights, np.asarray([_flag(element, "is_rare_earth_metal") for element in elements]))
    )
    light_anion_fraction = sum(fractions.get(symbol, 0.0) for symbol in LIGHT_ATOMS)
    oxygen_fraction = fractions.get("O", 0.0)
    magnetic_fraction = sum(fractions.get(symbol, 0.0) for symbol in COMMON_MAGNETIC)

    features.extend(
        [
            max_fraction,
            second_fraction,
            float(np.count_nonzero(basis_values)),
            herfindahl,
            entropy,
            fcn_fraction,
            mn_cr_fraction,
            rare_earth_fraction,
            light_anion_fraction,
            oxygen_fraction,
            fcn_fraction * oxygen_fraction,
            mn_cr_fraction * oxygen_fraction,
            rare_earth_fraction * fcn_fraction,
            light_anion_fraction * magnetic_fraction,
            magnetic_fraction * herfindahl,
        ]
    )

    clean = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in clean]


def magnetic_pair_identity_v1(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    items = list(reduced.element_composition.items())
    elements = [Element(str(element)) for element, _ in items]
    amounts = np.asarray([float(amount) for _, amount in items], dtype=float)
    weights = amounts / float(np.sum(amounts))
    fractions = {element.symbol: float(weight) for element, weight in zip(elements, weights)}

    features = element_fraction_v1(composition)

    magnetic_pair_products: list[float] = []
    for idx, first in enumerate(MAGNETIC_PAIR_BASIS):
        for second in MAGNETIC_PAIR_BASIS[idx + 1 :]:
            product = fractions.get(first, 0.0) * fractions.get(second, 0.0)
            magnetic_pair_products.append(product)
            features.append(product)

    pnictogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in PNICTOGENS)
    chalcogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in CHALCOGENS)
    halogen_fraction = sum(fractions.get(symbol, 0.0) for symbol in HALOGENS)
    light_anion_fraction = sum(fractions.get(symbol, 0.0) for symbol in LIGHT_ATOMS)
    anion_contexts = [
        fractions.get("O", 0.0),
        pnictogen_fraction,
        chalcogen_fraction,
        halogen_fraction,
        light_anion_fraction,
    ]

    magnetic_anion_products: list[float] = []
    for symbol in MAGNETIC_IDENTITY:
        symbol_fraction = fractions.get(symbol, 0.0)
        for context_fraction in anion_contexts:
            product = symbol_fraction * context_fraction
            magnetic_anion_products.append(product)
            features.append(product)

    fcn_fraction = sum(fractions.get(symbol, 0.0) for symbol in FCN)
    mn_cr_fraction = sum(fractions.get(symbol, 0.0) for symbol in MN_CR)
    rare_earth_fraction = float(
        np.dot(weights, np.asarray([_flag(element, "is_rare_earth_metal") for element in elements]))
    )
    magnetic_fraction = sum(fractions.get(symbol, 0.0) for symbol in COMMON_MAGNETIC)
    nonmagnetic_fraction = 1.0 - magnetic_fraction

    features.extend(
        [
            fcn_fraction * rare_earth_fraction,
            mn_cr_fraction * rare_earth_fraction,
            magnetic_fraction * nonmagnetic_fraction,
            float(np.sum(magnetic_pair_products)),
            float(np.max(magnetic_pair_products)) if magnetic_pair_products else 0.0,
            float(np.sum(magnetic_anion_products)),
            float(np.max(magnetic_anion_products)) if magnetic_anion_products else 0.0,
        ]
    )

    clean = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in clean]


def shell_block_refinement_v1(composition: Composition) -> list[float]:
    reduced = composition.reduced_composition
    items = list(reduced.element_composition.items())
    elements = [Element(str(element)) for element, _ in items]
    amounts = np.asarray([float(amount) for _, amount in items], dtype=float)
    weights = amounts / float(np.sum(amounts))
    fractions = {element.symbol: float(weight) for element, weight in zip(elements, weights)}

    features = magnetic_pair_identity_v1(composition)

    block_fractions = {}
    for block in ["s", "p", "d", "f"]:
        mask = np.asarray([1.0 if _safe_attr(element, "block") == block else 0.0 for element in elements])
        block_fractions[block] = float(np.dot(weights, mask))

    df_mask = np.asarray(
        [1.0 if _safe_attr(element, "block") in {"d", "f"} else 0.0 for element in elements],
        dtype=float,
    )
    df_weight = float(np.dot(weights, df_mask))
    if df_weight > 0.0:
        df_weights = weights * df_mask / df_weight
        groups = np.asarray([_safe_float(element.group) for element in elements])
        rows = np.asarray([_safe_float(element.row) for element in elements])
        df_group_mean = float(np.dot(df_weights, groups))
        df_group_std = float(np.sqrt(np.dot(df_weights, (groups - df_group_mean) ** 2)))
        df_row_mean = float(np.dot(df_weights, rows))
    else:
        df_group_mean = 0.0
        df_group_std = 0.0
        df_row_mean = 0.0

    magnetic_fraction = sum(fractions.get(symbol, 0.0) for symbol in COMMON_MAGNETIC)

    features.extend(
        [
            block_fractions["s"],
            block_fractions["p"],
            block_fractions["d"],
            block_fractions["f"],
            block_fractions["d"] * block_fractions["p"],
            block_fractions["d"] * block_fractions["f"],
            block_fractions["f"] * block_fractions["p"],
            df_group_mean,
            df_group_std,
            df_row_mean,
            magnetic_fraction * block_fractions["d"],
            magnetic_fraction * block_fractions["f"],
            magnetic_fraction * block_fractions["p"],
            df_weight,
        ]
    )

    clean = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) for value in clean]


def no_element_summary_block_v1(composition: Composition) -> list[float]:
    features = shell_block_refinement_v1(composition)
    del features[276:295]
    return features
