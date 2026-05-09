from __future__ import annotations

from collections.abc import Callable

from pymatgen.core import Composition

AVAILABLE_COMPOSITION_DESCRIPTORS: dict[str, Callable[[Composition], list[float]]] = {}

