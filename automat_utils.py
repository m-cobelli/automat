from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from descriptors import AVAILABLE_COMPOSITION_DESCRIPTORS
from run_config import config_get

SUPPORTED_MODEL = "random_forest_regressor"


def validate_columns(
    frame: pd.DataFrame,
    target_column: str,
    composition_column: str,
) -> None:
    if composition_column not in frame.columns:
        raise ValueError(f"Missing required composition column: {composition_column}")
    if target_column not in frame.columns:
        raise ValueError(f"Missing required target column: {target_column}")


def load_local_frame(
    data_dir: Path,
    filename: str,
    target_column: str,
    composition_column: str,
) -> pd.DataFrame:
    path = data_dir / filename
    frame = pd.read_csv(path)
    validate_columns(frame, target_column, composition_column)
    return frame


def extract_xy(
    frame: pd.DataFrame,
    target_column: str,
    composition_column: str,
) -> tuple[pd.Series, np.ndarray]:
    return frame[composition_column], frame[target_column].to_numpy(dtype=float)


def normalize_formula(value) -> str:
    if isinstance(value, Composition):
        return value.formula
    return str(value)


def make_featurizer(descriptor_name: str):
    try:
        descriptor_fn = AVAILABLE_COMPOSITION_DESCRIPTORS[descriptor_name]
    except KeyError as exc:
        available = ", ".join(sorted(AVAILABLE_COMPOSITION_DESCRIPTORS))
        raise KeyError(
            f"Unknown descriptor '{descriptor_name}'. Available descriptors: {available}"
        ) from exc

    @lru_cache(maxsize=None)
    def featurize_formula(formula: str) -> tuple[float, ...]:
        comp = Composition(formula)
        features = np.asarray(descriptor_fn(comp), dtype=np.float32)
        if features.ndim != 1:
            raise ValueError(
                f"Descriptor '{descriptor_name}' must return a 1D feature vector, "
                f"got shape {features.shape}."
            )
        return tuple(float(x) for x in features)

    def featurize(values) -> np.ndarray:
        formulas = [normalize_formula(value) for value in values]
        x = np.asarray([featurize_formula(formula) for formula in formulas], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {x.shape}")
        return x

    return featurize


def predict_values(model, x_values: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x_values), dtype=float)


def mean_absolute_prediction_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def build_model_from_config(
    config: dict,
    *,
    random_state: int,
) -> RandomForestRegressor:
    model_name = config_get(config, "model.name")
    if model_name != SUPPORTED_MODEL:
        raise ValueError(
            f"Unsupported model.name '{model_name}'. Supported: {SUPPORTED_MODEL}"
        )
    return RandomForestRegressor(
        n_estimators=int(config_get(config, "model.n_estimators")),
        max_depth=config_get(config, "model.max_depth"),
        min_samples_split=int(config_get(config, "model.min_samples_split")),
        min_samples_leaf=int(config_get(config, "model.min_samples_leaf")),
        max_features=float(config_get(config, "model.max_features")),
        n_jobs=int(config_get(config, "model.n_jobs")),
        random_state=random_state,
    )
