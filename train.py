"""
Descriptor search runner on local composition-property data.

By default this script evaluates a descriptor with 3-fold CV inside train.csv
only. Use --evaluate-validation after a descriptor is kept to fit on all of
train.csv and evaluate validation.csv.

Usage:
    uv run python train.py
    uv run python train.py --evaluate-validation
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from automat_utils import (
    build_model_from_config,
    extract_xy,
    load_local_frame,
    make_featurizer,
    mean_absolute_prediction_error,
    predict_values,
)
from run_config import config_get, config_path, load_run_info_arg


def parse_args() -> argparse.Namespace:
    run_info_parser, config = load_run_info_arg()

    parser = argparse.ArgumentParser(
        description="Evaluate a composition descriptor on a local train.csv split.",
        parents=[run_info_parser],
    )
    parser.add_argument(
        "--descriptor-name",
        default=config_get(config, "descriptor.default_name"),
        help="Descriptor tag from descriptors.AVAILABLE_COMPOSITION_DESCRIPTORS.",
    )
    parser.add_argument(
        "--evaluate-validation",
        action="store_true",
        help="Fit all train.csv rows and evaluate validation.csv. Use only for kept descriptors.",
    )
    args = parser.parse_args()
    args.run_config = config
    args.data_dir = config_path(config, "data.dataset_dir")
    args.train_file = config_get(config, "data.train_file")
    args.validation_file = config_get(config, "data.validation_file")
    args.target_column = config_get(config, "data.target_column")
    args.composition_column = config_get(config, "data.composition_column")
    args.stratification_bins = int(config_get(config, "cv.stratification_bins"))
    args.cv_folds = int(config_get(config, "cv.folds"))
    args.random_seed = int(config_get(config, "model.random_seed"))
    return args


def make_stratification_labels(
    y_values: np.ndarray,
    n_splits: int,
    max_bins: int,
) -> np.ndarray | None:
    max_usable_bins = min(max_bins, max(2, len(y_values) // n_splits))
    for bins in range(max_usable_bins, 1, -1):
        try:
            labels = pd.qcut(y_values, q=bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        if labels is None:
            continue
        labels = np.asarray(labels, dtype=int)
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) >= 2 and int(counts.min()) >= n_splits:
            return labels
    return None


def cross_validate_train_set(
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, float]:
    labels = make_stratification_labels(
        y_train,
        n_splits=args.cv_folds,
        max_bins=args.stratification_bins,
    )
    if labels is None:
        splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
        splits = splitter.split(x_train)
    else:
        splitter = StratifiedKFold(
            n_splits=args.cv_folds,
            shuffle=True,
            random_state=args.random_seed,
        )
        splits = splitter.split(x_train, labels)

    fold_maes = []
    for fold_idx, (fit_idx, val_idx) in enumerate(splits, start=1):
        model = build_model_from_config(
            args.run_config,
            random_state=args.random_seed + fold_idx,
        )
        model.fit(x_train[fit_idx], y_train[fit_idx])
        mae = mean_absolute_prediction_error(
            y_train[val_idx],
            predict_values(model, x_train[val_idx]),
        )
        fold_maes.append(mae)
        print(f"  cv_fold {fold_idx:02d} | cv_mae: {mae:.6f}")

    return {
        "cv_mae": float(np.mean(fold_maes)),
        "cv_mae_std": float(np.std(fold_maes, ddof=0)),
        "cv_folds": float(args.cv_folds),
    }


def evaluate_descriptor(args: argparse.Namespace) -> dict[str, float | str]:
    featurize = make_featurizer(args.descriptor_name)
    train_frame = load_local_frame(
        data_dir=args.data_dir,
        filename=args.train_file,
        target_column=args.target_column,
        composition_column=args.composition_column,
    )
    train_inputs, y_train = extract_xy(train_frame, args.target_column, args.composition_column)
    x_train = featurize(train_inputs)

    print(f"train_rows: {x_train.shape[0]}")
    cv_metrics = cross_validate_train_set(
        args=args,
        x_train=x_train,
        y_train=y_train,
    )

    model = build_model_from_config(args.run_config, random_state=args.random_seed)
    model.fit(x_train, y_train)
    train_mae = mean_absolute_prediction_error(y_train, predict_values(model, x_train))

    metrics: dict[str, float | str] = {
        "target_column": args.target_column,
        "train_rows": float(x_train.shape[0]),
        "train_mae": train_mae,
        **cv_metrics,
        "validation_rows": float("nan"),
        "val_mae": float("nan"),
    }

    if args.evaluate_validation:
        val_frame = load_local_frame(
            data_dir=args.data_dir,
            filename=args.validation_file,
            target_column=args.target_column,
            composition_column=args.composition_column,
        )
        val_inputs, y_val = extract_xy(val_frame, args.target_column, args.composition_column)
        x_val = featurize(val_inputs)
        val_mae = mean_absolute_prediction_error(y_val, predict_values(model, x_val))
        print(f"validation_rows: {x_val.shape[0]}")
        metrics.update(
            {
                "validation_rows": float(x_val.shape[0]),
                "val_mae": val_mae,
            }
        )

    return metrics


def format_float(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if np.isnan(value):
        return "nan"
    return f"{value:.6f}"


def format_count(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if np.isnan(value):
        return "nan"
    return str(int(value))


def main() -> None:
    args = parse_args()

    t_start = time.time()
    print(f"Descriptor set: {args.descriptor_name}")
    print(f"Model: {config_get(args.run_config, 'model.name')}")
    print(f"Data directory: {args.data_dir}")
    print(f"Run info: {args.run_info}")
    print(
        "Evaluation contract: use train.csv CV for keep/discard; "
        "validation.csv only with --evaluate-validation."
    )

    metrics = evaluate_descriptor(args)
    t_end = time.time()

    print("---")
    print(f"cv_mae:           {format_float(metrics['cv_mae'])}")
    print(f"cv_mae_std:       {format_float(metrics['cv_mae_std'])}")
    print(f"train_mae:        {format_float(metrics['train_mae'])}")
    print(f"val_mae:          {format_float(metrics['val_mae'])}")
    print(f"train_seconds:    {t_end - t_start:.1f}")
    print(f"descriptor_set:   {args.descriptor_name}")
    print(f"target_column:    {metrics['target_column']}")
    print(f"train_rows:       {int(metrics['train_rows'])}")
    print(f"validation_rows:  {format_count(metrics['validation_rows'])}")
    print(f"cv_folds:         {int(metrics['cv_folds'])}")


if __name__ == "__main__":
    main()
