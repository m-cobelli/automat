from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from automat_utils import (
    build_model_from_config,
    extract_xy,
    load_local_frame,
    make_featurizer,
    mean_absolute_prediction_error,
    predict_values,
)
from descriptors import AVAILABLE_COMPOSITION_DESCRIPTORS
from run_config import config_get, config_path, load_run_info_arg


def parse_args() -> argparse.Namespace:
    run_info_parser, config = load_run_info_arg()

    parser = argparse.ArgumentParser(
        description=(
            "Final held-out test evaluation. Fits train.csv plus validation.csv "
            "and evaluates the manually added test.csv."
        ),
        parents=[run_info_parser],
    )
    parser.add_argument(
        "descriptor_name",
        nargs="?",
        default=config_get(config, "descriptor.default_name"),
        help="Descriptor tag from descriptors.AVAILABLE_COMPOSITION_DESCRIPTORS.",
    )
    parser.add_argument(
        "--list-descriptors",
        action="store_true",
        help="Print available descriptor tags and exit.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write final test predictions as CSV.",
    )
    args = parser.parse_args()

    if not args.list_descriptors and not args.descriptor_name:
        parser.error("descriptor_name is required unless --list-descriptors is used")

    args.run_config = config
    args.data_dir = config_path(config, "data.dataset_dir")
    args.train_file = config_get(config, "data.train_file")
    args.validation_file = config_get(config, "data.validation_file")
    args.test_file = config_get(config, "data.test_file")
    args.target_column = config_get(config, "data.target_column")
    args.composition_column = config_get(config, "data.composition_column")
    args.random_seed = int(config_get(config, "model.random_seed"))
    return args


def run_evaluation(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, float]]:
    featurize = make_featurizer(args.descriptor_name)
    train_frame = load_local_frame(
        data_dir=args.data_dir,
        filename=args.train_file,
        target_column=args.target_column,
        composition_column=args.composition_column,
    )
    val_frame = load_local_frame(
        data_dir=args.data_dir,
        filename=args.validation_file,
        target_column=args.target_column,
        composition_column=args.composition_column,
    )
    test_path = args.data_dir / args.test_file
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing final holdout test file: {test_path}. "
            "Add it manually only after autoresearch is complete."
        )
    test_frame = load_local_frame(
        data_dir=args.data_dir,
        filename=args.test_file,
        target_column=args.target_column,
        composition_column=args.composition_column,
    )

    fit_frame = pd.concat([train_frame, val_frame], ignore_index=True)
    fit_inputs, y_fit = extract_xy(fit_frame, args.target_column, args.composition_column)
    test_inputs, y_test = extract_xy(test_frame, args.target_column, args.composition_column)
    x_fit = featurize(fit_inputs)
    x_test = featurize(test_inputs)

    model = build_model_from_config(args.run_config, random_state=args.random_seed)
    model.fit(x_fit, y_fit)

    predictions = predict_values(model, x_test)
    test_mae = mean_absolute_prediction_error(y_test, predictions)

    output_frame = test_frame.copy()
    output_frame["prediction"] = predictions
    output_frame["absolute_error"] = np.abs(output_frame[args.target_column] - predictions)
    return output_frame, {"mae": test_mae}


def main() -> None:
    args = parse_args()

    if args.list_descriptors:
        for name in sorted(AVAILABLE_COMPOSITION_DESCRIPTORS):
            print(name)
        return

    predictions, metrics = run_evaluation(args)

    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"Saved final test predictions to {args.output}")

    print("---")
    print(f"test_mae:  {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
