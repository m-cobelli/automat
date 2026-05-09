from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from run_config import DEFAULT_RUN_INFO, config_get, config_path, load_run_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect results.tsv and decide whether autoresearch should stop."
    )
    parser.add_argument(
        "--run-info",
        type=Path,
        default=DEFAULT_RUN_INFO,
        help="YAML file containing run metadata and stop criteria.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help="Override results TSV path.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only STOP or CONTINUE.",
    )
    return parser.parse_args()


def parse_metric(value: str | None) -> float:
    if value is None:
        return float("nan")
    value = value.strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def count_validation_since_last_improvement(
    rows: list[dict[str, str]],
    metric_name: str,
    lower_is_better: bool,
) -> tuple[int, float]:
    best_value = float("inf") if lower_is_better else -float("inf")
    since_improvement = 0

    for row in rows:
        value = parse_metric(row.get(metric_name))
        if np.isnan(value):
            continue

        improved = value < best_value if lower_is_better else value > best_value
        if improved:
            best_value = value
            since_improvement = 0
        else:
            since_improvement += 1

    return since_improvement, best_value


def main() -> None:
    args = parse_args()
    config = load_run_info(args.run_info)
    results_file = args.results_file or config_path(config, "logging.results_file")
    max_iterations = int(config_get(config, "autoresearch.max_iterations"))
    validation_patience = int(config_get(config, "autoresearch.validation_patience"))
    validation_metric = str(config_get(config, "autoresearch.validation_metric"))
    lower_is_better = bool(config_get(config, "autoresearch.lower_is_better"))

    rows = read_results(results_file)
    iterations = len(rows)
    since_improvement, best_value = count_validation_since_last_improvement(
        rows=rows,
        metric_name=validation_metric,
        lower_is_better=lower_is_better,
    )

    max_iterations_reached = max_iterations > 0 and iterations >= max_iterations
    patience_reached = validation_patience > 0 and since_improvement >= validation_patience
    decision = "STOP" if max_iterations_reached or patience_reached else "CONTINUE"

    if not args.quiet:
        best_display = (
            "nan" if np.isnan(best_value) or np.isinf(best_value) else f"{best_value:.6f}"
        )
        print(f"results_file: {results_file}")
        print(f"iterations: {iterations}")
        print(f"max_iterations: {max_iterations}")
        print(f"validation_metric: {validation_metric}")
        print(f"best_validation_metric: {best_display}")
        print(f"non_nan_validation_since_last_improvement: {since_improvement}")
        print(f"validation_patience: {validation_patience}")
        print(f"max_iterations_reached: {str(max_iterations_reached).lower()}")
        print(f"validation_patience_reached: {str(patience_reached).lower()}")
    print(decision)


if __name__ == "__main__":
    main()
