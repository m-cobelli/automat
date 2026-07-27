#!/usr/bin/env python3
"""Collect provenance-rich evidence for an automat end-of-run report."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


PROPOSAL_PATH = "descriptors/idea.md"
NON_METRIC_COLUMNS = {
    "commit",
    "parent_commit",
    "root_commit",
    "descriptor_name",
    "change_kind",
    "risk_level",
    "status",
    "description",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect run configuration, data, Git proposals, results, and complexity."
    )
    parser.add_argument("--run-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-info", type=Path, default=Path("run_info.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-complexity",
        action="store_true",
        help="Do not evaluate descriptor dimensions at historical commits.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    value = yaml.load(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def nested_get(mapping: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric_metrics(row: dict[str, str]) -> dict[str, float | None]:
    return {
        key: finite_float(value)
        for key, value in row.items()
        if key not in NON_METRIC_COLUMNS
    }


def git(run_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=run_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def resolve_commit(run_root: Path, commit: str) -> tuple[str | None, str | None]:
    try:
        full_hash = git(
            run_root, "rev-parse", "--verify", f"{commit}^{{commit}}"
        ).stdout.strip()
        subject = git(run_root, "show", "-s", "--format=%s", full_hash).stdout.strip()
        return full_hash, subject
    except subprocess.CalledProcessError:
        return None, None


def proposal_at_commit(run_root: Path, commit: str) -> str | None:
    try:
        return git(run_root, "show", f"{commit}:{PROPOSAL_PATH}").stdout.strip()
    except subprocess.CalledProcessError:
        return None


def safe_extract_tar(payload: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def descriptor_dimension_at_commit(
    run_root: Path,
    commit: str,
    descriptor_name: str,
    formula: str,
) -> tuple[int | None, str | None]:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit, "descriptors"],
        cwd=run_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if archive.returncode != 0:
        return None, archive.stderr.decode("utf-8", errors="replace").strip()

    snippet = """
import json
import sys
from pymatgen.core import Composition
from descriptors import AVAILABLE_COMPOSITION_DESCRIPTORS

name, formula = sys.argv[1], sys.argv[2]
values = AVAILABLE_COMPOSITION_DESCRIPTORS[name](Composition(formula))
print(json.dumps({"dimension": len(values)}))
"""
    with tempfile.TemporaryDirectory(prefix="automat-report-") as temp_dir:
        destination = Path(temp_dir)
        safe_extract_tar(archive.stdout, destination)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(destination)
        process = subprocess.run(
            [sys.executable, "-c", snippet, descriptor_name, formula],
            cwd=destination,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
    if process.returncode != 0:
        return None, process.stderr.strip()
    try:
        return int(json.loads(process.stdout)["dimension"]), None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return (
            None,
            f"Could not parse descriptor dimension output: {process.stdout.strip()}",
        )


def summarize_csv(
    path: Path,
    target_column: str | None,
    composition_column: str | None,
) -> tuple[dict[str, Any], set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    missing_cells = sum(
        value is None or value.strip() == "" for row in rows for value in row.values()
    )
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": len(rows),
        "columns": columns,
        "missing_cells": missing_cells,
    }

    formulas: set[str] = set()
    if composition_column and composition_column in columns:
        values = [row[composition_column].strip() for row in rows]
        formulas = set(values)
        summary["unique_compositions"] = len(formulas)
        summary["duplicate_compositions"] = len(values) - len(formulas)

    if target_column and target_column in columns:
        raw_targets = [
            row[target_column].strip() for row in rows if row[target_column].strip()
        ]
        numeric_targets = [finite_float(value) for value in raw_targets]
        if raw_targets and all(value is not None for value in numeric_targets):
            numbers = [float(value) for value in numeric_targets if value is not None]
            ordered = sorted(numbers)
            summary["target"] = {
                "column": target_column,
                "kind": "numeric",
                "count": len(numbers),
                "unique": len(set(numbers)),
                "minimum": min(numbers),
                "median": statistics.median(numbers),
                "mean": statistics.fmean(numbers),
                "maximum": max(numbers),
                "zero_count": sum(value == 0.0 for value in numbers),
                "q1": ordered[(len(ordered) - 1) // 4],
                "q3": ordered[(3 * (len(ordered) - 1)) // 4],
            }
        else:
            counts = Counter(raw_targets)
            summary["target"] = {
                "column": target_column,
                "kind": "categorical",
                "count": len(raw_targets),
                "unique": len(counts),
                "most_common": counts.most_common(10),
            }
    elif target_column:
        summary["target_error"] = f"Missing configured target column: {target_column}"
    return summary, formulas


def duplicate_values(rows: list[dict[str, str]], key: str) -> list[str]:
    counts = Counter(row.get(key, "").strip() for row in rows)
    return sorted(value for value, count in counts.items() if value and count > 1)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    run_info_path = (
        args.run_info if args.run_info.is_absolute() else run_root / args.run_info
    )
    config = load_yaml(run_info_path)

    errors: list[str] = []
    warnings: list[str] = []
    results_path = run_root / str(
        nested_get(config, "logging.results_file", "results.tsv")
    )
    ideas_path = run_root / str(nested_get(config, "logging.ideas_file", "ideas.tsv"))
    plot_path = run_root / "plot_run_results.py"
    for required in (results_path, ideas_path, plot_path):
        if not required.exists():
            errors.append(
                f"Missing required artifact: {required.relative_to(run_root)}"
            )
    if errors:
        evidence = {
            "run_root": str(run_root),
            "integrity": {"errors": errors, "warnings": warnings},
        }
        write_evidence(evidence, args.output)
        raise SystemExit(1)

    result_columns, results = read_tsv(results_path)
    idea_columns, ideas = read_tsv(ideas_path)
    result_commits = [row.get("commit", "").strip() for row in results]
    idea_commits = [row.get("commit", "").strip() for row in ideas]
    result_set, idea_set = set(result_commits), set(idea_commits)
    result_duplicates = duplicate_values(results, "commit")
    idea_duplicates = duplicate_values(ideas, "commit")
    if result_duplicates:
        errors.append(f"Duplicate results commits: {result_duplicates}")
    if idea_duplicates:
        errors.append(f"Duplicate idea commits: {idea_duplicates}")
    for commit in sorted(idea_set - result_set):
        warnings.append(f"Idea commit {commit} has no matching results row")
    for commit in sorted(result_set - idea_set):
        warnings.append(f"Results commit {commit} has no matching ideas row")

    ideas_by_commit = {row.get("commit", "").strip(): row for row in ideas}
    dataset_dir = run_root / str(nested_get(config, "data.dataset_dir", "data"))
    target_column = nested_get(config, "data.target_column")
    composition_column = nested_get(config, "data.composition_column")
    dataset_summaries: dict[str, Any] = {}
    split_formulas: dict[str, set[str]] = {}
    representative_formula: str | None = None
    for split_key in ("train_file", "validation_file", "test_file"):
        filename = nested_get(config, f"data.{split_key}")
        if not filename:
            continue
        path = dataset_dir / str(filename)
        split_name = split_key.removesuffix("_file")
        if not path.exists():
            dataset_summaries[split_name] = {"path": str(path), "present": False}
            if split_name != "test":
                errors.append(
                    f"Missing configured {split_name} data file: {path.relative_to(run_root)}"
                )
            continue
        summary, formulas = summarize_csv(path, target_column, composition_column)
        summary["present"] = True
        summary["path"] = str(path.relative_to(run_root))
        dataset_summaries[split_name] = summary
        split_formulas[split_name] = formulas
        if representative_formula is None and formulas:
            representative_formula = sorted(formulas)[0]
    if "train" in split_formulas and "validation" in split_formulas:
        dataset_summaries["train_validation_composition_overlap"] = len(
            split_formulas["train"] & split_formulas["validation"]
        )

    keep_metric = str(nested_get(config, "autoresearch.keep_metric", "cv_mae"))
    validation_metric = str(
        nested_get(config, "autoresearch.validation_metric", "val_mae")
    )
    lower_is_better = bool(nested_get(config, "autoresearch.lower_is_better", True))
    experiments: list[dict[str, Any]] = []
    for iteration, result in enumerate(results, start=1):
        commit = result.get("commit", "").strip()
        idea = ideas_by_commit.get(commit)
        full_hash, subject = resolve_commit(run_root, commit)
        if full_hash is None:
            errors.append(f"Results commit {commit} cannot be resolved by Git")
        proposal = proposal_at_commit(run_root, commit) if full_hash else None
        if proposal is None:
            warnings.append(f"Proposal document unavailable at commit {commit}")
        if idea and idea.get("descriptor_name") != result.get("descriptor_name"):
            warnings.append(
                f"Descriptor name differs between results and ideas at {commit}: "
                f"{result.get('descriptor_name')} vs {idea.get('descriptor_name')}"
            )
        metrics = numeric_metrics(result)
        complexity: int | None = None
        should_measure = (
            not args.skip_complexity
            and representative_formula is not None
            and result.get("status", "").strip().lower() == "keep"
            and metrics.get(validation_metric) is not None
            and full_hash is not None
            and bool(result.get("descriptor_name"))
        )
        if should_measure:
            complexity, complexity_error = descriptor_dimension_at_commit(
                run_root, commit, result["descriptor_name"], representative_formula
            )
            if complexity_error:
                warnings.append(
                    f"Could not measure descriptor complexity at {commit}: {complexity_error}"
                )
        experiments.append(
            {
                "iteration": iteration,
                "commit": commit,
                "full_commit": full_hash,
                "commit_subject": subject,
                "descriptor_name": result.get("descriptor_name"),
                "status": result.get("status"),
                "description": result.get("description"),
                "metrics": metrics,
                "idea": idea,
                "proposal": proposal,
                "feature_count": complexity,
            }
        )

    def best_experiment(metric: str) -> dict[str, Any] | None:
        eligible = [
            item for item in experiments if item["metrics"].get(metric) is not None
        ]
        if not eligible:
            return None

        def metric_value(item: dict[str, Any]) -> float:
            return item["metrics"][metric]

        chosen = (
            min(eligible, key=metric_value)
            if lower_is_better
            else max(eligible, key=metric_value)
        )
        return {
            "commit": chosen["commit"],
            "descriptor_name": chosen["descriptor_name"],
            "value": chosen["metrics"][metric],
            "feature_count": chosen["feature_count"],
        }

    evidence = {
        "run_root": str(run_root),
        "artifacts": {
            "run_info": str(run_info_path.relative_to(run_root)),
            "results": str(results_path.relative_to(run_root)),
            "ideas": str(ideas_path.relative_to(run_root)),
            "plot_script": str(plot_path.relative_to(run_root)),
        },
        "task": config.get("task", {}),
        "data_config": config.get("data", {}),
        "datasets": dataset_summaries,
        "model": config.get("model", {}),
        "cv": config.get("cv", {}),
        "autoresearch": config.get("autoresearch", {}),
        "results_columns": result_columns,
        "ideas_columns": idea_columns,
        "summary": {
            "experiment_count": len(experiments),
            "idea_count": len(ideas),
            "status_counts": dict(Counter(item["status"] for item in experiments)),
            "keep_metric": keep_metric,
            "validation_metric": validation_metric,
            "lower_is_better": lower_is_better,
            "best_keep_metric": best_experiment(keep_metric),
            "best_validation_metric": best_experiment(validation_metric),
        },
        "integrity": {"errors": errors, "warnings": warnings},
        "experiments": experiments,
    }
    write_evidence(evidence, args.output)
    if errors:
        raise SystemExit(1)


def write_evidence(evidence: dict[str, Any], output_path: Path | None) -> None:
    output = json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote evidence to {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
