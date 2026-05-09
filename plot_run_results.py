#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.26",
#   "pandas>=2.1",
# ]
# ///

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REQUIRED_IDEA_COLUMNS = {
    "commit",
    "parent_commit",
    "change_kind",
}
REQUIRED_RESULT_COLUMNS = {
    "commit",
    "cv_mae",
    "val_mae",
    "status",
}

STATUS_COLORS = {
    "keep": "#2E8B57",
    "discard": "#DC2626",
    "crash": "#111111",
    "pending": "#9CA3AF",
}
CV_COLOR = "#C97A3A"
VAL_COLOR = "#2E8B57"
CHANGE_KIND_COLORS = {
    "new_family": "#3B6FB6",
    "feature_addition": "#059669",
    "feature_removal": "#dc2626",
    "feature_refinement": "#7c3aed",
    "feature_refined": "#7c3aed",
}


def change_kind_color(value: object) -> str:
    return CHANGE_KIND_COLORS.get(str(value).strip(), "#6b7280")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one autoresearch run from ideas.tsv and results.tsv."
    )
    parser.add_argument(
        "ideas",
        nargs="?",
        type=Path,
        default=Path("ideas.tsv"),
        help="Idea lineage TSV. Defaults to ideas.tsv.",
    )
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=Path("results.tsv"),
        help="Run results TSV. Defaults to results.tsv.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("run_results.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def read_tsv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    frame = pd.read_csv(path, sep="\t")
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def normalize_parent(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text


def prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    frame["commit"] = frame["commit"].astype(str).str.strip()
    frame["status"] = frame["status"].astype(str).str.strip().str.lower()
    frame["cv_mae"] = pd.to_numeric(frame["cv_mae"], errors="coerce")
    frame["val_mae"] = pd.to_numeric(frame["val_mae"], errors="coerce")
    frame.loc[frame["status"] == "crash", ["cv_mae", "val_mae"]] = np.nan
    frame["iteration"] = np.arange(1, len(frame) + 1)
    frame["best_cv_mae"] = frame["cv_mae"].cummin()
    return frame


def prepare_ideas(ideas: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    frame = ideas.copy()
    frame["commit"] = frame["commit"].astype(str).str.strip()
    frame["parent_commit"] = frame["parent_commit"].map(normalize_parent)

    result_columns = results[["commit", "iteration", "status"]]
    frame = frame.merge(result_columns, on="commit", how="left")
    frame["status"] = frame["status"].fillna("pending")
    frame["iteration"] = frame["iteration"].fillna(0).astype(int)
    return assign_lineage_positions(frame)


def assign_lineage_positions(ideas: pd.DataFrame) -> pd.DataFrame:
    frame = ideas.copy()
    parent_by_commit = dict(zip(frame["commit"], frame["parent_commit"]))
    depth_cache: dict[str, int] = {}

    def depth_for(commit: str) -> int:
        if commit in depth_cache:
            return depth_cache[commit]
        parent = parent_by_commit.get(commit)
        if parent is None or parent not in parent_by_commit:
            depth_cache[commit] = 0
        else:
            depth_cache[commit] = depth_for(parent) + 1
        return depth_cache[commit]

    frame["depth"] = [depth_for(commit) for commit in frame["commit"]]
    frame = frame.sort_values(["depth", "iteration", "commit"]).reset_index(drop=True)

    y_positions: dict[str, float] = {}
    for depth, depth_frame in frame.groupby("depth", sort=True):
        count = len(depth_frame)
        offsets = np.arange(count, dtype=float) - (count - 1) / 2.0
        for commit, offset in zip(depth_frame["commit"], offsets):
            y_positions[commit] = -offset

    frame["lineage_y"] = frame["commit"].map(y_positions)
    return frame


def plot_metric_history(ax: plt.Axes, results: pd.DataFrame) -> None:
    ax.step(
        results["iteration"],
        results["best_cv_mae"],
        color=CV_COLOR,
        linewidth=2.8,
        where="post",
        label="best cv_mae",
        zorder=2,
    )
    for status, rows in results.groupby("status", sort=False):
        if status == "keep":
            scatter_kwargs = {
                "facecolors": CV_COLOR,
                "edgecolors": "white",
                "linewidths": 0.9,
            }
        elif status == "discard":
            scatter_kwargs = {
                "facecolors": "white",
                "edgecolors": CV_COLOR,
                "linewidths": 1.6,
            }
        else:
            scatter_kwargs = {
                "facecolors": STATUS_COLORS.get(status, "#6b7280"),
                "edgecolors": "white",
                "linewidths": 0.9,
            }
        ax.scatter(
            rows["iteration"],
            rows["cv_mae"],
            s=58,
            marker="o",
            label=status,
            zorder=3,
            **scatter_kwargs,
        )

    validation_rows = results[results["val_mae"].notna()]
    if not validation_rows.empty:
        ax.step(
            validation_rows["iteration"],
            validation_rows["val_mae"],
            color=VAL_COLOR,
            linewidth=2.2,
            where="post",
            alpha=0.78,
            label="validation audit",
            zorder=2,
        )
        ax.scatter(
            validation_rows["iteration"],
            validation_rows["val_mae"],
            s=50,
            marker="D",
            facecolors=VAL_COLOR,
            edgecolors=VAL_COLOR,
            linewidths=1.0,
            label="val_mae",
            zorder=4,
        )

    ax.set_xlabel("Iteration", labelpad=8)
    ax.set_ylabel("MAE")
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.6, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=4, width=1.0, color="#555555", pad=5)
    ax.legend(frameon=False, ncols=2, loc="upper right")


def plot_lineage(ax: plt.Axes, ideas: pd.DataFrame) -> None:
    by_commit = ideas.set_index("commit")
    for _, row in ideas.iterrows():
        parent = row["parent_commit"]
        if parent is None or parent not in by_commit.index:
            continue
        parent_row = by_commit.loc[parent]
        ax.plot(
            [parent_row["depth"], row["depth"]],
            [parent_row["lineage_y"], row["lineage_y"]],
            color=change_kind_color(row["change_kind"]),
            linewidth=1.55,
            alpha=0.72,
            zorder=1,
        )

    for status, rows in ideas.groupby("status", sort=False):
        face_color = STATUS_COLORS.get(status, "#6b7280")
        ax.scatter(
            rows["depth"],
            rows["lineage_y"],
            s=116,
            marker="o",
            facecolors=face_color,
            edgecolors="#111111",
            linewidths=1.25,
            label=status,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(axis="x", color="#EAEAEA", linewidth=0.7, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(length=4, width=1.0, color="#555555", pad=5)

    status_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111111",
            markerfacecolor=STATUS_COLORS[status],
            markeredgecolor="#111111",
            linewidth=0,
            markersize=7.5,
            label=status,
        )
        for status in sorted(set(ideas["status"]))
        if status in STATUS_COLORS
    ]
    change_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=2,
            label=change_kind,
        )
        for change_kind, color in CHANGE_KIND_COLORS.items()
        if change_kind != "new_family" and change_kind in set(ideas["change_kind"])
    ]
    ax.legend(
        handles=status_handles + change_handles,
        frameon=False,
        ncols=2,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=8.5,
    )


def main() -> None:
    args = parse_args()
    ideas = read_tsv(args.ideas, REQUIRED_IDEA_COLUMNS)
    results = prepare_results(read_tsv(args.results, REQUIRED_RESULT_COLUMNS))
    ideas = prepare_ideas(ideas, results)

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9.5,
            "legend.handlelength": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.4),
        gridspec_kw={"height_ratios": [1.18, 1.0]},
        constrained_layout=False,
    )
    plot_metric_history(axes[0], results)
    plot_lineage(axes[1], ideas)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.09, hspace=0.46)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
