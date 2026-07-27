---
name: end-of-run-report
description: Generate a scientific end-of-run Markdown report for an automat autoresearch run by reconstructing descriptor proposals from Git commits referenced by results.tsv, comparing successful and unsuccessful ideas, evaluating validation performance against descriptor complexity, checking run integrity, and creating the run figure with plot_run_results.py. Manual invocation only
---

# Generate an End-of-Run Report

Create `end_of_run_report.md` and `end_of_run_results.png` in the run root. Treat Git history and the logged TSV files as the evidence base. Write a selective scientific narrative suitable for sharing; do not produce an iteration-by-iteration ledger.

## Preserve the run

Operate read-only except for the two report outputs and a temporary evidence file. Do not change run configuration, source code, Git state, TSV logs, or existing experiment artifacts. Do not commit the report. Overwrite existing report outputs when invoked.

Work from the repository root. Require:

- `run_info.yaml`
- the results and ideas paths configured under `logging`
- `plot_run_results.py`
- a Git worktree containing the experiment commits

If a required artifact is missing or Git commits cannot be resolved, collect all available evidence and explain the blocker. Do not invent proposal rationale or metrics.

## Collect evidence

Create a temporary JSON file outside the repository, then run:

```bash
uv run python .codex/skills/end-of-run-report/scripts/collect_run_evidence.py \
  --run-root . \
  --output /tmp/end_of_run_evidence.json
```

The collector:

- reads task, data, model, CV, metric, and stop settings from `run_info.yaml`;
- summarizes configured train and validation data without assuming a particular task name;
- matches `results.tsv` rows to `ideas.tsv`;
- resolves each result commit and reads `descriptors/idea.md` from that exact commit;
- preserves proposal text for kept, discarded, and crashed ideas;
- measures feature-vector length at historical commits for validation-tested keepers in isolated temporary archives;
- reports missing rows, unresolved commits, descriptor mismatches, unavailable proposals, and complexity failures.

Read the complete JSON. Inspect `program.md` as well when present to understand the run's keep/discard and validation contract. Use direct `git show <commit>:descriptors/idea.md` checks when evidence is surprising or incomplete.

Treat `results.tsv` as the metric record and the historical `idea.md` files as the scientific rationale. Never infer success from the current working-tree implementation alone. Distinguish:

- train/CV selection performance;
- validation audit performance;
- descriptor complexity, using feature count plus structural simplicity;
- operational status such as `keep`, `discard`, or `crash`.

## Generate the figure

Generate the figure only through the run's existing plotting script:

```bash
uv run ./plot_run_results.py ideas.tsv results.tsv -o end_of_run_results.png
```

If the configured TSV paths differ from the defaults, pass those paths instead. Do not recreate or modify plotting logic inside the skill. Verify that the PNG exists and is non-empty. If plotting fails, still generate the report and disclose the missing figure and reason.

## Analyze the run

Reconstruct the scientific progression across related proposal families. Group experiments by the underlying hypothesis and describe only the important turning points.

Identify what worked using evidence such as strict improvement in the configured keep metric, strong validation audits, successful simplification, or a useful new descriptor family. Identify what did not work using discarded regressions, unproductive additions, failed ablations, crashes, or repeated hypotheses. Explain plausible scientific or modeling reasons grounded in the committed proposal documents; label interpretations as interpretations rather than facts.

Do not equate `keep` with best validation performance. Respect the run's selection contract while comparing validation results for the final recommendation.

Select two to four best descriptor candidates for the conclusion. A candidate must have a recorded finite validation metric and a defensible complexity/performance tradeoff. Give priority to descriptors near the best validation result, then prefer fewer features and simpler scientific structure. Include the exact descriptor name, logged short commit hash, validation metric, selection metric when available, feature count when measurable, and a concise justification. Include the selection-metric winner when it remains scientifically credible, even if another candidate has the absolute best validation audit.

## Write the report

Write `end_of_run_report.md` with exactly these second-level sections and no others:

```markdown
# End-of-Run Report: <task name>

## Introduction

## Results

## Conclusion
```

Use third-level headings only if essential; prefer a cohesive narrative with bold lead-ins inside `Results`.

In `Introduction`, describe the task, available dataset splits and sizes, target/input columns, model, CV design, selection metric, validation role, number of experiments, and relevant target distribution characteristics. Keep absent test data distinct from missing required data.

In `Results`, explain the proposal progression, important successes, important failures, simplification outcomes, and the difference between CV and validation findings. Embed the generated figure after first introducing the overall trajectory:

```markdown
![Autoresearch metric history and descriptor lineage](end_of_run_results.png)

*Figure 1. <Specific caption explaining both panels and the status/lineage encoding.>*
```

Follow the caption with a short interpretation of what the figure shows; do not merely restate the caption. Mention exact metrics and commits where they materially support the narrative. Report integrity problems clearly in this section without allowing a non-critical mismatch to prevent report generation.

In `Conclusion`, synthesize the scientific lessons and present the best descriptor candidates in prose. Do not use a standalone experiment table. State uncertainty and tradeoffs honestly; validation audits are evidence, not an additional tuning loop.

Use scientific, publication-ready language. Avoid process commentary, raw JSON, exhaustive iteration lists, unsupported causal claims, and generic filler.

## Verify and clean up

Confirm that:

- the report contains only the three required second-level sections;
- all stated metrics, commit hashes, statuses, feature counts, and dataset facts match the evidence;
- important claims about proposals are supported by historical committed `idea.md` content;
- every recommended candidate includes its descriptor name and commit hash;
- the figure link resolves and its caption describes the actual plot;
- integrity issues are disclosed;
- only `end_of_run_report.md` and `end_of_run_results.png` were created or overwritten in the repository.

Delete the temporary evidence JSON after verification.