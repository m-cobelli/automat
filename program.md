# automat 

This is the normative operating spec for agents running autonomous descriptor
search in this repo. 

## Core Contract

`automat` searches for composition-only descriptors for materials
regression tasks.

- Input features must be derived from chemical formulas only.
- Each run uses the pre-split local data declared in `run_info.yaml`.
- `data.target_column` must be explicit in `run_info.yaml`.
- The model is fixed for the entire run. Use the model and parameters declared
  in `run_info.yaml`; do not tune or change them mid-run.
- Descriptor keep/discard decisions use only train-set CV, normally `cv_mae`.
- Validation is evaluated only for accepted descriptors and never decides branch
  state.
- The final test file is manually added only after autoresearch is complete.
  Never use it during descriptor search.
- `run_info.yaml` is immutable after setup unless the user explicitly directs a
  correction; changing it normally means starting a new run.

## Required Files

Read these before starting or resuming a run:

- `run_info.yaml` - task, data paths, columns, model, CV, stop criteria, and log paths.
- `train.py` - train-CV evaluator and optional validation audit.
- `run_status.py` - stop/continue checker.
- `automat_utils.py` - loading, featurization, model, and metric helpers.
- `descriptors/idea.md` - current descriptor proposal.
- `descriptors/idea.py` - current descriptor implementation.
- `descriptors/__init__.py` - registry only.

## Setup

Default setup creates a new branch unless the user says otherwise.

1. Read `run_info.yaml`.
2. Validate required inputs:
   - `run_info.yaml` exists.
   - `task.name` and `task.description` are present.
   - `data.dataset_dir`, `data.train_file`, and `data.validation_file` resolve to existing files.
   - `data.test_file` names the eventual final holdout file, but the file is not
     required at setup and must not be used during autoresearch.
   - `data.composition_column` exists in both CSVs.
   - `data.target_column` is explicit and exists in both CSVs.
   - `model.name`, CV settings, autoresearch settings, and logging paths are present.
3. Derive the project tag from a sanitized `task.name`.
4. Derive the date tag from the actual setup date, such as `may01`.
5. Create `automat/<project-tag>/<date-tag>` from the current main branch. If it
   already exists, ask the user or choose a non-conflicting suffix.
6. Create local `results.tsv` and `ideas.tsv` if missing. Do not commit them.
7. Start from blank or absent `descriptors/idea.py`. `descriptors/idea.md` may
   contain only a generic template that instructs the agent to create a baseline
   from `run_info.yaml`.
8. Generate the run baseline from the task description in `run_info.yaml`.
   Document it in `descriptors/idea.md` before implementing it in
   `descriptors/idea.py`.
9. Register the baseline in `descriptors/__init__.py`. That file must contain
   only imports and `AVAILABLE_COMPOSITION_DESCRIPTORS`.
10. Run any quick practical smoke check you need to catch import or registry
    errors.
11. Commit the baseline code.
12. Run the baseline experiment and validation audit. Log it as the root node.

The baseline is generated fresh for the run. 

## Local Logs

`results.tsv` and `ideas.tsv` are required local artifacts, but they are not
committed.

`results.tsv` header:

```text
commit	cv_mae	cv_mae_std	val_mae	status	descriptor_name	description
```

`ideas.tsv` header:

```text
commit	parent_commit	root_commit	descriptor_name	change_kind	risk_level
```

`results.tsv` columns:

1. `commit`: exact short git hash for the experiment commit.
2. `cv_mae`: train-CV MAE, or `inf` for logged crashes.
3. `cv_mae_std`: fold MAE standard deviation, or `nan` for crashes.
4. `val_mae`: validation MAE for kept descriptors, otherwise `nan`.
5. `status`: `keep`, `discard`, or `crash`.
6. `descriptor_name`: unique descriptor key used by the runner.
7. `description`: short description of the descriptor change.

`ideas.tsv` columns:

1. `commit`: exact short git hash for the experiment commit.
2. `parent_commit`: parent idea-node hash, or `null` for the root baseline.
3. `root_commit`: root baseline hash for the lineage.
4. `descriptor_name`: unique descriptor key used by the runner.
5. `change_kind`: `new_family`, `feature_addition`, `feature_removal`, or `feature_refinement`.
6. `risk_level`: `low`, `medium`, or `high`.

Every row in `results.tsv` must have exactly one matching row in `ideas.tsv`.
The hashes must match real experiment commits exactly. Discarded and crashed
experiments are still logged. Revert/helper commits are not idea nodes.

## Descriptor Design Rules

Before changing descriptor code, write the proposal in `descriptors/idea.md`.

`descriptors/idea.md` is the working design document for the current
agent-authored descriptor. It forces the agent to justify the descriptor design,
supports interrupted-run restarts, and improves reproducibility. It must be
self-contained: if another agent receives only this file, that agent should have
enough natural-language instruction to reproduce the same descriptor idea.

Update `descriptors/idea.md` on every iteration before implementing
`descriptors/idea.py`. It must contain exactly these sections:

- `Problem Knowledge`: short summary of the problem, enriched by insights from
  previous iterations.
- `Scientific Insight`: physical and chemical considerations relevant to the
  problem, and how they shape the current descriptor.
- `Implementation Strategy`: natural-language descriptor plan grounded in
  machine-learning intuition and physical insight. Describe the descriptor
  clearly enough to implement from this file alone. Do not include code.
- `Dependencies`: Python libraries or files to rely on for implementing the
  current idea. Do not self-reference other repo files as part of the idea.

Descriptors may use any deterministic formula-derived information from
`pymatgen` and local code. They must:

- use no validation labels or external task data
- be computable from composition only
- return a one-dimensional finite numeric vector for every composition
- be grounded in a physical or chemical argument relevant to the task

The autonomous loop may overwrite `descriptors/idea.md` and
`descriptors/idea.py` each iteration. Git history plus the local TSVs preserve
the run lineage.

## Evaluation Commands

Run train-CV only:

```bash
uv run python train.py > run.log 2>&1
```

For kept descriptors, run validation audit:

```bash
uv run python train.py --evaluate-validation > validation.log 2>&1
```

Extract metrics from the printed summaries. Keep/discard comparisons should use
enough metric precision from the run output, not informal visual judgment.
`train.py` is evaluation-only; it must not append to `results.tsv`.

`test_descriptors.py` stays separate and is not part of autoresearch. Use it
only after the user manually adds the final holdout file named by
`data.test_file`. It fits the selected descriptor and fixed model on
`train.csv` plus `validation.csv`, evaluates on `test.csv`, and can export final
test predictions. Do not run it for descriptor selection.

## Keep/Discard Policy

The root baseline is kept by definition and receives validation immediately.

After that, a descriptor is kept only if its `cv_mae` strictly improves over the
current best `cv_mae`. Ties and worse results are discarded.

- If kept: run validation, log `status=keep`, keep the commit as the new best.
- If discarded: log `status=discard` with `val_mae=nan`, then reset back to the
  previous best commit.
- If crashed: fix obvious implementation mistakes and retry before logging. If
  the idea is fundamentally broken, log `status=crash`; it counts as an
  iteration.

Validation results never override CV selection. A CV keeper remains the active
best even if validation worsens.

### Novelty Requirement

Before implementing a descriptor, compare the proposed idea against all prior rows in `ideas.tsv` and the current git history.

Do not run an iteration whose descriptor is functionally equivalent to a prior descriptor under a new name. Renaming, reordering identical features, adding duplicate features, or reusing the same template with unchanged parameters does not count as a new descriptor.

If a proposed descriptor is similar to a prior discarded descriptor, `descriptors/idea.md` must explicitly explain what is scientifically or algorithmically different this time.

### Simplicity criterion

A small performance improvement is not worth it if it adds unnecessary or messy complexity. On the other hand, if removing something gives equal or better results, that is a strong outcome.

Do not keep adding features blindly. Be mindful of the descriptor size keep it under 400 max the. The smaller and performant the better. If the descriptors become very large, you probably do not need all of them.

Focus on bespoke, task-relevant features that are likely to help with the specific problem you are trying to solve.

## Stop Policy

At the end of each logged iteration, run:

```bash
uv run python run_status.py
```

Continue only if the final line is:

```text
CONTINUE
```

Stop if the final line is:

```text
STOP
```

## Experiment Loop

Repeat until `run_status.py` says `STOP`:

1. Confirm current branch, current best commit, best `cv_mae`, root commit, and
   local TSV state.
2. Propose the next descriptor from the task description and prior results.
3. Update `descriptors/idea.md`.
4. Implement the new descriptor in `descriptors/idea.py`.
5. Add the new descriptor unique name key in `descriptors/__init__.py`.
6. Ensure the selected descriptor name key is what `train.py` will evaluate.
7. Commit the experiment.
8. Resolve the short commit hash and append the matching `ideas.tsv` row.
9. Run train-CV.
10. If the run crashed, decide whether to fix and retry or log a crash.
11. Compare `cv_mae` to the current best using strict improvement.
12. For keepers, run validation.
13. Append the `results.tsv` row.
14. Keep the commit if it improved; otherwise reset to the previous best.
15. Run `run_status.py`.

## Execution Discipline

Each experiment iteration must be performed manually and sequentially by the agent following the `Experiment Loop` steps exactly.

Do not create, run, or rely on any helper script, generated driver, batch loop, candidate generator, meta-optimizer, shell loop, Python loop, or other automation that performs multiple descriptor iterations. The agent may run only the explicitly documented commands for the current single iteration, plus small one-off inspection commands needed to read files, validate imports, or parse the current run output.

Do not pre-generate a list of future descriptors. Do not cycle through descriptor templates. Do not choose descriptors from a scripted schedule. Each descriptor proposal must be newly reasoned from:
- `run_info.yaml`
- `descriptors/idea.md`
- current `results.tsv`
- current `ideas.tsv`
- the current best commit
- prior logged outcomes