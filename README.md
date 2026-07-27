# automat

`automat` is a minimal autoresearch harness for designing
composition-only descriptors for materials regression tasks. It is based on the
autoresearch paradigm introduced by A. Karpathy:
https://github.com/karpathy/autoresearch.

The detailed research protocol is provided in `program.md`. At the beginning of each iteration, the agent is instructed to justify the scientific reasoning behind newly proposed descriptor ideas in `descriptors/idea.md`.

## Quick Start

1. Install dependencies:

```bash
uv sync
```

2. Prepare a local dataset with non-overlapping, pre-split CSV files:

```text
data/<task-name>/train.csv
data/<task-name>/validation.csv
data/<task-name>/test.csv
```

`train.csv`, `validation.csv`, and `test.csv` must use the same composition
column and target column. The test split should remain untouched until the final
evaluation.

3. Update `run_info.yaml`.

Set the task description, dataset path, CSV filenames, composition column,
target column, model settings, logging paths, and stopping criteria.

At minimum, check:

```yaml
task:
  name: Tc
  description: Predict experimental Curie temperature of ferromagnets from chemical formula only.

data:
  dataset_dir: data/Tc
  train_file: train.csv
  validation_file: validation.csv
  test_file: test.csv
  composition_column: composition
  target_column: Tc
```

4. Start a new autoresearch run in Codex or Claude Code with:

```text
Set up a new experiment run. Follow strictly the directives in program.md.
```

5. Continue the run with:

```text
Continue performing new iterations, strictly following the instructions in program.md. Continue until run_status.py says STOP.
```

The agent will propose descriptors, implement them, evaluate them, commit each
experiment, and maintain the local run logs according to `program.md`.

## Stopping

The halting logic is implemented in `run_status.py` and configured through
`run_info.yaml`. By default, it can stop after a maximum number of iterations or
when validation patience is exhausted. You can extend `run_status.py` if your
experiment needs additional stopping criteria.

## Final Evaluation

Once the autoresearch run stops, insights can be gained by auditing the logs and the various commits. To facilitate this process, we provide an `end-of-run-report` skill in the `skills` folder. This skill can be invoked to automatically generate a report that summarises the run’s findings and helps the user select the appropriate descriptors.

To evaluate the selected descriptor on the untouched test
split:

```bash
uv run python test_descriptors.py <descriptor_name> --output test_predictions.csv
```

## Main Files

- `program.md` - agent instructions.
- `run_info.yaml` - run configuration.
- `train.py` - train-CV and validation evaluator.
- `run_status.py` - halting decision logic.
- `test_descriptors.py` - final test evaluator.
- `plot_run_results.py` - plotting script for visualizing autoresearch progress.
- `descriptors/idea.md` - current descriptor proposal.
- `descriptors/idea.py` - current descriptor implementation.
- `descriptors/__init__.py` - descriptor registry.
