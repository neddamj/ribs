# AGENTS.md — Research Code Guidelines

This is a **research codebase**, not a production system.

Prioritize:

1. Correct experimental logic.
2. Simple, readable implementations.
3. Reproducible experiments.
4. Good tests for important logic.
5. Clear experiment logging and saved results.
6. Code that is easy for an AI agent to run and interpret.

## Code Style

* Prefer the simplest correct implementation.
* Do not overengineer or add unnecessary abstractions.
* Use small, focused functions and descriptive names.
* Keep experiment logic separate from I/O and plotting where practical.
* Follow standard Python conventions.
* Use `black` for formatting, `ruff` for linting, and `pytest` for testing.
* Add comments only when they explain non-obvious reasoning or research-specific decisions.
* Use type hints where they improve clarity.

## Experiments

Experiments should be runnable from the command line without modifying source code.

Prefer explicit configuration files or CLI arguments for parameters such as:

* dataset
* model
* seed
* learning rate
* number of steps
* attack parameters
* batch size

Set and record random seeds.

Provide a lightweight way to test experiments, such as:

```bash
python run_experiment.py --config config.yaml --max-samples 2
```

Avoid interactive prompts so agents can run experiments autonomously.

## Testing

Use `pytest`.

Tests should focus on experimental correctness rather than coverage for its own sake.

Prioritize tests for:

* mathematical operations
* metrics
* preprocessing
* attack/update rules
* constraints and clipping
* result saving
* important edge cases

Use small synthetic inputs where possible.

Important experiment pipelines should have a lightweight smoke test.

## Experiment Results

Never rely only on terminal output.

Each run should save results to a unique directory, for example:

```text
results/
    experiment_name/
        run_id/
            config.yaml
            metrics.json
            run.log
            sample_metrics.csv
            artifacts/
```

Save important results in machine-readable formats such as JSON or CSV.

`metrics.json` should contain the main aggregate metrics and relevant metadata.

For example:

```json
{
  "status": "completed",
  "seed": 42,
  "model": "cheng2020_anchor",
  "dataset": "imagenette",
  "num_samples": 1000,
  "collision_rate": 0.843,
  "mean_psnr": 31.7
}
```

Save per-sample results separately when useful.

Do not overwrite previous experiment results.

## Logging

Use Python's `logging` module for important experiment information.

Log:

* experiment configuration
* major execution stages
* important metrics
* warnings/errors
* where results were saved

For substantial experiments, save logs to `run.log`.

Avoid excessive logging.

## Agent-Friendly Code

A coding agent should be able to determine:

* how to run an experiment
* what configuration was used
* whether the run succeeded
* where results were saved
* what the main results were

Prefer stable, predictable filenames and result schemas.

Experiment scripts should fail clearly and return errors rather than silently continuing.

Do not silently catch exceptions.

## Reproducibility

Record enough information to understand and reproduce a run, including where relevant:

* seed
* configuration
* model/checkpoint
* dataset/split
* Git commit
* important library versions

Exact determinism is preferred when practical. Document known sources of nondeterminism.

## Dependencies and Performance

Keep dependencies minimal.

Do not add libraries for functionality that can be implemented clearly with existing dependencies.

Prioritize:

```text
Correctness
→ Experimental validity
→ Reproducibility
→ Readability
→ Simplicity
→ Performance
→ Generality
```

Do not optimize prematurely.

## Agent Changes

When modifying the repository:

1. Inspect the relevant code and tests first.
2. Make the smallest change needed.
3. Avoid unrelated refactoring.
4. Format the code.
5. Run relevant tests.
6. Run a smoke test when appropriate.
7. Verify that expected result files are produced.

## Research Integrity

Do not change experimental methodology simply to obtain better results.

Do not silently:

* remove failed samples
* cherry-pick runs
* change metrics
* alter seeds to obtain favorable outcomes
* overwrite or manually modify results

Unexpected and negative results are valid research outcomes.
