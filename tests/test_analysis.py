import json

import pandas as pd
import yaml

from ribs.analysis import aggregate_completed_runs


def test_aggregate_completed_run_fixture(tmp_path):
    run_dir = tmp_path / "dimensional" / "run"
    evaluation_dir = run_dir / "evaluations" / "robustness-fixture-attempt0"
    evaluation_dir.mkdir(parents=True)
    config = {
        "seed": 0,
        "model": {"family": "dimensional", "dz": 512},
        "attack": {"input_epsilons": [0.0, 1.0], "latent_rhos": [1.0]},
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (run_dir / "COMPLETED").write_text("completed\n")
    (run_dir / "metrics.json").write_text(json.dumps({"codebook_collapsed": False}))
    (evaluation_dir / "config.json").write_text(json.dumps({"split": "final", "max_samples": None}))
    (evaluation_dir / "COMPLETED").write_text("completed\n")
    records = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "radius": [0.0, 1.0, 0.0, 1.0],
            "clean_correct": [True, True, True, True],
            "successful": [False, True, False, False],
        }
    )
    records.to_parquet(evaluation_dir / "input_pgd.parquet", index=False)
    records.to_parquet(evaluation_dir / "latent_pgd.parquet", index=False)

    summary = aggregate_completed_runs(tmp_path)

    assert set(summary.attack) == {"input_pgd", "latent_pgd"}
    assert (tmp_path / "phase1_hierarchical_bootstrap.parquet").is_file()
