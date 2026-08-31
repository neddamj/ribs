import pandas as pd

from ribs.plotting import render_phase1


def test_phase1_figure_fixture_renders(tmp_path):
    summary = pd.DataFrame(
        {
            "family": ["dimensional", "dimensional", "identity"],
            "attack": ["input_pgd", "input_pgd", "input_pgd"],
            "dz": [16, 512, 512],
            "seed": [0, 0, 0],
            "robust_auc": [0.4, 0.6, 0.7],
        }
    )
    summary_path = tmp_path / "phase1_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    outputs = render_phase1(summary_path, tmp_path / "figures")
    assert len(outputs) == 1
    assert outputs[0].is_file()
