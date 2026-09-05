import pandas as pd

from ribs.plotting import render_phase1, render_phase2


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


def test_phase2_figures_include_curves_clean_accuracy_and_synthesis(tmp_path):
    summary = pd.DataFrame(
        {
            "family": ["vib", "vib"],
            "attack": ["input_pgd", "latent_pgd"],
            "seed": [0, 0],
            "bottleneck_strength_ordinal": [0, 0],
            "robust_auc": [0.6, 0.5],
            "clean_accuracy": [0.8, 0.8],
            "median_nearest_opposing_distance": [0.3, 0.3],
        }
    )
    summary_path = tmp_path / "phase2_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    pd.DataFrame(
        {
            "family": ["vib", "vib"],
            "attack": ["input_pgd", "input_pgd"],
            "seed": [0, 0],
            "bottleneck_strength_ordinal": [0, 0],
            "radius": [0.0, 0.1],
            "robust_accuracy_nested": [0.8, 0.5],
        }
    ).to_parquet(tmp_path / "phase2_curves.parquet", index=False)
    pd.DataFrame(
        {
            "family": ["vib"],
            "local_robustness_auc": [0.6],
            "semantic_separation_robustness": [0.3],
        }
    ).to_parquet(tmp_path / "phase2_synthesis.parquet", index=False)

    outputs = render_phase2(summary_path, tmp_path / "figures")
    names = {path.name for path in outputs}
    assert "phase2_vib_input_pgd_curves.png" in names
    assert "phase2_vib_clean_accuracy.png" in names
    assert "phase2_local_vs_semantic_separation.png" in names
