"""Figures regenerated from machine-readable result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def render_phase1(summary_path: str | Path, output_dir: str | Path) -> list[Path]:
    summary = (
        pd.read_parquet(summary_path)
        if str(summary_path).endswith("parquet")
        else pd.read_csv(summary_path)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for attack in summary.attack.dropna().unique() if "attack" in summary else []:
        data = summary[summary.attack == attack]
        if "family" in data:
            data = data[data.family == "dimensional"]
        x = "dz" if "dz" in data else "bits"
        if x not in data or "robust_auc" not in data:
            continue
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.lineplot(data=data, x=x, y="robust_auc", marker="o", errorbar="sd", ax=axis)
        axis.set_title(f"{attack}: robustness AUC")
        axis.set_ylabel("Robust-accuracy AUC")
        figure.tight_layout()
        path = output_dir / f"{attack}_auc.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    curves_path = Path(summary_path).with_name("phase1_curves.parquet")
    if curves_path.exists():
        curves = pd.read_parquet(curves_path)
        for attack in curves.attack.dropna().unique():
            data = curves[curves.attack == attack]
            if "family" in data:
                data = data[data.family == "dimensional"]
            if data.empty:
                continue
            figure, axis = plt.subplots(figsize=(7, 5))
            sns.lineplot(
                data=data,
                x="radius",
                y="robust_accuracy_nested",
                hue="dz" if "dz" in data else None,
                style="seed" if "seed" in data else None,
                marker="o",
                ax=axis,
            )
            axis.set_title(f"{attack}: robust-accuracy curves")
            axis.set_ylabel("Robust accuracy")
            figure.tight_layout()
            path = output_dir / f"{attack}_curves.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(path)
    return outputs
