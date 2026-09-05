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


def render_phase2(summary_path: str | Path, output_dir: str | Path) -> list[Path]:
    """Render explicitly labeled within-family and ordinal cross-family plots."""
    summary = (
        pd.read_parquet(summary_path)
        if str(summary_path).endswith("parquet")
        else pd.read_csv(summary_path)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if summary.empty or "family" not in summary:
        return outputs
    for family, family_frame in summary.groupby("family", sort=True):
        for attack, attack_frame in family_frame.groupby("attack", sort=True):
            if "bottleneck_strength_ordinal" not in attack_frame:
                continue
            figure, axis = plt.subplots(figsize=(7, 5))
            sns.lineplot(
                data=attack_frame,
                x="bottleneck_strength_ordinal",
                y="robust_auc",
                marker="o",
                errorbar="sd",
                ax=axis,
            )
            axis.set_title(f"Phase 2 {family}: {attack} robustness AUC")
            axis.set_xlabel("Within-family ordinal bottleneck strength (0 = weakest)")
            axis.set_ylabel("Normalized robust-accuracy AUC")
            figure.tight_layout()
            path = output_dir / f"phase2_{family}_{attack}_auc.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(path)
    for attack, attack_frame in summary.groupby("attack", sort=True):
        figure, axis = plt.subplots(figsize=(8, 5))
        sns.lineplot(
            data=attack_frame,
            x="bottleneck_strength_ordinal",
            y="robust_auc",
            hue="family",
            marker="o",
            errorbar="sd",
            ax=axis,
        )
        axis.set_title(f"Phase 2 cross-family {attack} (ordinal scale only)")
        axis.set_xlabel("Ordinal bottleneck strength within each family")
        axis.set_ylabel("Normalized robust-accuracy AUC")
        figure.tight_layout()
        path = output_dir / f"phase2_cross_family_{attack}_auc.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    curves_path = Path(summary_path).with_name("phase2_curves.parquet")
    if curves_path.exists():
        curves = pd.read_parquet(curves_path)
        for (family, attack), data in curves.groupby(["family", "attack"], sort=True):
            if data.empty:
                continue
            figure, axis = plt.subplots(figsize=(7, 5))
            sns.lineplot(
                data=data,
                x="radius",
                y="robust_accuracy_nested",
                hue="bottleneck_strength_ordinal",
                units="seed" if "seed" in data else None,
                estimator=None if "seed" in data else "mean",
                marker="o",
                palette="viridis",
                ax=axis,
            )
            axis.set_title(f"Phase 2 {family}: {attack} robust-accuracy curves")
            axis.set_ylabel("Robust accuracy")
            axis.legend(title="Bottleneck strength")
            figure.tight_layout()
            path = output_dir / f"phase2_{family}_{attack}_curves.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(path)
    clean = summary[summary.attack == "input_pgd"].copy()
    for family, data in clean.groupby("family", sort=True):
        if "clean_accuracy" not in data:
            continue
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.lineplot(
            data=data,
            x="bottleneck_strength_ordinal",
            y="clean_accuracy",
            marker="o",
            errorbar="sd",
            ax=axis,
        )
        axis.set_title(f"Phase 2 {family}: clean accuracy")
        axis.set_xlabel("Within-family ordinal bottleneck strength (0 = weakest)")
        figure.tight_layout()
        path = output_dir / f"phase2_{family}_clean_accuracy.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    autoencoder = clean[clean.family == "autoencoder"]
    for metric, label in (
        ("reconstruction_mse", "Reconstruction MSE"),
        ("reconstruction_psnr_db", "Reconstruction PSNR (dB)"),
    ):
        if metric not in autoencoder or autoencoder[metric].dropna().empty:
            continue
        figure, axis = plt.subplots(figsize=(7, 5))
        sns.lineplot(
            data=autoencoder,
            x="bottleneck_strength_ordinal",
            y=metric,
            marker="o",
            errorbar="sd",
            ax=axis,
        )
        axis.set_title(f"Phase 2 autoencoder: {label}")
        axis.set_xlabel("Within-family ordinal bottleneck strength (0 = weakest)")
        axis.set_ylabel(label)
        figure.tight_layout()
        path = output_dir / f"phase2_autoencoder_{metric}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    geometry_columns = {
        "median_nearest_opposing_distance",
        "robust_auc",
        "attack",
        "family",
    }
    if geometry_columns.issubset(summary.columns):
        latent = summary[summary.attack == "latent_pgd"]
        if not latent.empty:
            figure, axis = plt.subplots(figsize=(7, 5))
            sns.scatterplot(
                data=latent,
                x="median_nearest_opposing_distance",
                y="robust_auc",
                hue="family",
                style="family",
                s=70,
                ax=axis,
            )
            axis.set_title("Phase 2 latent robustness versus geometry")
            axis.set_xlabel("Median nearest opposing-class distance")
            axis.set_ylabel("Latent-PGD normalized AUC")
            figure.tight_layout()
            path = output_dir / "phase2_geometry_latent_robustness.png"
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(path)
    synthesis_path = Path(summary_path).with_name("phase2_synthesis.parquet")
    if synthesis_path.exists():
        synthesis = pd.read_parquet(synthesis_path)
        synthesis_metrics = (
            (
                "semantic_separation_robustness",
                "Semantic-separation robustness",
                "phase2_local_vs_semantic_separation.png",
            ),
            (
                "collision_resistance_auc",
                "Collision-resistance AUC",
                "phase2_local_vs_collision_resistance.png",
            ),
        )
        for metric, label, filename in synthesis_metrics:
            if metric not in synthesis or synthesis[metric].dropna().empty:
                continue
            figure, axis = plt.subplots(figsize=(7, 5))
            sns.scatterplot(
                data=synthesis,
                x="local_robustness_auc",
                y=metric,
                hue="family",
                style="family",
                s=70,
                ax=axis,
            )
            axis.set_title(f"Local robustness versus {label.lower()}")
            axis.set_xlabel("Input-PGD normalized robust-accuracy AUC")
            axis.set_ylabel(label)
            figure.tight_layout()
            path = output_dir / filename
            figure.savefig(path, dpi=180)
            plt.close(figure)
            outputs.append(path)
    return outputs
