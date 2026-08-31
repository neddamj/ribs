"""Metrics for robustness curves and censored attack radii."""

from __future__ import annotations

import numpy as np
import pandas as pd


def conditional_asr(frame: pd.DataFrame, success_column: str = "successful") -> float:
    clean = frame["clean_correct"].astype(bool)
    return (
        float((frame.loc[clean, success_column].astype(bool)).mean())
        if clean.any()
        else float("nan")
    )


def shared_clean_correct(records: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the sample intersection clean-correct for every compared model."""
    if not records:
        return pd.DataFrame()
    merged = None
    for model_name, frame in records.items():
        current = (
            frame[["sample_id", "clean_correct"]]
            .drop_duplicates("sample_id")
            .rename(columns={"clean_correct": model_name})
        )
        merged = current if merged is None else merged.merge(current, on="sample_id", how="inner")
    clean_columns = [name for name in records]
    merged["shared_clean_correct"] = merged[clean_columns].all(axis=1)
    return merged[merged.shared_clean_correct].reset_index(drop=True)


def trapezoid_auc(
    radii: list[float] | np.ndarray,
    robust_accuracy: list[float] | np.ndarray,
    maximum_radius: float | None = None,
) -> float:
    radii = np.asarray(radii, dtype=float)
    accuracy = np.asarray(robust_accuracy, dtype=float)
    if radii.size < 2:
        return float(accuracy[0]) if accuracy.size else float("nan")
    maximum = float(radii[-1] if maximum_radius is None else maximum_radius)
    if maximum <= 0 or radii[0] < 0 or radii[-1] > maximum + 1e-12:
        raise ValueError("AUC radii must lie in [0, maximum_radius]")
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(accuracy, radii) / maximum)


def _kaplan_meier_median_status(
    success_radii: list[float], censored_count: int, maximum: float
) -> tuple[float, bool]:
    times = (
        sorted((float(value), True) for value in success_radii)
        + [(float(maximum), False)] * censored_count
    )
    at_risk = len(times)
    survival = 1.0
    for time in sorted({t for t, _ in times}):
        events = sum(event and t == time for t, event in times)
        censored = sum((not event) and t == time for t, event in times)
        if at_risk and events:
            survival *= 1.0 - events / at_risk
            if survival <= 0.5:
                return time, True
        at_risk -= events + censored
    # A numeric sentinel keeps aggregate tables rectangular. The companion
    # field in summarize_curve records whether this value is censored.
    return maximum, False


def kaplan_meier_median(success_radii: list[float], censored_count: int, maximum: float) -> float:
    return _kaplan_meier_median_status(success_radii, censored_count, maximum)[0]


def summarize_curve(
    records: pd.DataFrame, radius_column: str = "radius"
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    records = records.sort_values(["sample_id", radius_column]).copy()
    records["nested_success"] = records.groupby("sample_id")["successful"].cummax()
    for radius, group in records.groupby(radius_column, sort=True):
        rows.append(
            {
                "radius": float(radius),
                "robust_accuracy": float(group["nested_success"].eq(False).mean()),
                "conditional_asr": conditional_asr(group, "nested_success"),
                "num_samples": len(group),
            }
        )
    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve, {}
    clean_accuracy = float(records.groupby("sample_id").clean_correct.first().mean())
    if float(curve.radius.iloc[0]) > 0.0:
        curve = pd.concat(
            [
                pd.DataFrame(
                    [
                        {
                            "radius": 0.0,
                            "robust_accuracy": clean_accuracy,
                            "conditional_asr": 0.0,
                            "num_samples": records.sample_id.nunique(),
                        }
                    ]
                ),
                curve,
            ],
            ignore_index=True,
        )
    curve["robust_accuracy_nested"] = curve["robust_accuracy"]
    radii = curve.radius.to_numpy()
    accuracies = curve.robust_accuracy_nested.to_numpy()
    clean_correct = records[records.clean_correct.astype(bool)]
    first_success = (
        clean_correct[clean_correct.successful].groupby("sample_id")[radius_column].min()
    )
    max_radius = float(radii[-1])
    eligible_count = clean_correct.sample_id.nunique()
    censored = eligible_count - len(first_success)
    median, median_reached = _kaplan_meier_median_status(
        first_success.tolist(), censored, max_radius
    )
    return curve, {
        "robust_auc": trapezoid_auc(radii, accuracies),
        "median_min_success_radius": median,
        "median_min_success_right_censored": not median_reached,
        "clean_accuracy": clean_accuracy,
    }
