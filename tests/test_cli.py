from pathlib import Path

import pytest
import yaml

from ribs.cli import build_parser


def test_cli_parses_training_override():
    args = build_parser().parse_args(
        ["train", "--config", "configs/phase1.yaml", "--set", "model.dz=16"]
    )
    assert args.command == "train"
    assert args.set == ["model.dz=16"]


def test_phase2_matrix_requires_and_records_decision_argument():
    args = build_parser().parse_args(
        [
            "train-family-matrix",
            "--config",
            "configs/vib.yaml",
            "--parameter",
            "beta",
            "--values",
            "0.001",
            "--decision-record",
            "configs/phase2_decision.example.yaml",
        ]
    )
    assert args.decision_record.endswith("phase2_decision.example.yaml")


def test_phase2_matrix_accepts_explicit_single_run_resume():
    args = build_parser().parse_args(
        [
            "train-family-matrix",
            "--config",
            "configs/quantized.yaml",
            "--parameter",
            "bits",
            "--values",
            "6",
            "--seeds",
            "0",
            "--resume",
            "outputs/quantized/partial/checkpoints/last.pt",
            "--decision-record",
            "configs/phase2_decision.example.yaml",
        ]
    )
    assert args.resume.endswith("checkpoints/last.pt")


def test_attack_diagnostics_accepts_independent_audit_overrides():
    args = build_parser().parse_args(
        [
            "attack-diagnostics",
            "--run-dir",
            "outputs/vq/example",
            "--diagnostic-samples",
            "128",
            "--diagnostic-tolerance",
            "0.02",
        ]
    )
    assert args.command == "attack-diagnostics"
    assert args.diagnostic_samples == 128
    assert args.diagnostic_tolerance == 0.02


def test_attack_audit_amendment_is_preregistered_and_strict():
    from ribs.phase2 import load_attack_audit_amendment

    amendment = load_attack_audit_amendment("configs/phase2_attack_audit_amendment_20260921.yaml")
    assert amendment["amendment_id"] == "phase2-attack-audit-v1-20260921"
    assert amendment["protocol"]["sample_count"] == 128
    assert amendment["protocol"]["tolerance"] == 0.02



def test_phase2_matrix_has_fixed_strengths():
    import pandas as pd

    from ribs.phase2 import PHASE2_SPECS, ordinal_strength, phase2_strength_columns

    assert len(PHASE2_SPECS["vib"][1]) == 6
    assert ordinal_strength("vq", 512) == 0
    assert ordinal_strength("vq", 16) == 5
    assert ordinal_strength("quantized", "FP32") == 0
    mixed = phase2_strength_columns(
        pd.DataFrame(
            [
                {"family": "quantized", "bits": "FP32", "dz": 128},
                {"family": "quantized", "bits": 8, "dz": 128},
                {"family": "vq", "codebook_size": 512},
            ]
        )
    )
    assert mixed.bits.dropna().tolist() == ["FP32", "8"]
    assert mixed.strength_value.tolist() == ["FP32", "8", "512"]


def test_runtime_amendment_selects_balanced_reduced_matrix():
    from ribs.phase2 import load_runtime_amendment, runtime_policy_for_config

    amendment = load_runtime_amendment("configs/phase2_runtime_amendment_20260917.yaml")
    selected = runtime_policy_for_config(
        {"seed": 2, "model": {"family": "vq", "codebook_size": 128}}, amendment
    )
    omitted = runtime_policy_for_config(
        {"seed": 2, "model": {"family": "vq", "codebook_size": 256}}, amendment
    )
    masking = runtime_policy_for_config(
        {"seed": 0, "model": {"family": "quantized", "bits": 3}}, amendment
    )
    autoencoder = runtime_policy_for_config(
        {"seed": 0, "model": {"family": "autoencoder", "dz": 512}}, amendment
    )

    assert selected["phase2_robustness"]
    assert selected["phase3_collision"]
    assert not omitted["phase2_robustness"]
    assert masking["phase2_masking_checks"]
    assert not autoencoder["phase2_robustness"]
    assert not autoencoder["phase3_collision"]


def test_frozen_decision_record_schema_is_valid():
    from ribs.phase2 import validate_decision_record

    record = validate_decision_record("configs/phase2_decision_20260902T120000.yaml")
    assert record["seeds"]["training"] == [0, 1, 2]


def test_decision_record_rejects_strength_drift(tmp_path):
    from ribs.phase2 import validate_decision_record

    record = yaml.safe_load(Path("configs/phase2_decision_20260902T120000.yaml").read_text())
    record["included_model_families_and_strengths"]["vib"]["values"][-1] = 0.02
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(record))
    with pytest.raises(ValueError, match="strengths differ"):
        validate_decision_record(path)


def test_identity_followup_reuses_valid_completed_control(tmp_path):
    from ribs.phase2 import valid_identity_run

    run_dir = tmp_path / "identity" / "historical-seed0"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "history.parquet").write_bytes(b"history")
    (run_dir / "metrics.json").write_bytes(b"{}")
    (run_dir / "checkpoints" / "best_tune_accuracy.pt").write_bytes(b"checkpoint")
    (run_dir / "COMPLETED").write_text("completed\n")
    (run_dir / "data_manifest_hash.txt").write_text("placeholder\n")
    # An unavailable manifest hash is still allowed for the helper's structural
    # test when the current config points to a missing manifest.
    config = {"data": {"manifest": str(tmp_path / "missing.csv")}}
    (run_dir / "resolved_config.yaml").write_text(
        "model:\n  family: identity\n  dz: 512\nseed: 0\n"
    )
    assert valid_identity_run(tmp_path, 0, config) == run_dir
