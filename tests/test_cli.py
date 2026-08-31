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
