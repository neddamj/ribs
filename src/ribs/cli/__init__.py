"""Command-line interface for preparing, running, and analyzing experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

from ..analysis import (
    aggregate_completed_runs,
    analyze_geometry,
    analyze_invariance,
    summarize_attack_file,
    validate_phase1_acceptance,
)
from ..config import apply_overrides, load_config
from ..data import prepare_manifest
from ..evaluation import (
    evaluate_attacks,
    evaluate_autoencoder,
    evaluate_autoencoder_attacks,
    evaluate_clean_run,
    evaluate_collision_attacks,
    evaluate_square_attack,
    evaluate_transfer_attack,
    extract_latents,
)
from ..training import train_model


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return apply_overrides(load_config(args.config), args.set or [])


def _run_dir(args: argparse.Namespace) -> Path:
    path = Path(args.run_dir)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ribs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split-seed", type=int, default=2025)
    prepare.add_argument("--size", choices=["320px", "160px", "full"], default="320px")
    prepare.add_argument("--download", action="store_true")

    train = subparsers.add_parser("train")
    train.add_argument("--config", default="configs/phase1.yaml")
    train.add_argument("--set", action="append", default=[])
    train.add_argument("--resume")

    reference = subparsers.add_parser("train-reference")
    reference.add_argument("--config", default="configs/phase1.yaml")
    reference.add_argument("--set", action="append", default=[])

    matrix = subparsers.add_parser("train-matrix")
    matrix.add_argument("--config", default="configs/phase1.yaml")
    matrix.add_argument("--dimensions", nargs="+", type=int, default=[512, 256, 128, 64, 32, 16])
    matrix.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    matrix.add_argument("--set", action="append", default=[])
    matrix.add_argument("--skip-identity", action="store_true")

    family_matrix = subparsers.add_parser("train-family-matrix")
    family_matrix.add_argument("--config", required=True)
    family_matrix.add_argument(
        "--parameter", required=True, choices=["dz", "beta", "codebook_size", "bits"]
    )
    family_matrix.add_argument("--values", nargs="+", required=True)
    family_matrix.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    family_matrix.add_argument("--set", action="append", default=[])
    family_matrix.add_argument("--decision-record", required=True)

    for command in ("evaluate", "attack", "extract-latents", "analyze"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--run-dir", required=True)
        command_parser.add_argument("--split", default="final")
        command_parser.add_argument("--max-samples", type=int)
        if command == "analyze":
            command_parser.add_argument(
                "--experiment", choices=["geometry", "invariance", "attacks"], default="geometry"
            )
            command_parser.add_argument("--reference-run-dir")

    auto = subparsers.add_parser("evaluate-autoencoder")
    auto.add_argument("--run-dir", required=True)
    auto.add_argument("--reference-run-dir", required=True)
    auto.add_argument("--split", default="final")

    auto_attack = subparsers.add_parser("autoencoder-attack")
    auto_attack.add_argument("--run-dir", required=True)
    auto_attack.add_argument("--reference-run-dir", required=True)
    auto_attack.add_argument("--split", default="final")
    auto_attack.add_argument("--max-samples", type=int)

    collision = subparsers.add_parser("collision-attack")
    collision.add_argument("--run-dir", required=True)
    collision.add_argument("--reference-run-dir", required=True)
    collision.add_argument("--split", default="final")
    collision.add_argument("--max-pairs", type=int, default=1000)
    collision.add_argument("--lambda-sem", type=float)

    square = subparsers.add_parser("square-attack")
    square.add_argument("--run-dir", required=True)
    square.add_argument("--split", default="final")
    square.add_argument("--max-samples", type=int, default=1000)

    transfer = subparsers.add_parser("transfer-attack")
    transfer.add_argument("--run-dir", required=True)
    transfer.add_argument("--source-run-dir", required=True)
    transfer.add_argument("--split", default="final")
    transfer.add_argument("--max-samples", type=int, default=1000)

    render = subparsers.add_parser("render")
    render.add_argument("--summary", default="outputs/phase1_summary.parquet")
    render.add_argument("--output-dir", default="outputs/figures")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--output-root", default="outputs")
    validate = subparsers.add_parser("validate-phase1")
    validate.add_argument("--output-root", default="outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        path = prepare_manifest(args.root, args.output, args.split_seed, args.size, args.download)
        print(path)
        return 0
    if args.command == "train":
        config = _config(args)
        if args.resume:
            config["resume"] = args.resume
        path = train_model(config)
        print(path)
        return 0
    if args.command == "train-reference":
        config = _config(args)
        config["model"] = {**config["model"], "family": "identity", "dz": 512}
        print(train_model(config))
        return 0
    if args.command == "train-matrix":
        base = _config(args)
        paths = []
        for seed in args.seeds:
            for dz in args.dimensions:
                config = copy.deepcopy(base)
                config["seed"] = seed
                config["model"] = {**config["model"], "family": "dimensional", "dz": dz}
                paths.append(str(train_model(config)))
        if not args.skip_identity:
            config = copy.deepcopy(base)
            config["seed"] = 0
            config["model"] = {**config["model"], "family": "identity", "dz": 512}
            paths.append(str(train_model(config)))
        print("\n".join(paths))
        return 0
    if args.command == "train-family-matrix":
        import yaml

        base = _config(args)
        decision_path = Path(args.decision_record)
        decision = yaml.safe_load(decision_path.read_text(encoding="utf-8"))
        required = {
            "included_model_families_and_strengths",
            "family_specific_training_changes",
            "primary_phase2_outcomes",
            "final_attack_budgets",
            "implementation_deviations",
        }
        if not isinstance(decision, dict) or not required.issubset(decision):
            missing = sorted(required - set(decision or {}))
            raise ValueError(f"Incomplete Phase 2 decision record; missing: {missing}")
        nonempty = required - {"implementation_deviations"}
        empty = sorted(key for key in nonempty if not decision[key])
        if empty:
            raise ValueError(f"Phase 2 decision record has unfilled fields: {empty}")
        base["phase2_decision_record"] = {
            "path": str(decision_path),
            "sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            "contents": decision,
        }
        paths = []
        for seed in args.seeds:
            for value in args.values:
                config = copy.deepcopy(base)
                config["seed"] = seed
                config["model"] = {
                    **config["model"],
                    args.parameter: yaml.safe_load(value),
                }
                paths.append(str(train_model(config)))
        print("\n".join(paths))
        return 0
    if args.command in {"render", "aggregate", "validate-phase1"}:
        if args.command == "render":
            from ..plotting import render_phase1

            print("\n".join(str(path) for path in render_phase1(args.summary, args.output_dir)))
        elif args.command == "aggregate":
            print(aggregate_completed_runs(args.output_root).to_string(index=False))
        else:
            report = validate_phase1_acceptance(args.output_root)
            print(report)
            if report["status"] != "ready":
                return 2
        return 0

    run_dir = _run_dir(args)
    if args.command == "evaluate-autoencoder":
        print(evaluate_autoencoder(run_dir, args.reference_run_dir, args.split))
        return 0
    if args.command == "autoencoder-attack":
        paths = evaluate_autoencoder_attacks(
            args.run_dir, args.reference_run_dir, args.split, args.max_samples
        )
        print("\n".join(str(path) for path in paths.values()))
        return 0
    if args.command == "collision-attack":
        print(
            evaluate_collision_attacks(
                run_dir, args.reference_run_dir, args.split, args.max_pairs, args.lambda_sem
            )
        )
        return 0
    if args.command == "square-attack":
        print(evaluate_square_attack(run_dir, args.split, args.max_samples))
        return 0
    if args.command == "transfer-attack":
        print(evaluate_transfer_attack(run_dir, args.source_run_dir, args.split, args.max_samples))
        return 0
    if args.command == "evaluate":
        print(evaluate_clean_run(run_dir, args.split, max_samples=args.max_samples))
        return 0
    if args.command == "attack":
        paths = evaluate_attacks(run_dir, args.split, max_samples=args.max_samples)
        print("\n".join(str(path) for path in paths.values()))
        return 0
    if args.command == "extract-latents":
        print(extract_latents(run_dir, args.split))
        return 0
    if args.command == "analyze":
        if args.experiment == "geometry":
            print(
                analyze_geometry(
                    run_dir,
                    args.split,
                    jacobian_samples=args.max_samples or 500,
                    max_samples=args.max_samples,
                )
            )
        elif args.experiment == "invariance":
            print(analyze_invariance(run_dir, args.split, args.max_samples, args.reference_run_dir))
        else:
            evaluation_dirs = [
                path
                for path in (run_dir / "evaluations").glob("robustness-*-attempt*")
                if (path / "COMPLETED").exists()
            ]
            if not evaluation_dirs:
                evaluation_dirs = [run_dir / "evaluations"]
            for path in (
                evaluation_dir / filename
                for evaluation_dir in evaluation_dirs
                for filename in ("input_pgd.parquet", "latent_pgd.parquet")
            ):
                if path.exists():
                    print(summarize_attack_file(path))
        return 0
    return 1
