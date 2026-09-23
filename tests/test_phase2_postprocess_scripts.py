import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_postprocess_script_continues_after_independent_stage_failure(tmp_path):
    output_root = tmp_path / "outputs"
    identity = output_root / "identity" / "identity-seed0"
    identity.mkdir(parents=True)
    (identity / "COMPLETED").write_text("completed\n")
    (identity / "resolved_config.yaml").write_text("seed: 0\nmodel:\n  family: identity\n")
    first = output_root / "vib" / "run-first"
    second = output_root / "vib" / "run-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    command_log = tmp_path / "commands.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-c" && "$2" == import\\ csv* ]]; then
  exec "${REAL_PYTHON}" "$@"
fi
if [[ "$1" == "-c" ]]; then
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-run-dirs" ]]; then
  printf 'vib\\t%s\\nvib\\t%s\\n' "${FIRST_RUN}" "${SECOND_RUN}"
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-runtime-policy" ]]; then
  printf '1\t0\t0\n'
  exit 0
fi
printf '%s\\n' "$*" >>"${COMMAND_LOG}"
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "attack" && "$*" == *"${FIRST_RUN}"* ]]; then
  exit 7
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "tune-collision" ]]; then
  printf '%s\\n' "${FIRST_RUN}/selection.json"
fi
exit 0
""")
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "REAL_PYTHON": sys.executable,
        "OUTPUT_ROOT": str(output_root),
        "PHASE2_FAMILIES": "vib",
        "PHASE2_FINALIZE": "false",
        "PHASE2_RUN_TAG": "test",
        "FIRST_RUN": str(first),
        "SECOND_RUN": str(second),
        "COMMAND_LOG": str(command_log),
    }
    result = subprocess.run(
        ["bash", "scripts/run_phase2_postprocess.sh", "configs", "decision.yaml"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    commands = command_log.read_text()
    assert f"attack --run-dir {first}" in commands
    assert f"attack --run-dir {second}" in commands
    report = json.loads((output_root / "phase2_postprocess_test.json").read_text())
    assert any(
        row["run_dir"] == str(first) and row["stage"] == "robustness" and row["status"] == "failed"
        for row in report["runs"]
    )
    assert any(row["run_dir"] == str(second) for row in report["runs"])


def test_phase2_shell_scripts_parse():
    scripts = [
        PROJECT_ROOT / "scripts" / "run_phase2_postprocess.sh",
        PROJECT_ROOT / "scripts" / "run_phase2_postprocess_parallel.sh",
        PROJECT_ROOT / "scripts" / "run_phase3_collisions.sh",
        PROJECT_ROOT / "scripts" / "run_phase3_collisions_parallel.sh",
    ]
    result = subprocess.run(
        ["bash", "-n", *(str(path) for path in scripts)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_postprocess_retries_missing_diagnostics_without_repeating_robustness(tmp_path):
    output_root = tmp_path / "outputs"
    identity = output_root / "identity" / "identity-seed0"
    identity.mkdir(parents=True)
    (identity / "COMPLETED").write_text("completed\n")
    (identity / "resolved_config.yaml").write_text("seed: 0\nmodel:\n  family: identity\n")
    run_dir = output_root / "vq" / "run"
    robustness = run_dir / "evaluations" / "robustness-existing-attempt0"
    robustness.mkdir(parents=True)
    (robustness / "COMPLETED").write_text("completed\n")
    (robustness / "config.json").write_text(
        json.dumps(
            {
                "split": "final",
                "max_samples": None,
                "attack_protocol_version": 2,
            }
        )
    )

    command_log = tmp_path / "commands.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-c" && "$2" == import\\ csv* ]]; then
  exec "${REAL_PYTHON}" "$@"
fi
if [[ "$1" == "-c" ]]; then
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-run-dirs" ]]; then
  printf 'vq\\t%s\\n' "${RUN_DIR}"
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-runtime-policy" ]]; then
  printf '1\t0\t0\n'
  exit 0
fi
printf '%s\\n' "$*" >>"${COMMAND_LOG}"
exit 0
""")
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "REAL_PYTHON": sys.executable,
        "OUTPUT_ROOT": str(output_root),
        "PHASE2_FAMILIES": "vq",
        "PHASE2_FINALIZE": "false",
        "PHASE2_RUN_TAG": "diagnostics-test",
        "RUN_DIR": str(run_dir),
        "COMMAND_LOG": str(command_log),
    }

    result = subprocess.run(
        ["bash", "scripts/run_phase2_postprocess.sh", "configs", "decision.yaml"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text()
    assert f"attack-diagnostics --run-dir {run_dir}" in commands
    assert f"attack --run-dir {run_dir}" not in commands


def test_parallel_postprocess_assigns_each_run_once(tmp_path):
    output_root = tmp_path / "outputs"
    identity = output_root / "identity" / "identity-seed0"
    identity.mkdir(parents=True)
    (identity / "COMPLETED").write_text("completed\n")
    (identity / "resolved_config.yaml").write_text("seed: 0\nmodel:\n  family: identity\n")
    first = output_root / "vib" / "run-first"
    second = output_root / "vib" / "run-second"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    fake_python = tmp_path / "fake-python"
    fake_python.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-c" && "$2" == import\\ csv* ]]; then
  exec "${REAL_PYTHON}" "$@"
fi
if [[ "$1" == "-c" ]]; then
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-run-dirs" ]]; then
  printf 'vib\\t%s\\nvib\\t%s\\n' "${FIRST_RUN}" "${SECOND_RUN}"
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-runtime-policy" ]]; then
  printf '1\t0\t0\n'
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "tune-collision" ]]; then
  printf '%s\\n' "${FIRST_RUN}/selection.json"
fi
exit 0
""")
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "REAL_PYTHON": sys.executable,
        "OUTPUT_ROOT": str(output_root),
        "PHASE2_RUN_TAG": "parallel-test",
        "GPU_COUNT": "2",
        "FIRST_RUN": str(first),
        "SECOND_RUN": str(second),
    }
    result = subprocess.run(
        [
            "bash",
            "scripts/run_phase2_postprocess_parallel.sh",
            "configs",
            "decision.yaml",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    logs = list((output_root / "phase2_postprocess_logs" / "parallel-test").glob("*.log"))
    assert len(logs) == 2
    assert len({path.name for path in logs}) == 2
    assert not list((output_root / ".phase2_postprocess_claims").iterdir())


def test_postprocess_does_not_repeat_a_completed_failed_primary_diagnostic(tmp_path):
    output_root = tmp_path / "outputs"
    identity = output_root / "identity" / "identity-seed0"
    identity.mkdir(parents=True)
    (identity / "COMPLETED").write_text("completed\n")
    (identity / "resolved_config.yaml").write_text("seed: 0\nmodel:\n  family: identity\n")
    run_dir = output_root / "vq" / "run"
    robustness = run_dir / "evaluations" / "robustness-existing-attempt0"
    diagnostic = run_dir / "evaluations" / "attack-diagnostics-existing-attempt0"
    robustness.mkdir(parents=True)
    diagnostic.mkdir(parents=True)
    final_config = {"split": "final", "max_samples": None, "attack_protocol_version": 2}
    (robustness / "config.json").write_text(json.dumps(final_config))
    (robustness / "COMPLETED").write_text("completed\n")
    (diagnostic / "diagnostics.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "attack_protocol_version": 2,
                "independent_audit": False,
            }
        )
    )
    (diagnostic / "COMPLETED").write_text("completed\n")
    command_log = tmp_path / "commands.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-c" && "$2" == import\\ torch* ]]; then
  exit 0
fi
if [[ "$1" == "-c" ]]; then
  exec "${REAL_PYTHON}" "$@"
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-run-dirs" ]]; then
  printf 'vq\\t%s\\n' "${RUN_DIR}"
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-runtime-policy" ]]; then
  printf '1\\t0\\t0\\n'
  exit 0
fi
printf '%s\\n' "$*" >>"${COMMAND_LOG}"
exit 0
""")
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "REAL_PYTHON": sys.executable,
        "OUTPUT_ROOT": str(output_root),
        "PHASE2_FAMILIES": "vq",
        "PHASE2_FINALIZE": "false",
        "PHASE2_RUN_TAG": "failed-diagnostic-test",
        "RUN_DIR": str(run_dir),
        "COMMAND_LOG": str(command_log),
    }

    result = subprocess.run(
        ["bash", "scripts/run_phase2_postprocess.sh", "configs", "decision.yaml"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    commands = command_log.read_text()
    assert f"attack-diagnostics --run-dir {run_dir}" not in commands
    assert "existing_primary_diagnostic_failed" in result.stderr


def test_postprocess_does_not_treat_partial_masking_as_final(tmp_path):
    output_root = tmp_path / "outputs"
    identity = output_root / "identity" / "identity-seed0"
    identity.mkdir(parents=True)
    (identity / "COMPLETED").write_text("completed\n")
    (identity / "resolved_config.yaml").write_text("seed: 0\nmodel:\n  family: identity\n")
    run_dir = output_root / "vq" / "run"
    robustness = run_dir / "evaluations" / "robustness-existing-attempt0"
    diagnostic = run_dir / "evaluations" / "attack-diagnostics-existing-attempt0"
    square = run_dir / "evaluations" / "square-partial-attempt0"
    transfer = run_dir / "evaluations" / "transfer-final-attempt0"
    for path in (robustness, diagnostic, square, transfer):
        path.mkdir(parents=True)
        (path / "COMPLETED").write_text("completed\n")
    (robustness / "config.json").write_text(
        json.dumps({"split": "final", "max_samples": None, "attack_protocol_version": 2})
    )
    (diagnostic / "diagnostics.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "attack_protocol_version": 2,
                "independent_audit": False,
            }
        )
    )
    (square / "config.json").write_text(
        json.dumps({"split": "final", "max_samples": 2, "attack_protocol_version": 2})
    )
    (transfer / "config.json").write_text(
        json.dumps({"split": "final", "max_samples": 1000, "attack_protocol_version": 2})
    )
    command_log = tmp_path / "commands.log"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("""#!/usr/bin/env bash
if [[ "$1" == "-c" && "$2" == import\\ torch* ]]; then
  exit 0
fi
if [[ "$1" == "-c" ]]; then
  exec "${REAL_PYTHON}" "$@"
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-run-dirs" ]]; then
  printf 'vq\\t%s\\n' "${RUN_DIR}"
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "ribs.cli" && "$3" == "phase2-runtime-policy" ]]; then
  printf '1\\t1\\t0\\n'
  exit 0
fi
printf '%s\\n' "$*" >>"${COMMAND_LOG}"
exit 0
""")
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "REAL_PYTHON": sys.executable,
        "OUTPUT_ROOT": str(output_root),
        "PHASE2_FAMILIES": "vq",
        "PHASE2_FINALIZE": "false",
        "PHASE2_RUN_TAG": "masking-test",
        "RUN_DIR": str(run_dir),
        "COMMAND_LOG": str(command_log),
    }

    result = subprocess.run(
        ["bash", "scripts/run_phase2_postprocess.sh", "configs", "decision.yaml"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text()
    assert f"square-attack --run-dir {run_dir}" in commands
    assert f"transfer-attack --run-dir {run_dir}" not in commands


def test_parallel_training_drains_final_partial_worker_batch(tmp_path):
    output_root = tmp_path / "outputs"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_python.chmod(0o755)
    environment = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "OUTPUT_ROOT": str(output_root),
        "GPU_COUNT": "4",
        "PHASE2_RUN_TAG": "drain-test",
    }

    result = subprocess.run(
        [
            "bash",
            "scripts/run_phase2_parallel.sh",
            "configs",
            "configs/phase2_decision_20260902T120000.yaml",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
