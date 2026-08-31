# RIBS experiments

This repository implements the experiments in `RIBS_RESEARCH_PLAN.md`. The
pipeline is configuration-driven and stores every run in a unique directory.

Create and synchronize the pinned project environment with uv:

```bash
uv sync --extra dev
source .venv/bin/activate
```

`uv.lock` is the authoritative resolved environment; run commands through
`uv run ...` or activate `.venv` first.

## Phase 1 quick start

Prepare Imagenette through torchvision. The command below downloads the
official 320-pixel archive into `data/imagenette` if it is not already there,
then creates the fixed experiment manifest:

```bash
PYTHONPATH=src python -m ribs.cli prepare-data \
  --root data/imagenette \
  --output data/imagenette_manifest.csv \
  --size 320px \
  --download
```

The production loader uses `torchvision.datasets.Imagenette` for both the
official train and validation splits. The manifest adds stable sample IDs,
image hashes, class indices, and the fixed development-train/tuning split.

Train one configuration without editing source code:

```bash
PYTHONPATH=src python -m ribs.cli train \
  --config configs/phase1.yaml \
  --set data.root=data/imagenette \
  --set data.manifest=data/imagenette_manifest.csv \
  --set model.dz=128 \
  --set seed=0
```

Run the complete dimensional Phase 1 matrix:

```bash
PYTHONPATH=src python -m ribs.cli train-matrix \
  --config configs/phase1.yaml \
  --dimensions 512 256 128 64 32 16 \
  --seeds 0 1 2
```

For a completed run, use the following sequence:

```bash
PYTHONPATH=src python -m ribs.cli evaluate --run-dir outputs/dimensional/<run-id>
PYTHONPATH=src python -m ribs.cli attack --run-dir outputs/dimensional/<run-id>
PYTHONPATH=src python -m ribs.cli extract-latents --run-dir outputs/dimensional/<run-id> --split development_tune
PYTHONPATH=src python -m ribs.cli extract-latents --run-dir outputs/dimensional/<run-id> --split final
PYTHONPATH=src python -m ribs.cli analyze --run-dir outputs/dimensional/<run-id> --experiment geometry
PYTHONPATH=src python -m ribs.cli analyze --run-dir outputs/dimensional/<run-id> --experiment invariance
PYTHONPATH=src python -m ribs.cli aggregate --output-root outputs
PYTHONPATH=src python -m ribs.cli render --summary outputs/phase1_summary.parquet
PYTHONPATH=src python -m ribs.cli validate-phase1 --output-root outputs
```

Phase 2 sweeps are available through the family-matrix command or the helper
script:

```bash
PYTHONPATH=src python -m ribs.cli train-family-matrix \
  --decision-record configs/phase2_decision.yaml \
  --config configs/vib.yaml --parameter beta \
  --values 0 0.0001 0.0003 0.001 0.003 0.01 --seeds 0 1 2
./scripts/run_phase2.sh configs configs/phase2_decision.yaml
```

Copy `configs/phase2_decision.example.yaml`, fill it with the frozen Phase 2
decision, and save it as a new decision record before launching the matrix.
Its contents and SHA-256 hash are embedded in every Phase 2 resolved config.

For the autoencoder family, first train a reference classifier with
`train-reference`, then use `evaluate-autoencoder` and `autoencoder-attack`
with the two run directories. Discrete models should also be checked with
`square-attack`; their latent evaluation writes separate pre-quantization and
post-bottleneck attack files. Before running final targeted-collision analyses,
set `collision.tuned=true` only after the collision loss weights have been fixed
on the tuning split. Cross-model input transfer is available through
`transfer-attack --source-run-dir ... --run-dir ...`; the command rejects
runs that do not share the same dataset manifest.

Use `--set train.epochs=2 --set train.batch_size=2 --set data.image_size=32`
for a small development run. All important outputs are machine-readable and
the `COMPLETED` marker is written only after a run finishes successfully.
Evaluations also use unique, configuration-derived attempt directories, so a
rerun never silently overwrites an earlier result.
