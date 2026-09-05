from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from ribs import evaluation, training
from ribs.config import load_config


class TinyDataset(Dataset):
    def __init__(self):
        self.images = torch.tensor(
            [
                [[[0.9, 0.8], [0.7, 0.6]]],
                [[[0.1, 0.2], [0.3, 0.4]]],
                [[[0.8, 0.7], [0.6, 0.5]]],
                [[[0.2, 0.3], [0.4, 0.5]]],
            ]
        )
        self.labels = torch.tensor([0, 1, 0, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "sample_id": f"sample-{index}",
        }


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 2)
        self.classifier = torch.nn.Linear(2, 2)

    def forward(self, images, sample=False):
        latent = self.encoder(images.flatten(1))
        return SimpleNamespace(
            logits=self.classifier(latent),
            latent=latent,
            canonical_latent=latent,
            pre_bottleneck=latent,
            aux_losses={},
            metadata={},
        )

    def classify_latent(self, latent):
        return self.classifier(latent)


def test_lightweight_train_resume_evaluate_attack_and_extract_smoke(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("smoke fixture\n")
    config = load_config("configs/phase1.yaml")
    config.update({"seed": 0, "device": "cpu", "output_dir": str(tmp_path / "outputs")})
    config["data"].update({"manifest": str(manifest), "num_workers": 0})
    config["model"].update({"family": "dimensional", "dz": 2})
    config["train"].update(
        {
            "epochs": 1,
            "batch_size": 2,
            "effective_batch_size": 2,
            "mixed_precision": "off",
        }
    )
    config["attack"].update(
        {
            "evaluation_batch_size": 2,
            "input_epsilons": [0.0, 0.1],
            "latent_rhos": [0.1],
            "steps": 1,
            "restarts": 1,
            "run_diagnostics": False,
        }
    )

    def loader(*args, **kwargs):
        return DataLoader(
            TinyDataset(),
            batch_size=2,
            shuffle=bool(args[2]),
            generator=torch.Generator().manual_seed(17),
        )

    monkeypatch.setattr(training, "create_model", lambda _: TinyModel())
    monkeypatch.setattr(training, "_loader", loader)
    first_run = training.train_model(config)
    resumed_config = {**config, "resume": str(first_run / "checkpoints" / "last.pt")}
    resumed_run = training.train_model(resumed_config)

    def load_tiny(run_dir, checkpoint=None):
        checkpoint = checkpoint or "best_tune_accuracy.pt"
        state = torch.load(
            resumed_run / "checkpoints" / checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model = TinyModel()
        model.load_state_dict(state["model"])
        return model.eval(), state["config"], torch.device("cpu")

    batches = list(DataLoader(TinyDataset(), batch_size=2, shuffle=False))
    monkeypatch.setattr(evaluation, "load_model", load_tiny)
    monkeypatch.setattr(evaluation, "make_loader", lambda *args, **kwargs: batches)
    clean_path = evaluation.evaluate_clean_run(resumed_run, max_samples=2)
    attack_paths = evaluation.evaluate_attacks(resumed_run, max_samples=2)
    latent_path = evaluation.extract_latents(resumed_run)

    assert (resumed_run / "COMPLETED").is_file()
    assert clean_path.is_file()
    assert attack_paths["input"].is_file()
    assert attack_paths["latent"].is_file()
    assert latent_path.is_file()
