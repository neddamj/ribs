"""Imagenette manifests, fixed splits, and raw-pixel preprocessing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _find_split(root: Path, split: str) -> Path:
    candidates = [root / split, root / "imagenette2-320" / split]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find Imagenette {split!r} directory below {root}")


def _torchvision_root(root: Path, size: str) -> Path:
    archive_stem = {"320px": "imagenette2-320", "160px": "imagenette2-160", "full": "imagenette2"}[
        size
    ]
    if (root / archive_stem).is_dir():
        return root
    if root.name == archive_stem and root.is_dir():
        return root.parent
    return root


def _torchvision_samples(
    root: Path, split: str, size: str, download: bool
) -> tuple[list[tuple[Path, int]], str]:
    """Use torchvision's official dataset/index and return file paths for manifests."""
    tv_root = _torchvision_root(root, size)
    dataset = datasets.Imagenette(tv_root, split=split, size=size, download=download)
    samples = [(Path(path), int(label)) for path, label in dataset._samples]
    return samples, str(getattr(dataset, "_md5", "unknown"))


def _filesystem_samples(root: Path, split: str) -> list[tuple[Path, int]]:
    split_root = _find_split(root, split)
    class_names = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    samples = []
    for class_name in class_names:
        for image_path in sorted((split_root / class_name).rglob("*")):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append((image_path, class_to_index[class_name]))
    return samples


def prepare_manifest(
    root: str | Path,
    output: str | Path,
    split_seed: int = 2025,
    size: str = "320px",
    download: bool = False,
) -> Path:
    root = Path(root)
    source_root = _torchvision_root(root, size)
    rows: list[dict[str, object]] = []
    standard_root = (
        source_root
        / {
            "320px": "imagenette2-320",
            "160px": "imagenette2-160",
            "full": "imagenette2",
        }[size]
    )
    use_torchvision = standard_root.is_dir() or download
    archive_md5 = "filesystem"
    entries_by_split: dict[str, list[tuple[Path, int]]] = {}
    for original_split in ("train", "val"):
        if use_torchvision:
            entries_by_split[original_split], archive_md5 = _torchvision_samples(
                root, original_split, size, download
            )
        else:
            entries_by_split[original_split] = _filesystem_samples(root, original_split)
    class_to_index = {}
    for path, label in entries_by_split["train"]:
        class_to_index[path.parent.name] = label
    for original_split in ("train", "val"):
        for image_path, class_index in entries_by_split[original_split]:
            relative = image_path.relative_to(source_root if use_torchvision else root).as_posix()
            class_name = image_path.parent.name
            rows.append(
                {
                    "sample_id": relative,
                    "relative_path": relative,
                    "class_name": class_name,
                    "class_index": class_index,
                    "original_split": original_split,
                    "sha256": sha256_file(image_path),
                    "source_archive_md5": archive_md5,
                    "split": "final" if original_split == "val" else "development",
                }
            )
    frame = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    source_checksum = hashlib.sha256(
        "\n".join(
            f"{sample_id}\t{digest}"
            for sample_id, digest in zip(frame["sample_id"], frame["sha256"])
        ).encode("utf-8")
    ).hexdigest()
    frame["source_checksum"] = source_checksum
    train_mask = frame.original_split == "train"
    train_frame = frame[train_mask]
    train_ids, tune_ids = train_test_split(
        train_frame,
        test_size=0.1,
        stratify=train_frame["class_index"],
        random_state=split_seed,
    )
    frame.loc[frame.sample_id.isin(train_ids.sample_id), "split"] = "development_train"
    frame.loc[frame.sample_id.isin(tune_ids.sample_id), "split"] = "development_tune"
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    (output.with_suffix(output.suffix + ".classes.json")).write_text(
        json.dumps(class_to_index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def manifest_hash(path: str | Path) -> str:
    return sha256_file(Path(path))


def build_transform(split: str, image_size: int = 224) -> Callable:
    if split == "development_train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


class ImagenetteDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        split: str,
        image_size: int = 224,
        size: str = "320px",
    ):
        self.root = _torchvision_root(Path(root), size)
        self.frame = pd.read_csv(manifest)
        self.frame = self.frame[self.frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Manifest has no samples for split={split}")
        self.transform = build_transform(split, image_size)
        self.source = None
        self.source_indices: list[int] | None = None
        standard_root = (
            _torchvision_root(self.root, size)
            / {
                "320px": "imagenette2-320",
                "160px": "imagenette2-160",
                "full": "imagenette2",
            }[size]
        )
        if standard_root.is_dir():
            original_split = "val" if split == "final" else "train"
            self.source = datasets.Imagenette(
                _torchvision_root(self.root, size),
                split=original_split,
                size=size,
                download=False,
                transform=self.transform,
            )
            source_indices = {
                Path(path).relative_to(self.root).as_posix(): index
                for index, (path, _) in enumerate(self.source._samples)
            }
            try:
                self.source_indices = [
                    source_indices[sample_id] for sample_id in self.frame.sample_id
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Manifest sample is missing from torchvision Imagenette: {exc}"
                ) from exc

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        if self.source is not None and self.source_indices is not None:
            tensor, target = self.source[self.source_indices[index]]
            if int(target) != int(row.class_index):
                raise ValueError(f"Manifest label mismatch for sample_id={row.sample_id}")
        else:
            with Image.open(self.root / row.relative_path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)
        return {
            "image": tensor,
            "label": int(row.class_index),
            "sample_id": row.sample_id,
        }
