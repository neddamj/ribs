from pathlib import Path

import pandas as pd
from PIL import Image

from ribs.data import ImagenetteDataset, prepare_manifest


def test_manifest_and_dataset(tmp_path: Path):
    for split in ("train", "val"):
        for class_index in range(10):
            directory = tmp_path / split / f"class_{class_index}"
            directory.mkdir(parents=True)
            count = 2 if split == "val" else 10
            for image_index in range(count):
                Image.new("RGB", (32, 32), (class_index, image_index, 0)).save(
                    directory / f"{image_index}.jpg"
                )
    manifest = prepare_manifest(tmp_path, tmp_path / "manifest.csv")
    frame = pd.read_csv(manifest)
    assert set(frame.split) == {"development_train", "development_tune", "final"}
    tune_counts = frame[frame.split == "development_tune"].groupby("class_index").size()
    assert tune_counts.eq(1).all()
    sample = ImagenetteDataset(tmp_path, manifest, "final", image_size=16)[0]
    assert sample["image"].shape == (3, 16, 16)
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 1


def test_torchvision_imagenette_source_is_used(tmp_path: Path):
    wnids = [
        "n01440764",
        "n02102040",
        "n02979186",
        "n03000684",
        "n03028079",
        "n03394916",
        "n03417042",
        "n03425413",
        "n03445777",
        "n03888257",
    ]
    for split in ("train", "val"):
        for class_name in wnids:
            directory = tmp_path / "imagenette2-320" / split / class_name
            directory.mkdir(parents=True)
            count = 10 if split == "train" else 1
            for image_index in range(count):
                Image.new("RGB", (32, 32), (image_index, 0, 0)).save(
                    directory / f"{image_index}.jpeg"
                )
    manifest = prepare_manifest(tmp_path, tmp_path / "manifest.csv")
    frame = pd.read_csv(manifest)
    assert frame.source_archive_md5.iloc[0] == "3df6f0d01a2c9592104656642f5e78a3"
    dataset = ImagenetteDataset(tmp_path, manifest, "final", image_size=16)
    assert dataset.source is not None
    assert dataset[0]["image"].shape == (3, 16, 16)
