import torch
from torch.utils.data import DataLoader, Dataset

from ribs.training import _training_batches


class RandomAugmentationDataset(Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {"index": index, "augmentation": torch.rand(())}


def _augmentation_sequence(model_draws_per_batch: int) -> list[float]:
    loader = DataLoader(
        RandomAugmentationDataset(),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(17),
        num_workers=0,
    )
    augmentation_generator = torch.Generator().manual_seed(1000)
    torch.manual_seed(3000)
    values = []
    for batch in _training_batches(loader, augmentation_generator):
        values.extend(float(value) for value in batch["augmentation"])
        torch.rand(model_draws_per_batch)
    return values


def test_main_process_augmentation_rng_is_independent_of_model_rng_draws():
    assert _augmentation_sequence(1) == _augmentation_sequence(25)
