import random
from typing import Optional

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def _seeded_loader_kwargs(seed: Optional[int]) -> dict:
    if seed is None:
        return {}
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return {"worker_init_fn": seed_worker, "generator": generator}


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
    pin_memory: bool = False,
    seed: Optional[int] = None,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = datasets.CIFAR10(
        root=data_dir, train=True, transform=transform, download=download
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, transform=transform, download=download
    )

    #loader_kwargs = _seeded_loader_kwargs(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
        #**loader_kwargs,
,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader
