from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch_geometric.loader import DataLoader


def download_ogbg(data_dir: str) -> None:
    PygGraphPropPredDataset(name="ogbg-molpcba", root=data_dir)


def get_ogbg_molpcba_loaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = False,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Evaluator, int, int]:
    loader_kwargs = {}
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

        def seed_worker(worker_id: int) -> None:
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        loader_kwargs = {"worker_init_fn": seed_worker, "generator": generator}

    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", root=data_dir)
    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )

    evaluator = Evaluator(name="ogbg-molpcba")
    return (
        train_loader,
        valid_loader,
        evaluator,
        dataset.num_tasks,
        dataset.num_node_features,
    )
