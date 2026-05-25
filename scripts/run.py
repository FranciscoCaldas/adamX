from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import wandb

from datasets import (
    get_cifar10_loaders,
    get_mnist_loaders,
    get_ogbg_molpcba_loaders,
)
from models import CifarNet, GNN, ImprovedCNN, SimpleMLP
from optimizers import AdamX
from training import train
from utils import get_device, set_seed
from utils.metrics import accuracy


def build_model(dataset: str, model_name: str, dataset_info: Optional[dict] = None):
    if dataset == "mnist":
        if model_name == "mlp":
            return SimpleMLP(input_dim=28 * 28, num_classes=10)
        if model_name == "cnn":
            return ImprovedCNN(in_channels=1, num_classes=10, input_size=28)
    if dataset == "cifar10":
        if model_name == "mlp":
            return SimpleMLP(input_dim=32 * 32 * 3, num_classes=10)
        if model_name == "cnn":
            return ImprovedCNN(in_channels=3, num_classes=10, input_size=32)
        if model_name == "cifarnet":
            return CifarNet(in_channels=3, num_classes=10, input_size=32)
    if dataset == "ogbg_molpcba":
        if model_name != "gnn":
            raise ValueError("OGBG requires model_name='gnn'.")
        if GNN is None:
            raise ImportError("GNN model requires torch-geometric.")
        if not dataset_info:
            raise ValueError("Missing dataset info for OGBG model build.")
        return GNN(
            num_node_features=dataset_info["num_node_features"],
            num_outputs=dataset_info["num_tasks"],
        )

    raise ValueError(f"Unsupported dataset/model combination: {dataset}/{model_name}")


def build_optimizer(name: str, model: torch.nn.Module, args: argparse.Namespace):
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps
        )
    if name == "adamx":
        return AdamX(
            model.parameters(),
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            alpha=args.alpha,
            lambda_exp=args.lambda_exp,
            cc=args.cc,
            track_stats=args.log_every_steps > 0,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum
        )

    raise ValueError(f"Unsupported optimizer: {name}")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object.")
    return data


def normalize_config_keys(raw_config: dict) -> dict:
    normalized: dict = {}
    for key, value in raw_config.items():
        normalized_key = key.replace("-", "_")
        if normalized_key in normalized:
            raise ValueError(
                f"Duplicate config key after normalization: {normalized_key}"
            )
        normalized[normalized_key] = value
    return normalized


def extract_dataset_overrides(raw_config: dict) -> tuple[dict, dict]:
    config = dict(raw_config)
    overrides = config.pop("dataset_overrides", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("Config key 'dataset_overrides' must be a JSON object.")
    normalized_overrides: dict = {}
    for dataset_name, override in overrides.items():
        if not isinstance(override, dict):
            raise ValueError("Each dataset override must be a JSON object.")
        normalized_overrides[dataset_name] = normalize_config_keys(override)
    return config, normalized_overrides


def arg_in_argv(flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def list_config_names(config_dir: Path, fallback: list[str]) -> list[str]:
    if not config_dir.exists():
        return list(fallback)
    names = sorted(path.stem for path in config_dir.glob("*.json") if path.is_file())
    return names or list(fallback)


def load_dataset_target(dataset: str, repo_root: Path) -> tuple[str, float]:
    config_path = repo_root / "configs" / "dataset_target" / f"{dataset}.json"
    config = normalize_config_keys(load_config(config_path))
    config_dataset = config.get("dataset", dataset)
    if config_dataset != dataset:
        raise ValueError(
            f"Dataset target config {config_path} is for '{config_dataset}', "
            f"expected '{dataset}'."
        )
    metric_name = None
    target_value = None
    if "target_metric" in config and "target_value" in config:
        metric_name = str(config["target_metric"]).lower()
        target_value = config["target_value"]
    elif "target_val_acc" in config:
        metric_name = "acc"
        target_value = config["target_val_acc"]
    elif "target_val_map" in config:
        metric_name = "map"
        target_value = config["target_val_map"]

    if metric_name is None or target_value is None:
        raise ValueError(
            f"Missing target metric config in dataset target config: {config_path}"
        )
    if not isinstance(target_value, (int, float)):
        raise ValueError("Target metric value must be a number.")
    return metric_name, float(target_value)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_config_dir = repo_root / "configs" / "datasets"
    optimizer_config_dir = repo_root / "configs" / "optimizers"
    dataset_config_names = list_config_names(
        dataset_config_dir,
        ["mnist", "cifar10", "ogbg_molpcba"],
    )
    optimizer_config_names = list_config_names(
        optimizer_config_dir,
        ["adam", "adamx", "sgd"],
    )

    parser = argparse.ArgumentParser(description="Run AdamX benchmarks.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Legacy path to a JSON config file with benchmark settings.",
    )
    parser.add_argument(
        "--dataset-config",
        "-dc",
        choices=dataset_config_names,
        type=str,
        default=None,
        help="Dataset config name (defaults to --dataset).",
    )
    parser.add_argument(
        "--optimizer-config",
        "-oc",
        choices=optimizer_config_names,
        type=str,
        default=None,
        help="Optimizer config name (defaults to --optimizer).",
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "ogbg_molpcba"],
        default="mnist",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "cifarnet", "gnn"],
        default="cnn",
    )
    parser.add_argument("--optimizer", choices=["adam", "adamx", "sgd"], default="adamx")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--lambda-exp", type=float, default=1.0)
    parser.add_argument("--cc", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=1,
        help="Log metrics every N optimizer steps (0 disables).",
    )
    parser.add_argument(
        "--closure",
        action="store_false",
        help="Whether to use optimizer closures (AdamX only).",
    )

    pre_args, _ = parser.parse_known_args()
    if pre_args.config and (pre_args.dataset_config or pre_args.optimizer_config):
        raise ValueError(
            "Use --config or --dataset-config/--optimizer-config, not both."
        )

    dataset_overrides: dict = {}
    config: dict = {}

    if pre_args.config:
        config = normalize_config_keys(load_config(pre_args.config))
        config, dataset_overrides = extract_dataset_overrides(config)
    else:
        dataset_config_name = pre_args.dataset_config or pre_args.dataset
        optimizer_config_name = pre_args.optimizer_config or pre_args.optimizer
        dataset_config_path = dataset_config_dir / f"{dataset_config_name}.json"
        optimizer_config_path = optimizer_config_dir / f"{optimizer_config_name}.json"
        
        dataset_config = normalize_config_keys(load_config(dataset_config_path))
        optimizer_config = normalize_config_keys(load_config(optimizer_config_path))
        optimizer_config, dataset_overrides = extract_dataset_overrides(optimizer_config)
        config.update(dataset_config)
        config.update(optimizer_config)

        dataset_flag = arg_in_argv("--dataset")
        optimizer_flag = arg_in_argv("--optimizer")
        
        dataset_from_config = dataset_config.get("dataset", dataset_config_name)
        if dataset_from_config != dataset_config_name:
            raise ValueError(
                f"Dataset config {dataset_config_path} is for '{dataset_from_config}', "
                f"expected '{dataset_config_name}'."
            )
        if dataset_flag and dataset_from_config != pre_args.dataset:
            raise ValueError(
                f"Dataset config {dataset_config_path} is for '{dataset_from_config}', "
                f"expected '{pre_args.dataset}'."
            )

        optimizer_from_config = optimizer_config.get("optimizer", optimizer_config_name)
        if optimizer_from_config != optimizer_config_name:
            raise ValueError(
                f"Optimizer config {optimizer_config_path} is for '{optimizer_from_config}', "
                f"expected '{optimizer_config_name}'."
            )
        if optimizer_flag and optimizer_from_config != pre_args.optimizer:
            raise ValueError(
                f"Optimizer config {optimizer_config_path} is for '{optimizer_from_config}', "
                f"expected '{pre_args.optimizer}'."
            )

    if dataset_overrides:
        dataset_name = config.get("dataset", pre_args.dataset)
        override = dataset_overrides.get(dataset_name)
        if override:
            config.update(override)

    valid_keys = {
        action.dest
        for action in parser._actions
        if action.dest not in {"help", "config", "dataset_config", "optimizer_config"}
    }
    unknown = sorted(key for key in config if key not in valid_keys)
    if unknown:
        raise ValueError("Unknown config keys: " + ", ".join(unknown))
    if "betas" in config:
        betas = config["betas"]
        if not isinstance(betas, (list, tuple)) or len(betas) != 2:
            raise ValueError("Config key 'betas' must be a list of two floats.")
    parser.set_defaults(**config)

    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return get_device(prefer_cuda=True)


def get_loaders(args: argparse.Namespace):
    if args.dataset == "mnist":
        train_loader, val_loader = get_mnist_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.device != "cpu",
        )
        return train_loader, val_loader, {}
    if args.dataset == "ogbg_molpcba":
        if get_ogbg_molpcba_loaders is None:
            raise ImportError("OGBG loaders require ogb and torch-geometric.")
        train_loader, val_loader, evaluator, num_tasks, num_node_features = (
            get_ogbg_molpcba_loaders(
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.device != "cpu",
            )
        )
        return train_loader, val_loader, {
            "evaluator": evaluator,
            "num_tasks": num_tasks,
            "num_node_features": num_node_features,
        }
    train_loader, val_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )
    return train_loader, val_loader, {}


def save_history(
    history: dict, output_dir: Path, run_id: str, metric_name: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"losses_{run_id}.csv"
    train_metric_key = f"train_{metric_name}"
    val_metric_key = f"val_{metric_name}"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "train_loss", "val_loss", train_metric_key, val_metric_key]
        )
        for epoch in range(len(history["train_loss"])):
            val_loss = history["val_loss"][epoch] if history["val_loss"] else ""
            val_metric = history[val_metric_key][epoch] if history[val_metric_key] else ""
            train_metric = (
                history[train_metric_key][epoch] if history[train_metric_key] else ""
            )
            writer.writerow(
                [
                    epoch + 1,
                    history["train_loss"][epoch],
                    val_loss,
                    train_metric,
                    val_metric,
                ]
            )
    return path


def main() -> None:
    args = parse_args()

    #seed = 42
    set_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    metric_name, target_val_metric = load_dataset_target(args.dataset, repo_root)
    metric_name = metric_name.lower()
    wandb_config = vars(args).copy()
    wandb_config["target_metric_name"] = metric_name
    wandb_config["target_val_metric"] = target_val_metric

    run = wandb.init(
        entity="mlspace",
        project="adamX",
        config=wandb_config,
    )

    device = resolve_device(args.device)
    train_loader, val_loader, dataset_info = get_loaders(args)

    model = build_model(args.dataset, args.model, dataset_info)
    optimizer = build_optimizer(args.optimizer, model, args)

    if args.optimizer == "adamx" and not args.closure:
        use_closure = False
    elif args.optimizer == "adamx" and args.closure:
        use_closure = True
    else: 
        use_closure = False

    loss_fn = None
    train_metric_fn = accuracy
    val_metric_fn = accuracy
    if args.dataset == "ogbg_molpcba":
        evaluator = dataset_info.get("evaluator")
        if evaluator is None:
            raise ValueError("Missing evaluator for OGBG metric handling.")
        if metric_name != "map":
            raise ValueError("OGBG metric must be mAP.")
        metric_name = "map"

        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            targets = targets.to(logits.dtype)
            if targets.shape != logits.shape:
                targets = targets.view_as(logits)
            is_labeled = ~torch.isnan(targets)
            if is_labeled.sum().item() == 0:
                return torch.tensor(0.0, device=logits.device)
            return F.binary_cross_entropy_with_logits(logits[is_labeled], targets[is_labeled])

        def val_metric_fn(logits: torch.Tensor, targets: torch.Tensor) -> float:
            y_true = targets.detach().cpu().numpy()
            y_pred = torch.sigmoid(logits.detach()).cpu().numpy()
            return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["ap"]

        train_metric_fn = None

    def log_metrics(metrics: dict, step: int) -> None:
        run.log(metrics, step=step)

    history = train(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        device=device,
        log_every_steps=args.log_every_steps,
        loss_fn=loss_fn,
        train_metric_fn=train_metric_fn,
        val_metric_fn=val_metric_fn,
        metric_name=metric_name,
        target_val_metric=target_val_metric,
        use_closure=use_closure,
        log_fn=log_metrics,
    )

    metrics_path = save_history(history, repo_root / "metrics", run.id, metric_name)

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else 0.0
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
    print(
        f"Done: dataset={args.dataset} model={args.model} optimizer={args.optimizer} "
        f"train_loss={final_train_loss:.4f} val_loss={final_val_loss:.4f}"
    )
    print(f"Saved losses to {metrics_path}")

    run.finish()


if __name__ == "__main__":
    main()
