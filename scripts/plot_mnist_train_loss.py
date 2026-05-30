#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", context="paper", font_scale=1.4)

DATASETS = ["ogbg_molpcba", "cifar10", "mnist"]
OPTIMIZER_ORDER = [
    "adamx",
    "yogi",
    "sgd",
    "rmsprop",
    "radam",
    "lion",
    "amsgrad",
    "adan",
    "adamw",
    "adam",
    "adagrad",
]
BASE_COLORS = sns.color_palette("Set2", 8)
EXTRA_COLORS = [
    "#7A4F98",
    "#7DEAECFF",
    "#F2E391FF",
]
COLORS = list(BASE_COLORS) + EXTRA_COLORS
OPTIMIZER_RANK = {name: idx for idx, name in enumerate(OPTIMIZER_ORDER)}
OPTIMIZER_COLORS = {
    name: COLORS[idx] for idx, name in enumerate(OPTIMIZER_ORDER)
}


@dataclass
class RunSeries:
    optimizer: str
    dataset: str
    tag: str
    epochs: list[int]
    train_loss: list[float]
    source: Path


@dataclass
class OptimizerSeries:
    optimizer: str
    epochs: list[int]
    train_loss: list[float]
    sources: list[Path]


def parse_filename(path: Path) -> Optional[tuple[str, str, str]]:
    name = path.name
    if not name.startswith("losses_") or not name.endswith(".csv"):
        return None
    base = name[len("losses_") : -len(".csv")]
    if "_" not in base:
        return None
    optimizer, rest = base.split("_", 1)
    for dataset in sorted(DATASETS, key=len, reverse=True):
        if rest == dataset or rest.startswith(dataset + "_"):
            tag = rest[len(dataset) :]
            if tag.startswith("_"):
                tag = tag[1:]
            return optimizer, dataset, tag
    return None


def read_train_loss(path: Path) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    losses: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_loss = (row.get("train_loss") or "").strip()
            if not raw_loss:
                continue
            raw_epoch = (row.get("epoch") or "").strip()
            try:
                loss = float(raw_loss)
            except ValueError:
                continue
            if raw_epoch:
                try:
                    epoch_val = int(float(raw_epoch))
                except ValueError:
                    epoch_val = len(epochs) + 1
            else:
                epoch_val = len(epochs) + 1
            epochs.append(epoch_val)
            losses.append(loss)
    return epochs, losses


def _seed_tag_pattern(seed: int) -> re.Pattern:
    return re.compile(rf"(?:^|[-_])seed{seed}(?:$|[-_])")


def tag_has_seed(tag: str, seed: int) -> bool:
    return bool(_seed_tag_pattern(seed).search(tag))


def strip_seed_from_tag(tag: str, seed: Optional[int]) -> str:
    if seed is None:
        return tag or "default"
    pattern = _seed_tag_pattern(seed)
    stripped = pattern.sub("", tag)
    stripped = stripped.strip("-_ ")
    return stripped or "default"


def tag_has_no_tvm(tag: str) -> bool:
    return "no_tvm" in tag


def collect_series(
    metrics_dir: Path, dataset: str, seed: Optional[int]
) -> list[RunSeries]:
    series: list[RunSeries] = []
    for path in metrics_dir.glob("losses_*.csv"):
        parsed = parse_filename(path)
        if not parsed:
            continue
        optimizer, parsed_dataset, tag = parsed
        if parsed_dataset != dataset:
            continue
        if seed is not None and not tag_has_seed(tag, seed):
            continue
        if not tag_has_no_tvm(tag):
            continue
        epochs, losses = read_train_loss(path)
        if not epochs or not losses:
            continue
        series.append(
            RunSeries(
                optimizer=optimizer,
                dataset=parsed_dataset,
                tag=strip_seed_from_tag(tag or "default", seed),
                epochs=epochs,
                train_loss=losses,
                source=path,
            )
        )
    return series


def aggregate_by_optimizer(series: list[RunSeries]) -> list[OptimizerSeries]:
    by_optimizer: dict[str, list[RunSeries]] = {}
    for item in series:
        by_optimizer.setdefault(item.optimizer, []).append(item)

    aggregated: list[OptimizerSeries] = []
    for optimizer, runs in sorted(by_optimizer.items()):
        min_len = min(len(run.train_loss) for run in runs)
        if min_len == 0:
            continue
        epochs = runs[0].epochs[:min_len]
        avg_losses: list[float] = []
        for idx in range(min_len):
            avg = sum(run.train_loss[idx] for run in runs) / len(runs)
            avg_losses.append(avg)
        aggregated.append(
            OptimizerSeries(
                optimizer=optimizer,
                epochs=epochs,
                train_loss=avg_losses,
                sources=[run.source for run in runs],
            )
        )
    return aggregated


def sort_aggregated(aggregated: list[OptimizerSeries]) -> list[OptimizerSeries]:
    return sorted(
        aggregated,
        key=lambda item: (
            OPTIMIZER_RANK.get(item.optimizer, len(OPTIMIZER_RANK)),
            item.optimizer,
        ),
    )


def print_final_train_loss(aggregated: list[OptimizerSeries]) -> None:
    if not aggregated:
        return
    print("Final train_loss (last epoch):")
    for item in sort_aggregated(aggregated):
        if not item.train_loss:
            continue
        final_epoch = item.epochs[len(item.train_loss) - 1]
        final_loss = item.train_loss[-1]
        print(f"  {item.optimizer}: epoch {final_epoch} loss={final_loss:.6g}")


def write_final_train_loss(
    aggregated: list[OptimizerSeries], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["optimizer", "final_epoch", "final_train_loss"])
        for item in sort_aggregated(aggregated):
            if not item.train_loss:
                continue
            final_epoch = item.epochs[len(item.train_loss) - 1]
            writer.writerow([item.optimizer, final_epoch, item.train_loss[-1]])


def plot_series(
    series: list[RunSeries],
    dataset: str,
    output_path: Path,
    show: bool,
    legend: bool,
) -> list[OptimizerSeries]:
    if not series:
        raise SystemExit(f"No matching CSV files found for dataset '{dataset}'.")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    aggregated = aggregate_by_optimizer(series)
    aggregated = sort_aggregated(aggregated)
    for run in aggregated:
        color = OPTIMIZER_COLORS.get(run.optimizer)
        linewidth = 2.6 if run.optimizer == "adamx" else 1.8
        ax.plot(
            run.epochs,
            run.train_loss,
            linewidth=linewidth,
            label=run.optimizer,
            color=color,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_yscale("log")
    ax.legend(frameon=True, edgecolor="black")

    sns.despine()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot train_loss curves per optimizer for a dataset."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("metrics"),
        help="Directory containing losses_*.csv files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=DATASETS,
        help="Dataset name to plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to filter from filename tags (omit for all seeds).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image file path.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Output CSV for final train_loss summary.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window.",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Force legends even when there are many runs.",
    )
    args = parser.parse_args()

    metrics_dir = args.metrics_dir
    output_path = args.out
    summary_out = args.summary_out
    if output_path is None:
        if args.seed is None:
            output_path = metrics_dir / f"train_loss_{args.dataset}_no_tvm.pdf"
        else:
            output_path = (
                metrics_dir
                / f"train_loss_{args.dataset}_seed{args.seed}_no_tvm.pdf"
            )
    if summary_out is None:
        summary_dir = Path("notebooks")
        if args.seed is None:
            summary_out = (
                summary_dir / f"final_train_loss_{args.dataset}_no_tvm.csv"
            )
        else:
            summary_out = (
                summary_dir
                / f"final_train_loss_{args.dataset}_seed{args.seed}_no_tvm.csv"
            )

    series = collect_series(metrics_dir, args.dataset, args.seed)
    aggregated = plot_series(
        series, args.dataset, output_path, show=args.show, legend=args.legend
    )
    print_final_train_loss(aggregated)
    write_final_train_loss(aggregated, summary_out)
    print(f"Saved plot to {output_path}")
    print(f"Saved final train_loss to {summary_out}")


if __name__ == "__main__":
    main()
