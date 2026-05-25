from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils.metrics import accuracy

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


def _prepare_batch(batch, device: torch.device):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        inputs, targets = batch
        return inputs.to(device), targets.to(device)
    if hasattr(batch, "to") and hasattr(batch, "y"):
        batch = batch.to(device)
        return batch, batch.y
    raise ValueError("Unsupported batch format from data loader.")


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: LossFn,
    device: torch.device,
    metric_fn: Optional[MetricFn] = None,
) -> Tuple[float, Optional[float]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    outputs_list = []
    targets_list = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = _prepare_batch(batch, device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            num_batches += 1
            if metric_fn is not None:
                outputs_list.append(outputs.detach().cpu())
                targets_list.append(targets.detach().cpu())

    if num_batches == 0:
        return 0.0, 0.0

    metric_value = None
    if metric_fn is not None:
        if outputs_list:
            all_outputs = torch.cat(outputs_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            metric_value = metric_fn(all_outputs, all_targets)
        else:
            metric_value = 0.0

    return total_loss / num_batches, metric_value


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 10,
    device: Optional[torch.device] = None,
    log_every_steps: int = 0,
    criterion: Optional[nn.Module] = None,
    loss_fn: Optional[LossFn] = None,
    train_metric_fn: Optional[MetricFn] = accuracy,
    val_metric_fn: Optional[MetricFn] = accuracy,
    metric_name: str = "acc",
    target_val_metric: Optional[float] = None,
    use_closure: bool = False,
    log_fn: Optional[Callable[[Dict[str, float], int], None]] = None,
) -> Dict[str, list]:
    device = device or torch.device("cpu")
    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    if loss_fn is None:
        def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return criterion(outputs, targets)

    train_metric_key = f"train_{metric_name}"
    val_metric_key = f"val_{metric_name}"

    history: Dict[str, list] = {
        "train_loss": [],
        train_metric_key: [],
        "val_loss": [],
        val_metric_key: [],
    }

    global_step = 0
    epoch_log_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_metric = 0.0
        metric_batches = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=0):
            inputs, targets = _prepare_batch(batch, device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            if use_closure:
                def closure():
                    outputs_cl = model(inputs)
                    return loss_fn(outputs_cl, targets)

                optimizer.step(closure=closure)
            else:
                optimizer.step()

            global_step += 1
            if log_fn is not None and log_every_steps > 0:
                if global_step % log_every_steps == 0:
                    step_metrics: Dict[str, float] = {
                        "train_loss_step": loss.item(),
                        "epoch": float(epoch + 1),
                        "batch": float(batch_idx),
                    }
                    step_stats = getattr(optimizer, "last_step_stats", None)
                    if isinstance(step_stats, dict) and step_stats:
                        for key in ("gamma", "update_rms"):
                            if key in step_stats:
                                step_metrics[key] = step_stats[key]
                    log_fn(step_metrics, global_step)

            total_loss += loss.item()
            if train_metric_fn is not None:
                total_metric += train_metric_fn(outputs.detach(), targets.detach())
                metric_batches += 1
            num_batches += 1

        if num_batches == 0:
            avg_loss = 0.0
            avg_metric = 0.0
        else:
            avg_loss = total_loss / num_batches
            avg_metric = (
                total_metric / metric_batches if metric_batches > 0 else 0.0
            )

        history["train_loss"].append(avg_loss)
        history[train_metric_key].append(avg_metric)

        if val_loader is not None:
            val_loss, val_metric = evaluate(
                model, val_loader, loss_fn, device, val_metric_fn
            )
            history["val_loss"].append(val_loss)
            history[val_metric_key].append(val_metric if val_metric is not None else 0.0)

        metrics = {"train_loss": avg_loss, train_metric_key: avg_metric}
        if val_loader is not None:
            metrics["val_loss"] = history["val_loss"][-1]
            metrics[val_metric_key] = history[val_metric_key][-1]
        if target_val_metric is not None:
            metrics["target_val_metric"] = target_val_metric

        if log_fn is not None:
            log_fn(metrics, global_step)

        
        if val_loader is None:
            print(
                f"Epoch [{epoch + 1}/{epochs}] loss={avg_loss:.4f} "
                f"{metric_name}={avg_metric:.4f}"
            )
        else:
            val_loss = metrics["val_loss"]
            val_metric = metrics[val_metric_key]
            print(
                f"Epoch [{epoch + 1}/{epochs}] loss={avg_loss:.4f} "
                f"{metric_name}={avg_metric:.4f} val_loss={val_loss:.4f} "
                f"val_{metric_name}={val_metric:.4f}"
            )

        if val_loader is not None and target_val_metric is not None:
            current_val_metric = history[val_metric_key][-1]
            if current_val_metric >= target_val_metric:
                print(
                    f"Target val_{metric_name} {target_val_metric:.4f} reached at "
                    f"epoch {epoch + 1} (val_{metric_name}={current_val_metric:.4f}). "
                    "Stopping."
                )
                break

    return history
