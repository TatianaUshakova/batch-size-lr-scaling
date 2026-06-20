"""Empirical test of optimal one-step learning rate vs batch size on MNIST.

This script implements the local experiment described in the README:
train a small MNIST MLP to fixed checkpoints, freeze each checkpoint, and
measure which hypothetical SGD step size gives the smallest one-step loss change for
different gradient minibatch sizes.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch import nn
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ----------------------------- Configuration -----------------------------

SEED = 1234
DEVICE = "auto"  # "auto", "cpu", "cuda", or "mps"
OUTPUT_DIR = "outputs/mnist_optimal_batch_lr"
DATA_DIR = "data"

CHECKPOINT_STEPS = [100, 500, 1500]
BATCH_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
K = 50
EPSILON_GRID = [
    1e-4,
    3e-4,
    1e-3,
    2e-3,
    3e-3,
    5e-3,
    7e-3,
    1e-2,
    1.5e-2,
    2e-2,
    3e-2,
    5e-2,
    7e-2,
    1e-1,
    1.5e-1,
    2e-1,
    3e-1,
    5e-1,
    1.0,
]
EVAL_SUBSET_SIZE = 10_000
EVAL_BATCH_SIZE = 512
TRAINING_BATCH_SIZE = 64
TRAINING_LR = 0.1


# ------------------------------- Experiment -------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    output_dir: str
    data_dir: str
    checkpoint_steps: list[int]
    batch_sizes: list[int]
    k: int
    epsilon_grid: list[float]
    eval_subset_size: int
    eval_batch_size: int
    training_batch_size: int
    training_lr: float


class MnistMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_device(device_config: str) -> torch.device:
    if device_config != "auto":
        return torch.device(device_config)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def config_from_constants() -> ExperimentConfig:
    return ExperimentConfig(
        seed=SEED,
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        data_dir=DATA_DIR,
        checkpoint_steps=CHECKPOINT_STEPS,
        batch_sizes=BATCH_SIZES,
        k=K,
        epsilon_grid=EPSILON_GRID,
        eval_subset_size=EVAL_SUBSET_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        training_batch_size=TRAINING_BATCH_SIZE,
        training_lr=TRAINING_LR,
    )


def make_datasets(config: ExperimentConfig) -> tuple[datasets.MNIST, list[int], list[int]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=config.data_dir,
        train=True,
        transform=transform,
        download=True,
    )

    if config.eval_subset_size >= len(train_dataset):
        raise ValueError("EVAL_SUBSET_SIZE must be smaller than the MNIST train set.")

    split_generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(len(train_dataset), generator=split_generator).tolist()
    eval_indices = permutation[: config.eval_subset_size]
    gradient_pool_indices = permutation[config.eval_subset_size :]
    return train_dataset, eval_indices, gradient_pool_indices


def infinite_loader(loader: DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def train_checkpoints(
    config: ExperimentConfig,
    device: torch.device,
    dataset: datasets.MNIST,
    gradient_pool_indices: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    model = MnistMlp().to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training_lr)

    training_subset = Subset(dataset, gradient_pool_indices)
    train_generator = torch.Generator().manual_seed(config.seed + 1)
    train_loader = DataLoader(
        training_subset,
        batch_size=config.training_batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
    )

    checkpoints: dict[int, dict[str, torch.Tensor]] = {}
    target_steps = set(config.checkpoint_steps)
    max_step = max(target_steps)

    model.train()
    for step, (images, labels) in enumerate(infinite_loader(train_loader), start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        if step in target_steps:
            checkpoints[step] = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            print(f"saved checkpoint at step {step}", flush=True)
        if step >= max_step:
            break

    return checkpoints


@torch.no_grad()
def eval_loss(
    model: nn.Module,
    params: dict[str, torch.Tensor],
    eval_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    for images, labels in eval_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = functional_call(model, params, (images,))
        batch_loss = criterion(logits, labels)
        total_loss += float(batch_loss.item()) * images.shape[0]
        total_examples += images.shape[0]

    return total_loss / total_examples


def sample_minibatch(
    dataset: datasets.MNIST,
    pool_indices: list[int],
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_positions = torch.randperm(len(pool_indices), generator=generator)[:batch_size]
    images: list[torch.Tensor] = []
    labels: list[int] = []
    for position in selected_positions.tolist():
        image, label = dataset[pool_indices[position]]
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def compute_gradient(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    parameter_names = {name for name, _ in model.named_parameters()}
    trainable_params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in state.items()
        if name in parameter_names
    }
    call_state = {
        name: trainable_params[name] if name in trainable_params else tensor
        for name, tensor in state.items()
    }
    images = images.to(device)
    labels = labels.to(device)

    logits = functional_call(model, call_state, (images,))
    loss = criterion(logits, labels)
    gradients = torch.autograd.grad(loss, tuple(trainable_params.values()))
    return {
        name: grad.detach()
        for name, grad in zip(trainable_params.keys(), gradients, strict=True)
    }


def make_updated_params(
    state: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    epsilon: float,
) -> dict[str, torch.Tensor]:
    return {
        name: tensor - epsilon * gradients[name] if name in gradients else tensor
        for name, tensor in state.items()
    }


def run_measurements(
    config: ExperimentConfig,
    device: torch.device,
    dataset: datasets.MNIST,
    eval_indices: list[int],
    gradient_pool_indices: list[int],
    checkpoints: dict[int, dict[str, torch.Tensor]],
    output_dir: Path,
) -> Path:
    model = MnistMlp().to(device)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    eval_loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=0,
    )

    raw_csv_path = output_dir / "raw_results.csv"
    sample_generator = torch.Generator().manual_seed(config.seed + 2)

    with raw_csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint_step",
                "batch_size",
                "epsilon",
                "sample_id",
                "loss_before",
                "loss_after",
                "delta_loss",
            ],
        )
        writer.writeheader()

        for checkpoint_step in config.checkpoint_steps:
            print(f"measuring checkpoint {checkpoint_step}", flush=True)
            params = {
                name: tensor.to(device)
                for name, tensor in checkpoints[checkpoint_step].items()
            }
            loss_before = eval_loss(model, params, eval_loader, criterion, device)

            for batch_size in config.batch_sizes:
                print(
                    f"  batch_size={batch_size}, loss_before={loss_before:.6f}",
                    flush=True,
                )
                for sample_id in range(config.k):
                    images, labels = sample_minibatch(
                        dataset,
                        gradient_pool_indices,
                        batch_size,
                        sample_generator,
                    )
                    gradients = compute_gradient(
                        model,
                        params,
                        images,
                        labels,
                        criterion,
                        device,
                    )

                    for epsilon in config.epsilon_grid:
                        updated_params = make_updated_params(params, gradients, epsilon)
                        loss_after = eval_loss(
                            model,
                            updated_params,
                            eval_loader,
                            criterion,
                            device,
                        )
                        writer.writerow(
                            {
                                "checkpoint_step": checkpoint_step,
                                "batch_size": batch_size,
                                "epsilon": epsilon,
                                "sample_id": sample_id,
                                "loss_before": loss_before,
                                "loss_after": loss_after,
                                "delta_loss": loss_before - loss_after,
                            }
                        )

    return raw_csv_path


def load_raw_results(raw_csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with raw_csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "batch_size": int(row["batch_size"]),
                    "epsilon": float(row["epsilon"]),
                    "sample_id": int(row["sample_id"]),
                    "loss_before": float(row["loss_before"]),
                    "loss_after": float(row["loss_after"]),
                    "delta_loss": float(row["delta_loss"]),
                }
            )
    return rows


def quadratic_minimum(epsilons: list[float], values: list[float]) -> tuple[float, float] | None:
    if len(epsilons) < 3:
        return None
    min_index = int(np.argmin(values))
    if min_index == 0 or min_index == len(epsilons) - 1:
        return None

    x = np.array(epsilons[min_index - 1 : min_index + 2], dtype=float)
    y = np.array(values[min_index - 1 : min_index + 2], dtype=float)
    coefficients = np.polyfit(x, y, deg=2)
    a, b, c = coefficients
    if a <= 0:
        return None
    epsilon_min = -b / (2 * a)
    if epsilon_min < x.min() or epsilon_min > x.max():
        return None
    value_min = a * epsilon_min**2 + b * epsilon_min + c
    return float(epsilon_min), float(value_min)


def epsilon_model(batch_size: np.ndarray, epsilon_max: float, b_noise: float) -> np.ndarray:
    return epsilon_max / (1.0 + b_noise / batch_size)


def aggregate_and_fit(
    config: ExperimentConfig,
    raw_csv_path: Path,
    output_dir: Path,
) -> None:
    rows = load_raw_results(raw_csv_path)
    grouped: dict[tuple[int, int, float], list[float]] = defaultdict(list)
    for row in rows:
        key = (int(row["checkpoint_step"]), int(row["batch_size"]), float(row["epsilon"]))
        grouped[key].append(float(row["delta_loss"]))

    aggregate_rows: list[dict[str, float]] = []
    for (checkpoint_step, batch_size, epsilon), values in sorted(grouped.items()):
        mean_delta = float(np.mean(values))
        stderr_delta = (
            float(np.std(values, ddof=1) / math.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        mean_loss_change = -mean_delta
        aggregate_rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "batch_size": batch_size,
                "epsilon": epsilon,
                "mean_delta_loss": mean_delta,
                "stderr_delta_loss": stderr_delta,
                "mean_loss_change": mean_loss_change,
                "stderr_loss_change": stderr_delta,
                "n": len(values),
            }
        )

    aggregate_path = output_dir / "aggregate_results.csv"
    with aggregate_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint_step",
                "batch_size",
                "epsilon",
                "mean_delta_loss",
                "stderr_delta_loss",
                "mean_loss_change",
                "stderr_loss_change",
                "n",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    opt_rows: list[dict[str, float]] = []
    fit_rows: list[dict[str, float]] = []
    for checkpoint_step in config.checkpoint_steps:
        checkpoint_rows = [
            row for row in aggregate_rows if row["checkpoint_step"] == checkpoint_step
        ]
        by_batch: dict[int, list[dict[str, float]]] = defaultdict(list)
        for row in checkpoint_rows:
            by_batch[int(row["batch_size"])].append(row)

        for batch_size in config.batch_sizes:
            rows_for_batch = sorted(by_batch[batch_size], key=lambda row: row["epsilon"])
            loss_changes = [row["mean_loss_change"] for row in rows_for_batch]
            epsilons = [row["epsilon"] for row in rows_for_batch]
            best_index = int(np.argmin(loss_changes))
            best_epsilon = epsilons[best_index]
            best_loss_change = loss_changes[best_index]
            refined = quadratic_minimum(epsilons, loss_changes)
            refined_epsilon = refined[0] if refined else best_epsilon
            refined_loss_change = refined[1] if refined else best_loss_change
            opt_rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "batch_size": batch_size,
                    "epsilon_opt_grid": best_epsilon,
                    "loss_change_opt_grid": best_loss_change,
                    "delta_loss_opt_grid": -best_loss_change,
                    "epsilon_opt_quadratic": refined_epsilon,
                    "loss_change_opt_quadratic": refined_loss_change,
                    "delta_loss_opt_quadratic": -refined_loss_change,
                }
            )

        checkpoint_opt_rows = [
            row for row in opt_rows if row["checkpoint_step"] == checkpoint_step
        ]
        batch_array = np.array(
            [row["batch_size"] for row in checkpoint_opt_rows],
            dtype=float,
        )
        epsilon_array = np.array(
            [row["epsilon_opt_grid"] for row in checkpoint_opt_rows],
            dtype=float,
        )

        try:
            initial_guess = [float(epsilon_array.max()), float(np.median(batch_array))]
            fitted, covariance = curve_fit(
                epsilon_model,
                batch_array,
                epsilon_array,
                p0=initial_guess,
                bounds=([0.0, 0.0], [np.inf, np.inf]),
                maxfev=10_000,
            )
            epsilon_max, b_noise = fitted
            stderr = np.sqrt(np.diag(covariance))
            fit_rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "epsilon_max": float(epsilon_max),
                    "b_noise": float(b_noise),
                    "epsilon_max_stderr": float(stderr[0]),
                    "b_noise_stderr": float(stderr[1]),
                }
            )
        except RuntimeError:
            fit_rows.append(
                {
                    "checkpoint_step": checkpoint_step,
                    "epsilon_max": float("nan"),
                    "b_noise": float("nan"),
                    "epsilon_max_stderr": float("nan"),
                    "b_noise_stderr": float("nan"),
                }
            )

    opt_path = output_dir / "epsilon_opt_results.csv"
    with opt_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint_step",
                "batch_size",
                "epsilon_opt_grid",
                "loss_change_opt_grid",
                "delta_loss_opt_grid",
                "epsilon_opt_quadratic",
                "loss_change_opt_quadratic",
                "delta_loss_opt_quadratic",
            ],
        )
        writer.writeheader()
        writer.writerows(opt_rows)

    fit_path = output_dir / "fit_results.csv"
    with fit_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint_step",
                "epsilon_max",
                "b_noise",
                "epsilon_max_stderr",
                "b_noise_stderr",
            ],
        )
        writer.writeheader()
        writer.writerows(fit_rows)

    make_plots(config, aggregate_rows, opt_rows, fit_rows, output_dir)
    make_normalized_plots(config, opt_rows, fit_rows, output_dir)
    make_delta_opt_vs_epsilon_opt_plot(config, opt_rows, output_dir)


def make_plots(
    config: ExperimentConfig,
    aggregate_rows: list[dict[str, float]],
    opt_rows: list[dict[str, float]],
    fit_rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    fit_by_checkpoint = {
        int(row["checkpoint_step"]): row
        for row in fit_rows
    }

    fig, axes = plt.subplots(
        1,
        len(config.checkpoint_steps),
        figsize=(7 * len(config.checkpoint_steps), 5),
        sharey=False,
    )
    if len(config.checkpoint_steps) == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(config.batch_sizes)))
    for ax, checkpoint_step in zip(axes, config.checkpoint_steps, strict=True):
        checkpoint_rows = [
            row for row in aggregate_rows
            if int(row["checkpoint_step"]) == checkpoint_step
        ]
        fit = fit_by_checkpoint[checkpoint_step]
        epsilon_max = float(fit["epsilon_max"])
        b_noise = float(fit["b_noise"])
        local_region_values = [
            float(row["mean_loss_change"])
            for row in checkpoint_rows
            if float(row["mean_loss_change"]) < 0.2
        ]
        local_region_epsilons = [
            float(row["epsilon"])
            for row in checkpoint_rows
            if float(row["mean_loss_change"]) < 0.2
        ]

        for batch_size, color in zip(config.batch_sizes, colors, strict=True):
            batch_rows = sorted(
                [
                    row for row in checkpoint_rows
                    if int(row["batch_size"]) == batch_size
                ],
                key=lambda row: float(row["epsilon"]),
            )
            epsilons = np.array([float(row["epsilon"]) for row in batch_rows])
            means = np.array([float(row["mean_loss_change"]) for row in batch_rows])
            stderrs = np.array([float(row["stderr_loss_change"]) for row in batch_rows])
            ax.errorbar(
                epsilons,
                means,
                yerr=stderrs,
                marker="o",
                linewidth=1.25,
                color=color,
                label=f"B={batch_size}",
            )

            if not math.isnan(epsilon_max) and not math.isnan(b_noise):
                predicted_epsilon = float(epsilon_model(batch_size, epsilon_max, b_noise))
                local_region_epsilons.append(predicted_epsilon)
                predicted_loss_change = float(
                    np.interp(predicted_epsilon, epsilons, means, left=means[0], right=means[-1])
                )
                ax.scatter(
                    [predicted_epsilon],
                    [predicted_loss_change],
                    marker="x",
                    s=130,
                    linewidths=5.0,
                    color="red",
                    zorder=20,
                )
                ax.scatter(
                    [predicted_epsilon],
                    [predicted_loss_change],
                    marker="x",
                    s=75,
                    linewidths=2.2,
                    color=color,
                    zorder=21,
                )

        if local_region_values:
            y_min = min(local_region_values)
            y_max = max(local_region_values)
            margin = 0.1 * max(y_max - y_min, 1e-3)
            ax.set_ylim(y_min - margin, y_max + margin)
        if local_region_epsilons:
            x_min = min(local_region_epsilons)
            x_max = max(local_region_epsilons)
            x_margin = 0.08 * max(x_max - x_min, 1e-3)
            ax.set_xlim(max(0.0, x_min - x_margin), x_max + x_margin)
        ax.set_xlabel("epsilon")
        ax.set_ylabel("mean one-step loss change (error bars: SE)")
        ax.set_title(f"checkpoint {checkpoint_step}")
        ax.grid(True, alpha=0.25)

    axes[-1].legend(ncol=2, fontsize=8, loc="best")
    fig.suptitle(
        "Calculating epsilon_opt on linear epsilon axes\n"
        "red-outlined colored x markers show fitted-theory epsilon_opt(B); lower is better",
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "combined_loss_change_vs_epsilon_with_theory.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_normalized_plots(
    config: ExperimentConfig,
    opt_rows: list[dict[str, float]],
    fit_rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    fit_by_checkpoint = {
        int(row["checkpoint_step"]): row
        for row in fit_rows
    }

    fig, axes = plt.subplots(
        1,
        len(config.checkpoint_steps),
        figsize=(6.5 * len(config.checkpoint_steps), 5),
        sharey=True,
    )
    if len(config.checkpoint_steps) == 1:
        axes = [axes]

    for ax, checkpoint_step in zip(axes, config.checkpoint_steps, strict=True):
        fit = fit_by_checkpoint[checkpoint_step]
        epsilon_max = float(fit["epsilon_max"])
        b_noise = float(fit["b_noise"])
        checkpoint_rows = sorted(
            [
                row for row in opt_rows
                if int(row["checkpoint_step"]) == checkpoint_step
            ],
            key=lambda row: int(row["batch_size"]),
        )

        if math.isnan(epsilon_max) or math.isnan(b_noise) or epsilon_max <= 0 or b_noise <= 0:
            ax.set_title(f"checkpoint {checkpoint_step}: fit unavailable")
            continue

        x_exp = np.array(
            [float(row["batch_size"]) / b_noise for row in checkpoint_rows],
            dtype=float,
        )
        y_exp = np.array(
            [float(row["epsilon_opt_grid"]) / epsilon_max for row in checkpoint_rows],
            dtype=float,
        )
        x_min = min(x_exp.min(), 1e-2)
        x_max = max(x_exp.max(), 1e2)
        x_theory = np.logspace(np.log10(x_min), np.log10(x_max), 300)
        y_theory = x_theory / (1.0 + x_theory)

        ax.scatter(x_exp, y_exp, s=55, label="experimental grid optimum")
        ax.plot(x_theory, y_theory, linewidth=2.2, label="theory: x / (1 + x)")
        ax.set_xscale("log")
        ax.set_xlabel("B / B_noise")
        ax.set_ylabel("epsilon_opt / epsilon_max")
        ax.set_title(
            f"checkpoint {checkpoint_step}\n"
            f"epsilon_max={epsilon_max:.3g}, B_noise={b_noise:.3g}"
        )
        ax.grid(True, alpha=0.25, which="both")
        ax.legend(fontsize=8)

    fig.suptitle("Normalized epsilon_opt(B) vs theoretical curve", y=1.04)
    fig.tight_layout()
    fig.savefig(output_dir / "epsilon_opt_normalized_subplots.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_delta_opt_vs_epsilon_opt_plot(
    config: ExperimentConfig,
    opt_rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    """Plot the expected local relation: optimal loss change is proportional to -epsilon_opt."""
    fig, axes = plt.subplots(
        1,
        len(config.checkpoint_steps),
        figsize=(6.5 * len(config.checkpoint_steps), 5),
        sharey=False,
    )
    if len(config.checkpoint_steps) == 1:
        axes = [axes]

    fit_rows: list[dict[str, float]] = []
    for ax, checkpoint_step in zip(axes, config.checkpoint_steps, strict=True):
        checkpoint_rows = sorted(
            [
                row for row in opt_rows
                if int(row["checkpoint_step"]) == checkpoint_step
            ],
            key=lambda row: float(row["epsilon_opt_grid"]),
        )
        epsilon_opt = np.array(
            [float(row["epsilon_opt_grid"]) for row in checkpoint_rows],
            dtype=float,
        )
        loss_change_opt = np.array(
            [float(row["loss_change_opt_grid"]) for row in checkpoint_rows],
            dtype=float,
        )

        slope = float(np.dot(epsilon_opt, loss_change_opt) / np.dot(epsilon_opt, epsilon_opt))
        x_fit = np.linspace(0.0, max(epsilon_opt.max(), 1e-12), 200)
        y_fit = slope * x_fit

        ax.scatter(epsilon_opt, loss_change_opt, s=55, label="measured optima")
        ax.plot(x_fit, y_fit, linewidth=2.0, label=f"through-origin fit: slope={slope:.3g}")
        ax.set_xlabel("epsilon_opt")
        ax.set_ylabel("optimal mean loss change")
        ax.set_title(f"checkpoint {checkpoint_step}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        fit_rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "slope_loss_change_per_epsilon_opt": slope,
            }
        )

    fig.suptitle("Optimal loss change vs optimal epsilon", y=1.03)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_change_opt_vs_epsilon_opt_linear.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fit_path = output_dir / "loss_change_opt_linear_fit.csv"
    with fit_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["checkpoint_step", "slope_loss_change_per_epsilon_opt"],
        )
        writer.writeheader()
        writer.writerows(fit_rows)


def main() -> None:
    config = config_from_constants()
    set_reproducibility(config.seed)
    device = select_device(config.device)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as file:
        json.dump({**asdict(config), "resolved_device": str(device)}, file, indent=2)

    dataset, eval_indices, gradient_pool_indices = make_datasets(config)
    checkpoints = train_checkpoints(config, device, dataset, gradient_pool_indices)
    raw_csv_path = run_measurements(
        config,
        device,
        dataset,
        eval_indices,
        gradient_pool_indices,
        checkpoints,
        output_dir,
    )
    aggregate_and_fit(config, raw_csv_path, output_dir)


if __name__ == "__main__":
    main()
