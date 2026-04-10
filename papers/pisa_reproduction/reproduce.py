from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim

from baselines import DSPNSetAutoencoder, GRUSetAutoencoder, TSPNSetAutoencoder
from dataset import FusionScene, hop_union_set, sample_fusion_scene, sample_random_set
from model import DuplicateDetector, PISA


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class TrainingHistory:
    steps: List[int]
    reconstruction: List[float]
    size: List[float]
    total: List[float]
    eval_steps: List[int]
    eval_mse: List[float]
    eval_corr: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _zeros_like_count(length: int, dim: int, reference: torch.Tensor) -> torch.Tensor:
    return reference.new_zeros((length, dim))


def _pad_direct(target: torch.Tensor, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_n, prediction_n = target.shape[0], prediction.shape[0]
    max_n = max(target_n, prediction_n)
    if max_n == 0:
        return target.new_zeros((0, target.shape[-1])), prediction.new_zeros((0, prediction.shape[-1]))
    target_pad = _zeros_like_count(max_n, target.shape[-1], target)
    prediction_pad = _zeros_like_count(max_n, prediction.shape[-1], prediction)
    if target_n:
        target_pad[:target_n] = target
    if prediction_n:
        prediction_pad[:prediction_n] = prediction
    return target_pad, prediction_pad


def _pad_hungarian(target: torch.Tensor, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_n, prediction_n = target.shape[0], prediction.shape[0]
    dim = target.shape[-1] if target.ndim == 2 else prediction.shape[-1]
    max_n = max(target_n, prediction_n)
    if max_n == 0:
        return target.new_zeros((0, dim)), prediction.new_zeros((0, dim))

    target_pad = _zeros_like_count(max_n, dim, target if target_n else prediction)
    prediction_pad = _zeros_like_count(max_n, dim, prediction if prediction_n else target)
    if target_n == 0:
        prediction_pad[:prediction_n] = prediction
        return target_pad, prediction_pad
    if prediction_n == 0:
        target_pad[:target_n] = target
        return target_pad, prediction_pad

    cost = torch.cdist(target, prediction, p=2).pow(2).detach().cpu().numpy()
    rows, cols = linear_sum_assignment(cost)
    matched_target_rows = set()
    matched_pred_cols = set()

    slot = 0
    for row, col in zip(rows.tolist(), cols.tolist()):
        target_pad[slot] = target[row]
        prediction_pad[slot] = prediction[col]
        matched_target_rows.add(row)
        matched_pred_cols.add(col)
        slot += 1

    for row in range(target_n):
        if row not in matched_target_rows:
            target_pad[slot] = target[row]
            slot += 1
    for col in range(prediction_n):
        if col not in matched_pred_cols:
            prediction_pad[slot] = prediction[col]
            slot += 1
    return target_pad, prediction_pad


def align_sets(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> tuple[torch.Tensor, torch.Tensor]:
    if matching == "direct":
        return _pad_direct(target, prediction)
    if matching == "hungarian":
        return _pad_hungarian(target, prediction)
    raise ValueError(f"Unsupported matching mode: {matching}")


def mse_between_sets(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> torch.Tensor:
    aligned_target, aligned_prediction = align_sets(target, prediction, matching)
    if aligned_target.numel() == 0:
        return aligned_target.new_tensor(0.0)
    return torch.mean((aligned_prediction - aligned_target) ** 2)


def correlation_coefficient(target: torch.Tensor, prediction: torch.Tensor, matching: str) -> float:
    aligned_target, aligned_prediction = align_sets(target, prediction, matching)
    if aligned_target.numel() < 2:
        return 1.0
    a_flat = aligned_target.reshape(-1).float()
    b_flat = aligned_prediction.reshape(-1).float()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    denom = a_centered.norm() * b_centered.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((a_centered @ b_centered / denom).item())


def mean_minimum_arc_length(x0: torch.Tensor, x1: torch.Tensor) -> float:
    if x0.shape[0] == 0 and x1.shape[0] == 0:
        return 0.0
    aligned_x0, aligned_x1 = _pad_hungarian(x0, x1)
    return float(torch.norm(aligned_x1 - aligned_x0, dim=1).sum().item())


def train_autoencoder(
    model: nn.Module,
    steps: int,
    lr: float = 1e-3,
    size_loss_weight: float = 0.01,
    min_n: int = 0,
    max_n: int = 16,
    input_dim: int = 6,
    matching: str = "direct",
    decode_mode: str = "predicted",
    log_every: int = 500,
    eval_every: int = 500,
) -> TrainingHistory:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = TrainingHistory(
        steps=[],
        reconstruction=[],
        size=[],
        total=[],
        eval_steps=[],
        eval_mse=[],
        eval_corr=[],
    )

    for step in range(1, steps + 1):
        x = sample_random_set(min_n=min_n, max_n=max_n, dim=input_dim)
        output = model(x, decode_mode=decode_mode)
        target_n = torch.tensor([float(output.target.shape[0])], dtype=torch.float32)
        reconstruction_loss = mse_between_sets(output.target, output.reconstruction, matching)
        size_loss = torch.mean((output.predicted_count.view(1) - target_n) ** 2)
        total_loss = reconstruction_loss + size_loss_weight * size_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history.steps.append(step)
        history.reconstruction.append(float(reconstruction_loss.item()))
        history.size.append(float(size_loss.item()))
        history.total.append(float(total_loss.item()))

        if step % eval_every == 0:
            metrics = evaluate_random_reconstruction(
                model,
                matching=matching,
                trials_per_size=24,
                min_n=min_n,
                max_n=max_n,
                input_dim=input_dim,
                decode_mode="predicted",
            )
            history.eval_steps.append(step)
            history.eval_mse.append(float(metrics["overall_mse"]))
            history.eval_corr.append(float(metrics["overall_corr"]))

        if step % log_every == 0:
            print(
                f"[train] step={step:5d} "
                f"recon={reconstruction_loss.item():.4f} "
                f"size={size_loss.item():.4f} "
                f"total={total_loss.item():.4f}"
            )

    return history


def evaluate_random_reconstruction(
    model: nn.Module,
    matching: str,
    trials_per_size: int = 96,
    min_n: int = 0,
    max_n: int = 16,
    input_dim: int = 6,
    decode_mode: str = "predicted",
) -> Dict[str, object]:
    model.eval()
    metrics: Dict[str, object] = {
        "per_size": [],
        "overall_mse": 0.0,
        "overall_corr": 0.0,
        "overall_count_mae": 0.0,
    }
    all_mse: List[float] = []
    all_corr: List[float] = []
    all_count_errors: List[float] = []

    with torch.no_grad():
        for n in range(min_n, max_n + 1):
            size_mse: List[float] = []
            size_corr: List[float] = []
            size_count_errors: List[float] = []
            for _ in range(trials_per_size):
                x = torch.randn(n, input_dim)
                output = model(x, decode_mode=decode_mode)
                mse_value = float(mse_between_sets(output.target, output.reconstruction, matching).item())
                corr_value = correlation_coefficient(output.target, output.reconstruction, matching)
                count_error = abs(float(output.decode_count) - float(output.target.shape[0]))
                size_mse.append(mse_value)
                size_corr.append(corr_value)
                size_count_errors.append(count_error)
                all_mse.append(mse_value)
                all_corr.append(corr_value)
                all_count_errors.append(count_error)

            metrics["per_size"].append(
                {
                    "n": n,
                    "mse": float(sum(size_mse) / len(size_mse)),
                    "corr": float(sum(size_corr) / len(size_corr)),
                    "count_mae": float(sum(size_count_errors) / len(size_count_errors)),
                }
            )

    metrics["overall_mse"] = float(sum(all_mse) / len(all_mse))
    metrics["overall_corr"] = float(sum(all_corr) / len(all_corr))
    metrics["overall_count_mae"] = float(sum(all_count_errors) / len(all_count_errors))
    return metrics


def plot_training_history(histories: Dict[str, TrainingHistory], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, history in histories.items():
        axes[0].plot(history.steps, history.total, label=name)
        if history.eval_steps:
            axes[1].plot(history.eval_steps, history.eval_corr, marker="o", label=name)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Correlation")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Correlation")
    axes[1].set_ylim(0.0, 1.01)
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_size_metrics(metrics_map: Dict[str, Dict[str, object]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, metrics in metrics_map.items():
        per_size = metrics["per_size"]
        ns = [item["n"] for item in per_size]
        corr = [item["corr"] for item in per_size]
        mse = [item["mse"] for item in per_size]
        count_mae = [item["count_mae"] for item in per_size]
        axes[0].plot(ns, corr, marker="o", label=name)
        axes[1].plot(ns, mse, marker="o", label=name)
        axes[2].plot(ns, count_mae, marker="o", label=name)

    axes[0].set_title("Correlation vs Set Size")
    axes[0].set_xlabel("Number of Elements")
    axes[0].set_ylabel("Correlation")
    axes[0].set_ylim(0.0, 1.01)
    axes[1].set_title("MSE vs Set Size")
    axes[1].set_xlabel("Number of Elements")
    axes[1].set_ylabel("Mean Squared Error")
    axes[2].set_title("Count MAE vs Set Size")
    axes[2].set_xlabel("Number of Elements")
    axes[2].set_ylabel("Absolute Count Error")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_metrics(metrics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def build_random_suite(latent_dim: int, max_n: int = 16) -> Dict[str, tuple[nn.Module, str]]:
    return {
        "PISA": (PISA(latent_dim=latent_dim, hidden_dim=128, max_n=max_n), "direct"),
        "GRU": (GRUSetAutoencoder(latent_dim=latent_dim, hidden_dim=128, max_n=max_n), "direct"),
        "DSPN": (DSPNSetAutoencoder(latent_dim=latent_dim, hidden_dim=128, max_n=max_n), "hungarian"),
        "TSPN": (TSPNSetAutoencoder(latent_dim=latent_dim, hidden_dim=128, max_n=max_n), "hungarian"),
    }


def build_ablation_suite(latent_dim: int = 96, max_n: int = 16) -> Dict[str, tuple[nn.Module, str]]:
    return {
        "PISA": (PISA(latent_dim=latent_dim, hidden_dim=128, use_rho=True, encoder_variant="pisa", max_n=max_n), "direct"),
        "PISA (No rho)": (PISA(latent_dim=latent_dim, hidden_dim=128, use_rho=False, encoder_variant="pisa", max_n=max_n), "direct"),
        "PISA (Hungarian)": (PISA(latent_dim=latent_dim, hidden_dim=128, use_rho=True, encoder_variant="pisa", max_n=max_n), "hungarian"),
        "PISA (DeepSet Encoder)": (
            PISA(latent_dim=latent_dim, hidden_dim=128, use_rho=True, encoder_variant="deepset", max_n=max_n),
            "direct",
        ),
    }


def interpolation_statistics(
    model: nn.Module,
    matching: str,
    trials: int = 100,
    n: int = 8,
    samples: int = 8,
) -> Dict[str, float]:
    model.eval()
    arc_lengths: List[float] = []
    lower_bounds: List[float] = []
    with torch.no_grad():
        for _ in range(trials):
            x0 = torch.randn(n, model.input_dim)
            x1 = torch.randn(n, model.input_dim)
            output0 = model(x0, decode_mode="predicted")
            output1 = model(x1, decode_mode="predicted")
            alphas = torch.linspace(0.0, 1.0, samples)
            decoded_sets = []
            for alpha in alphas:
                z = (1.0 - alpha) * output0.latent + alpha * output1.latent
                decode_count = max(output0.decode_count, output1.decode_count)
                decoded_sets.append(model.decode(z, decode_count))
            arc_length = 0.0
            for prev, curr in zip(decoded_sets[:-1], decoded_sets[1:]):
                aligned_prev, aligned_curr = align_sets(prev, curr, matching)
                arc_length += float(torch.norm(aligned_curr - aligned_prev, dim=1).sum().item())
            arc_lengths.append(arc_length)
            lower_bounds.append(mean_minimum_arc_length(output0.target, output1.target))

    return {
        "mean_arc_length": float(sum(arc_lengths) / len(arc_lengths)),
        "min_arc_length": float(min(arc_lengths)),
        "max_arc_length": float(max(arc_lengths)),
        "mean_lower_bound": float(sum(lower_bounds) / len(lower_bounds)),
    }


def plot_interpolation_summary(summary: Dict[str, Dict[str, float]], output_path: Path) -> None:
    names = list(summary.keys())
    means = [summary[name]["mean_arc_length"] for name in names]
    lower_bounds = [summary[name]["mean_lower_bound"] for name in names]
    x = np.arange(len(names))
    width = 0.36
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, means, width=width, label="Mean decoded arc length")
    plt.bar(x + width / 2, lower_bounds, width=width, label="Mean endpoint lower bound")
    plt.xticks(x, names, rotation=10)
    plt.ylabel("Arc Length")
    plt.title("Latent Interpolation Statistics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_random_reconstruction(
    output_dir: Path,
    latent_dims: Sequence[int] = (48, 96),
    steps: int = 2000,
    trials_per_size: int = 96,
) -> Dict[str, object]:
    set_seed(42)
    summary: Dict[str, object] = {"latent_dims": {}}
    for latent_dim in latent_dims:
        histories: Dict[str, TrainingHistory] = {}
        metrics_map: Dict[str, Dict[str, object]] = {}
        latent_dir = output_dir / f"latent_{latent_dim}"
        latent_dir.mkdir(parents=True, exist_ok=True)
        suite = build_random_suite(latent_dim=latent_dim)

        for name, (model, matching) in suite.items():
            print(f"[4.1] training {name} with latent_dim={latent_dim}")
            history = train_autoencoder(
                model,
                steps=steps,
                matching=matching,
                decode_mode="predicted",
                log_every=max(100, steps // 4),
                eval_every=max(100, steps // 4),
            )
            histories[name] = history
            metrics_map[name] = evaluate_random_reconstruction(
                model,
                matching=matching,
                trials_per_size=trials_per_size,
                decode_mode="predicted",
            )

        plot_training_history(
            histories,
            latent_dir / "training_curves.png",
            f"4.1 Random Set Reconstruction (latent={latent_dim})",
        )
        plot_size_metrics(
            metrics_map,
            latent_dir / "size_metrics.png",
            f"4.1 Random Set Reconstruction (latent={latent_dim})",
        )
        save_metrics(metrics_map, latent_dir / "metrics.json")
        summary["latent_dims"][str(latent_dim)] = metrics_map

        if latent_dim == 96:
            interpolation_map: Dict[str, Dict[str, float]] = {}
            for name, (model, matching) in suite.items():
                interpolation_map[name] = interpolation_statistics(model, matching=matching)
            plot_interpolation_summary(interpolation_map, output_dir / "interpolation_summary.png")
            save_metrics(interpolation_map, output_dir / "interpolation_metrics.json")
            summary["interpolation"] = interpolation_map

    save_metrics(summary, output_dir / "summary.json")
    return summary


def canonicalize_with_ids(model: PISA, values: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[0] == 0:
        return values, ids
    order = model.canonical_order(values)
    return values[order], ids[order]


def duplicate_pairs_from_scene(scene: FusionScene, model: PISA) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left_examples: List[torch.Tensor] = []
    right_examples: List[torch.Tensor] = []
    labels: List[float] = []

    with torch.no_grad():
        per_agent_predictions: List[tuple[torch.Tensor, torch.Tensor]] = []
        for agent_idx in range(scene.num_agents):
            local_ids = scene.local_object_ids(agent_idx)
            local_values = scene.local_observations(agent_idx)
            canonical_values, canonical_ids = canonicalize_with_ids(model, local_values, local_ids)
            if canonical_values.shape[0] == 0:
                continue
            output = model(canonical_values, decode_mode="target")
            per_agent_predictions.append((output.reconstruction, canonical_ids))

    for i in range(len(per_agent_predictions)):
        pred_a, ids_a = per_agent_predictions[i]
        for j in range(i + 1, len(per_agent_predictions)):
            pred_b, ids_b = per_agent_predictions[j]
            for idx_a, elem_a in enumerate(pred_a):
                for idx_b, elem_b in enumerate(pred_b):
                    left_examples.append(elem_a)
                    right_examples.append(elem_b)
                    labels.append(float(ids_a[idx_a].item() == ids_b[idx_b].item()))

    if not left_examples:
        dim = model.input_dim
        return (
            torch.zeros(0, dim),
            torch.zeros(0, dim),
            torch.zeros(0),
        )

    return (
        torch.stack(left_examples),
        torch.stack(right_examples),
        torch.tensor(labels, dtype=torch.float32),
    )


def train_duplicate_detector(
    detector: DuplicateDetector,
    model: PISA,
    steps: int = 800,
    lr: float = 1e-3,
) -> List[float]:
    optimizer = optim.Adam(detector.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses: List[float] = []

    for step in range(1, steps + 1):
        scene = sample_fusion_scene()
        left, right, labels = duplicate_pairs_from_scene(scene, model)
        if left.shape[0] == 0:
            continue
        logits = detector.logits(left, right)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step % max(100, steps // 4) == 0:
            print(f"[filter] step={step:5d} loss={loss.item():.4f}")

    return losses


def evaluate_duplicate_detector(detector: DuplicateDetector, model: PISA, scenes: int = 32) -> Dict[str, float]:
    detector.eval()
    accuracies: List[float] = []
    positive_probs: List[float] = []
    negative_probs: List[float] = []

    with torch.no_grad():
        for _ in range(scenes):
            scene = sample_fusion_scene()
            left, right, labels = duplicate_pairs_from_scene(scene, model)
            if left.shape[0] == 0:
                continue
            probs = detector.probability(left, right)
            preds = (probs > 0.5).float()
            accuracies.append(float((preds == labels).float().mean().item()))
            if torch.any(labels == 1):
                positive_probs.append(float(probs[labels == 1].mean().item()))
            if torch.any(labels == 0):
                negative_probs.append(float(probs[labels == 0].mean().item()))

    return {
        "accuracy": float(sum(accuracies) / len(accuracies)),
        "same_mean_probability": float(sum(positive_probs) / len(positive_probs)),
        "different_mean_probability": float(sum(negative_probs) / len(negative_probs)),
    }


def _merge_sets(
    candidate_sets: List[torch.Tensor],
    detector: DuplicateDetector,
    threshold: float = 0.55,
    max_elements: int = 16,
) -> torch.Tensor:
    if not candidate_sets:
        return torch.empty(0, 6)

    merged_groups: List[List[torch.Tensor]] = []
    with torch.no_grad():
        for candidate_set in candidate_sets:
            for element in candidate_set:
                placed = False
                for group in merged_groups:
                    centroid = torch.stack(group, dim=0).mean(dim=0)
                    probability = float(
                        detector.probability(element.unsqueeze(0), centroid.unsqueeze(0)).item()
                    )
                    distance = float(torch.norm(element[:2] - centroid[:2]).item())
                    if probability >= threshold or distance <= 0.12:
                        group.append(element)
                        placed = True
                        break
                if not placed:
                    merged_groups.append([element])

    merged = torch.stack([torch.stack(group, dim=0).mean(dim=0) for group in merged_groups], dim=0)
    if merged.shape[0] > max_elements:
        keep = torch.argsort(merged[:, 0])[:max_elements]
        merged = merged[keep]
    return merged


def coverage_score(target: torch.Tensor, prediction: torch.Tensor, threshold: float = 0.14) -> float:
    if target.shape[0] == 0:
        return 1.0 if prediction.shape[0] == 0 else 0.0
    if prediction.shape[0] == 0:
        return 0.0
    distances = torch.cdist(target[:, :2], prediction[:, :2], p=2).cpu().numpy()
    rows, cols = linear_sum_assignment(distances)
    matched = 0
    for row, col in zip(rows.tolist(), cols.tolist()):
        if distances[row, col] <= threshold:
            matched += 1
    return matched / float(target.shape[0])


def run_fusion_rollout(
    scene: FusionScene,
    model: PISA,
    detector: DuplicateDetector,
    layers: int = 2,
) -> tuple[List[List[torch.Tensor]], Dict[str, object]]:
    with torch.no_grad():
        current_sets = [scene.local_observations(agent_idx).detach() for agent_idx in range(scene.num_agents)]
        rollouts = [current_sets]

        for _ in range(layers):
            next_sets: List[torch.Tensor] = []
            for agent_idx in range(scene.num_agents):
                reachable = torch.nonzero(scene.communication_adjacency[agent_idx], as_tuple=False).flatten().tolist()
                decoded_candidates: List[torch.Tensor] = []
                for neighbour_idx in reachable:
                    latent, _, _ = model.encode(current_sets[neighbour_idx])
                    decode_count = model.infer_count(latent)
                    decoded_candidates.append(model.decode(latent, decode_count).detach())
                merged = _merge_sets(decoded_candidates, detector, max_elements=model.max_n).detach()
                next_sets.append(merged)
            current_sets = next_sets
            rollouts.append(current_sets)

    layer_metrics: List[Dict[str, float]] = []
    for layer_idx, rollout in enumerate(rollouts):
        coverage_values: List[float] = []
        corr_values: List[float] = []
        count_errors: List[float] = []
        for agent_idx in range(scene.num_agents):
            truth = hop_union_set(scene, agent_idx, layer_idx)
            prediction = rollout[agent_idx]
            coverage_values.append(coverage_score(truth, prediction))
            corr_values.append(correlation_coefficient(truth, prediction, matching="hungarian"))
            count_errors.append(abs(float(prediction.shape[0]) - float(truth.shape[0])))
        layer_metrics.append(
            {
                "layer": float(layer_idx),
                "mean_coverage": float(sum(coverage_values) / len(coverage_values)),
                "mean_corr": float(sum(corr_values) / len(corr_values)),
                "mean_count_error": float(sum(count_errors) / len(count_errors)),
            }
        )

    summary = {"layers": layer_metrics}
    return rollouts, summary


def plot_fusion_rollout(
    scene: FusionScene,
    rollouts: List[List[torch.Tensor]],
    output_path: Path,
    agent_of_interest: int = 0,
) -> None:
    layers = len(rollouts) - 1
    fig, axes = plt.subplots(1, layers + 1, figsize=(4.2 * (layers + 1), 4))
    if layers == 0:
        axes = [axes]
    for layer_idx, axis in enumerate(axes):
        prediction = rollouts[layer_idx][agent_of_interest]
        truth = hop_union_set(scene, agent_of_interest, layer_idx)
        axis.scatter(
            truth[:, 0],
            truth[:, 1],
            c=truth[:, 2:5].clamp(0.0, 1.0),
            s=truth[:, 5].mul(600).tolist(),
            alpha=0.25,
            marker="o",
        )
        if prediction.shape[0]:
            axis.scatter(
                prediction[:, 0],
                prediction[:, 1],
                c=prediction[:, 2:5].clamp(0.0, 1.0),
                s=prediction[:, 5].abs().mul(600).clamp_min(20).tolist(),
                alpha=0.9,
                marker="o",
                edgecolors="black",
                linewidths=0.4,
            )
        axis.scatter(scene.agent_positions[:, 0], scene.agent_positions[:, 1], marker="s", c="black", s=30, alpha=0.8)
        axis.scatter(
            scene.agent_positions[agent_of_interest, 0],
            scene.agent_positions[agent_of_interest, 1],
            marker="s",
            c="#1f77b4",
            s=80,
        )
        axis.set_title(f"Layer {layer_idx}: {prediction.shape[0]} objects")
        axis.set_xlim(-0.05, 1.05)
        axis.set_ylim(-0.05, 1.05)
        axis.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_sensor_fusion(
    output_dir: Path,
    autoencoder_steps: int = 1500,
    detector_steps: int = 800,
    evaluation_scenes: int = 32,
    layers: int = 2,
) -> Dict[str, object]:
    set_seed(123)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = PISA(latent_dim=96, hidden_dim=128, max_n=16)
    print("[4.2] pretraining PISA backbone for sensor fusion")
    train_autoencoder(
        model,
        steps=autoencoder_steps,
        matching="direct",
        decode_mode="predicted",
        log_every=max(100, autoencoder_steps // 4),
        eval_every=max(100, autoencoder_steps // 4),
    )

    detector = DuplicateDetector()
    print("[4.2] training duplicate detector from decoded local observations")
    losses = train_duplicate_detector(detector, model=model, steps=detector_steps)
    filter_metrics = evaluate_duplicate_detector(detector, model=model, scenes=evaluation_scenes)

    rollout_metrics: List[Dict[str, object]] = []
    example_scene = None
    example_rollouts = None
    for scene_idx in range(evaluation_scenes):
        scene = sample_fusion_scene()
        rollouts, summary = run_fusion_rollout(scene, model, detector, layers=layers)
        rollout_metrics.append(summary)
        if scene_idx == 0:
            example_scene = scene
            example_rollouts = rollouts

    layer_stats: Dict[int, Dict[str, float]] = {}
    for rollout in rollout_metrics:
        for layer_info in rollout["layers"]:
            layer = int(layer_info["layer"])
            layer_stats.setdefault(layer, {"coverage": [], "corr": [], "count_error": []})
            layer_stats[layer]["coverage"].append(layer_info["mean_coverage"])
            layer_stats[layer]["corr"].append(layer_info["mean_corr"])
            layer_stats[layer]["count_error"].append(layer_info["mean_count_error"])

    aggregated_layers = []
    for layer in sorted(layer_stats):
        aggregated_layers.append(
            {
                "layer": layer,
                "mean_coverage": float(np.mean(layer_stats[layer]["coverage"])),
                "mean_corr": float(np.mean(layer_stats[layer]["corr"])),
                "mean_count_error": float(np.mean(layer_stats[layer]["count_error"])),
            }
        )

    if example_scene is not None and example_rollouts is not None:
        plot_fusion_rollout(example_scene, example_rollouts, output_dir / "fusion_rollout_agent0.png")

    if losses:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel("Training Step")
        plt.ylabel("BCE Loss")
        plt.title("Duplicate Filter Training")
        plt.tight_layout()
        plt.savefig(output_dir / "duplicate_filter_training.png", dpi=180)
        plt.close()

    metrics = {
        "duplicate_filter": filter_metrics,
        "fusion_layers": aggregated_layers,
    }
    save_metrics(metrics, output_dir / "metrics.json")
    return metrics


def run_ablation(
    output_dir: Path,
    steps: int = 1500,
    trials_per_size: int = 64,
) -> Dict[str, object]:
    set_seed(7)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite = build_ablation_suite()
    histories: Dict[str, TrainingHistory] = {}
    metrics_map: Dict[str, Dict[str, object]] = {}

    for name, (model, matching) in suite.items():
        print(f"[ablation] training {name}")
        histories[name] = train_autoencoder(
            model,
            steps=steps,
            matching=matching,
            decode_mode="predicted",
            log_every=max(100, steps // 4),
            eval_every=max(100, steps // 4),
        )
        metrics_map[name] = evaluate_random_reconstruction(
            model,
            matching=matching,
            trials_per_size=trials_per_size,
            decode_mode="predicted",
        )

    plot_training_history(histories, output_dir / "training_curves.png", "Ablation: PISA Components")
    plot_size_metrics(metrics_map, output_dir / "size_metrics.png", "Ablation: PISA Components")
    save_metrics(metrics_map, output_dir / "metrics.json")
    return metrics_map


def run_full(output_dir: Path, random_steps: int = 2000, fusion_steps: int = 1500, detector_steps: int = 800) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    random_metrics = run_random_reconstruction(output_dir / "random_reconstruction", steps=random_steps)
    fusion_metrics = run_sensor_fusion(
        output_dir / "sensor_fusion",
        autoencoder_steps=fusion_steps,
        detector_steps=detector_steps,
    )
    ablation_metrics = run_ablation(output_dir / "ablation", steps=max(1200, random_steps // 2))
    summary = {
        "random_reconstruction": random_metrics,
        "sensor_fusion": fusion_metrics,
        "ablation": ablation_metrics,
    }
    save_metrics(summary, output_dir / "summary.json")
    return summary
