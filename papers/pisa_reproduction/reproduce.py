from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import hop_union_set, sample_fusion_scene, sample_random_set
from model import DuplicateDetector, PISA


@dataclass
class TrainingHistory:
    steps: List[int]
    reconstruction: List[float]
    size: List[float]
    total: List[float]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def correlation_coefficient(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    if a_flat.numel() < 2:
        return 1.0
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    denom = a_centered.norm() * b_centered.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((a_centered @ b_centered / denom).item())


def train_autoencoder(
    model: PISA,
    steps: int,
    lr: float = 1e-3,
    size_loss_weight: float = 0.01,
    min_n: int = 1,
    max_n: int = 16,
    input_dim: int = 6,
    log_every: int = 500,
) -> TrainingHistory:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    history = TrainingHistory(steps=[], reconstruction=[], size=[], total=[])

    for step in range(1, steps + 1):
        x = sample_random_set(min_n=min_n, max_n=max_n, dim=input_dim)
        output = model(x)
        target_n = torch.tensor([float(output.target.shape[0])], dtype=torch.float32)

        reconstruction_loss = mse(output.reconstruction, output.target)
        size_loss = mse(output.predicted_count.view(1), target_n)
        total_loss = reconstruction_loss + size_loss_weight * size_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        history.steps.append(step)
        history.reconstruction.append(float(reconstruction_loss.item()))
        history.size.append(float(size_loss.item()))
        history.total.append(float(total_loss.item()))

        if step % log_every == 0:
            print(
                f"[train] step={step:5d} "
                f"recon={reconstruction_loss.item():.4f} "
                f"size={size_loss.item():.4f} "
                f"total={total_loss.item():.4f}"
            )

    return history


def evaluate_random_reconstruction(
    model: PISA,
    trials_per_size: int = 96,
    min_n: int = 1,
    max_n: int = 16,
    input_dim: int = 6,
) -> Dict[str, object]:
    model.eval()
    metrics: Dict[str, object] = {
        "per_size": [],
        "overall_mse": 0.0,
        "overall_corr": 0.0,
    }
    all_mse: List[float] = []
    all_corr: List[float] = []

    with torch.no_grad():
        for n in range(min_n, max_n + 1):
            size_mse: List[float] = []
            size_corr: List[float] = []
            for _ in range(trials_per_size):
                x = torch.randn(n, input_dim)
                output = model(x)
                mse_value = float(torch.mean((output.reconstruction - output.target) ** 2).item())
                corr_value = correlation_coefficient(output.target, output.reconstruction)
                size_mse.append(mse_value)
                size_corr.append(corr_value)
                all_mse.append(mse_value)
                all_corr.append(corr_value)

            metrics["per_size"].append(
                {
                    "n": n,
                    "mse": float(sum(size_mse) / len(size_mse)),
                    "corr": float(sum(size_corr) / len(size_corr)),
                }
            )

    metrics["overall_mse"] = float(sum(all_mse) / len(all_mse))
    metrics["overall_corr"] = float(sum(all_corr) / len(all_corr))
    return metrics


def plot_training_history(history: TrainingHistory, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history.steps, history.reconstruction, label="Reconstruction Loss")
    plt.plot(history.steps, history.size, label="Size Loss")
    plt.plot(history.steps, history.total, label="Total Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_size_metrics(metrics_map: Dict[str, Dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, metrics in metrics_map.items():
        per_size = metrics["per_size"]
        ns = [item["n"] for item in per_size]
        corr = [item["corr"] for item in per_size]
        mse = [item["mse"] for item in per_size]
        axes[0].plot(ns, corr, marker="o", label=name)
        axes[1].plot(ns, mse, marker="o", label=name)

    axes[0].set_title("Correlation vs Set Size")
    axes[0].set_xlabel("Number of Elements")
    axes[0].set_ylabel("Correlation")
    axes[0].set_ylim(0.0, 1.01)
    axes[1].set_title("MSE vs Set Size")
    axes[1].set_xlabel("Number of Elements")
    axes[1].set_ylabel("Mean Squared Error")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_interpolation_figure(model: PISA, output_path: Path, samples: int = 6) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x0 = torch.randn(8, model.input_dim)
        x1 = torch.randn(8, model.input_dim)
        z0, _ = model.encode(x0)
        z1, _ = model.encode(x1)
        alphas = torch.linspace(0.0, 1.0, samples)
        decoded_sets = []
        for alpha in alphas:
            z = (1.0 - alpha) * z0 + alpha * z1
            decoded_sets.append(model.decode(z, 8))

    fig, axes = plt.subplots(1, samples, figsize=(3 * samples, 3))
    if samples == 1:
        axes = [axes]
    for axis, alpha, decoded in zip(axes, alphas.tolist(), decoded_sets):
        axis.scatter(decoded[:, 0], decoded[:, 1], c=decoded[:, 2:5].clamp(0.0, 1.0))
        axis.set_title(f"alpha={alpha:.2f}")
        axis.set_xlim(-2.5, 2.5)
        axis.set_ylim(-2.5, 2.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    arc_length = 0.0
    for prev, curr in zip(decoded_sets[:-1], decoded_sets[1:]):
        arc_length += float(torch.norm(curr - prev, dim=1).sum().item())
    return {"arc_length": arc_length}


def train_duplicate_detector(
    detector: DuplicateDetector,
    steps: int = 2000,
    lr: float = 1e-3,
    input_dim: int = 6,
) -> List[float]:
    optimizer = optim.Adam(detector.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses: List[float] = []

    for step in range(1, steps + 1):
        base = torch.rand(64, input_dim)
        same_a = base + 0.01 * torch.randn_like(base)
        same_b = base + 0.01 * torch.randn_like(base)
        diff_a = torch.rand(64, input_dim)
        diff_b = torch.rand(64, input_dim)

        a = torch.cat([same_a, diff_a], dim=0)
        b = torch.cat([same_b, diff_b], dim=0)
        labels = torch.cat([torch.ones(64), torch.zeros(64)], dim=0)

        logits = detector.logits(a, b)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if step % 500 == 0:
            print(f"[filter] step={step:5d} loss={loss.item():.4f}")

    return losses


def evaluate_duplicate_detector(detector: DuplicateDetector, input_dim: int = 6) -> Dict[str, float]:
    detector.eval()
    with torch.no_grad():
        base = torch.rand(512, input_dim)
        same_a = base + 0.01 * torch.randn_like(base)
        same_b = base + 0.01 * torch.randn_like(base)
        diff_a = torch.rand(512, input_dim)
        diff_b = torch.rand(512, input_dim)

        same_prob = detector.probability(same_a, same_b)
        diff_prob = detector.probability(diff_a, diff_b)
        predictions = torch.cat([same_prob > 0.5, diff_prob <= 0.5], dim=0)
        accuracy = float(predictions.float().mean().item())
        return {
            "accuracy": accuracy,
            "same_mean_probability": float(same_prob.mean().item()),
            "different_mean_probability": float(diff_prob.mean().item()),
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
                    distance = float(torch.norm(element - centroid).item())
                    probability = float(
                        detector.probability(element.unsqueeze(0), centroid.unsqueeze(0)).item()
                    )
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


def run_fusion_demo(
    model: PISA,
    detector: DuplicateDetector,
    output_path: Path,
    layers: int = 2,
    seed: int = 123,
) -> Dict[str, object]:
    set_seed(seed)
    scene = sample_fusion_scene()

    with torch.no_grad():
        current_sets = [scene.local_observations(agent_idx).detach() for agent_idx in range(scene.num_agents)]
        rollouts = [current_sets]

        for _ in range(layers):
            next_sets: List[torch.Tensor] = []
            for agent_idx in range(scene.num_agents):
                reachable = torch.nonzero(scene.communication_adjacency[agent_idx], as_tuple=False).flatten().tolist()
                decoded_candidates: List[torch.Tensor] = []
                for neighbour_idx in reachable:
                    latent, _ = model.encode(current_sets[neighbour_idx])
                    decode_count = int(current_sets[neighbour_idx].shape[0])
                    decoded_candidates.append(model.decode(latent, decode_count).detach())
                merged = _merge_sets(decoded_candidates, detector).detach()
                next_sets.append(merged)
            current_sets = next_sets
            rollouts.append(current_sets)

    agent_of_interest = 0
    ground_truth_global = hop_union_set(scene, agent_of_interest, layers).detach()
    predicted_counts = [int(rollout[agent_of_interest].shape[0]) for rollout in rollouts]
    truth_count = int(ground_truth_global.shape[0])

    fig, axes = plt.subplots(1, layers + 1, figsize=(4.2 * (layers + 1), 4))
    if layers == 0:
        axes = [axes]
    for layer_idx, axis in enumerate(axes):
        prediction = rollouts[layer_idx][agent_of_interest]
        axis.scatter(
            ground_truth_global[:, 0],
            ground_truth_global[:, 1],
            c=ground_truth_global[:, 2:5].clamp(0.0, 1.0),
            s=ground_truth_global[:, 5].mul(600).tolist(),
            alpha=0.25,
            marker="o",
        )
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
        axis.scatter(
            scene.agent_positions[:, 0],
            scene.agent_positions[:, 1],
            marker="s",
            c="black",
            s=30,
            alpha=0.8,
        )
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

    return {
        "agent_index": agent_of_interest,
        "predicted_counts": predicted_counts,
        "ground_truth_global_count": truth_count,
    }


def run_ablation(output_dir: Path) -> Dict[str, object]:
    variants = {
        "PISA": PISA(latent_dim=48, hidden_dim=128, use_rho=True, encoder_variant="pisa"),
        "PISA (No rho)": PISA(latent_dim=48, hidden_dim=128, use_rho=False, encoder_variant="pisa"),
        "PISA (DeepSet Encoder)": PISA(
            latent_dim=48,
            hidden_dim=128,
            use_rho=True,
            encoder_variant="deepset",
        ),
    }

    summary: Dict[str, object] = {}
    plt.figure(figsize=(10, 5))
    for name, model in variants.items():
        print(f"[ablation] training {name}")
        history = train_autoencoder(model, steps=2000, log_every=1000)
        metrics = evaluate_random_reconstruction(model, trials_per_size=48)
        summary[name] = metrics
        plt.plot(history.steps, history.total, label=name)

    plt.xlabel("Training Step")
    plt.ylabel("Total Loss")
    plt.title("Ablation Study")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_training.png", dpi=180)
    plt.close()
    return summary


def main(output_dir: Path | None = None) -> None:
    set_seed(42)
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_summary: Dict[str, object] = {}
    latent_results: Dict[str, Dict[str, object]] = {}

    for latent_dim, steps in [(48, 3000), (96, 3000)]:
        print(f"[main] training PISA latent_dim={latent_dim}")
        model = PISA(latent_dim=latent_dim, hidden_dim=128, max_n=16)
        history = train_autoencoder(model, steps=steps, log_every=750)
        metrics = evaluate_random_reconstruction(model)
        latent_name = f"PISA-{latent_dim}"
        latent_results[latent_name] = metrics
        plot_training_history(
            history,
            output_dir / f"training_latent_{latent_dim}.png",
            f"PISA Training Curves (latent={latent_dim})",
        )

        if latent_dim == 96:
            interpolation_metrics = run_interpolation_figure(
                model,
                output_dir / "latent_interpolation.png",
            )
            detector = DuplicateDetector()
            train_duplicate_detector(detector)
            detector_metrics = evaluate_duplicate_detector(detector)
            fusion_metrics = run_fusion_demo(
                model,
                detector,
                output_dir / "fusion_rollout_agent0.png",
                layers=2,
            )
            experiment_summary["interpolation"] = interpolation_metrics
            experiment_summary["duplicate_filter"] = detector_metrics
            experiment_summary["fusion"] = fusion_metrics

    plot_size_metrics(latent_results, output_dir / "size_metrics.png")
    experiment_summary["random_reconstruction"] = latent_results

    ablation_summary = run_ablation(output_dir)
    experiment_summary["ablation"] = ablation_summary

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
    print(f"[main] metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
