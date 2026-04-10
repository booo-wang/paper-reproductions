from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class AutoencoderOutput:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    predicted_count: torch.Tensor
    target: torch.Tensor
    decode_count: int


class PISA(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 96,
        max_n: int = 16,
        use_rho: bool = True,
        encoder_variant: str = "pisa",
    ):
        super().__init__()
        if encoder_variant not in {"pisa", "deepset"}:
            raise ValueError(f"Unsupported encoder variant: {encoder_variant}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_n = max_n
        self.use_rho = use_rho
        self.encoder_variant = encoder_variant

        projection = torch.randn(input_dim)
        projection = projection / projection.norm()
        self.register_buffer("projection", projection)

        self.key_embedding = nn.Embedding(max_n + 1, latent_dim)
        self.value_net = _mlp(input_dim, hidden_dim, latent_dim)
        self.size_encoder = nn.Linear(1, latent_dim, bias=False)

        self.deepset_phi = _mlp(input_dim, hidden_dim, latent_dim)
        self.deepset_post = _mlp(latent_dim, hidden_dim, latent_dim)

        self.decoder = _mlp(latent_dim, hidden_dim, input_dim)
        self.size_predictor = _mlp(latent_dim, hidden_dim, 1)

    def canonical_order(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= 1:
            return torch.arange(x.shape[0], device=x.device)
        if not self.use_rho:
            return torch.arange(x.shape[0], device=x.device)
        scores = x @ self.projection
        return torch.argsort(scores)

    def canonicalize(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.canonical_order(x)]

    def _key_indices(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.arange(1, n + 1, device=device, dtype=torch.long)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = self.canonical_order(x)
        target = x[order]
        n = target.shape[0]
        size_term = self.size_encoder(
            torch.tensor([[float(n)]], dtype=target.dtype, device=target.device)
        ).squeeze(0)

        if n == 0:
            return size_term, target, order

        if self.encoder_variant == "pisa":
            key_indices = self._key_indices(n, target.device)
            keys = self.key_embedding(key_indices)
            values = self.value_net(target)
            latent = (keys * values).sum(dim=0)
        else:
            features = self.deepset_phi(target)
            pooled = features.sum(dim=0, keepdim=True)
            latent = self.deepset_post(pooled).squeeze(0)

        return latent + size_term, target, order

    def decode(self, z: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0:
            return z.new_zeros((0, self.input_dim))
        key_indices = self._key_indices(n, z.device)
        queries = self.key_embedding(key_indices)
        element_states = z.unsqueeze(0) * queries
        return self.decoder(element_states)

    def predict_count(self, z: torch.Tensor) -> torch.Tensor:
        raw_count = self.size_predictor(z.unsqueeze(0)).squeeze(0)
        return torch.sigmoid(raw_count).squeeze(-1) * float(self.max_n)

    def infer_count(self, z: torch.Tensor) -> int:
        prediction = float(self.predict_count(z).item())
        return int(max(0, min(self.max_n, round(prediction))))

    def forward(
        self,
        x: torch.Tensor,
        decode_count: int | None = None,
        decode_mode: str = "predicted",
    ) -> AutoencoderOutput:
        latent, target, _ = self.encode(x)
        predicted_count = self.predict_count(latent)
        if decode_count is None:
            if decode_mode == "predicted":
                decode_count = self.infer_count(latent)
            elif decode_mode == "target":
                decode_count = int(target.shape[0])
            else:
                raise ValueError(f"Unsupported decode_mode: {decode_mode}")
        reconstruction = self.decode(latent, decode_count)
        return AutoencoderOutput(
            reconstruction=reconstruction,
            latent=latent,
            predicted_count=predicted_count,
            target=target,
            decode_count=int(decode_count),
        )


class DuplicateDetector(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def logits(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        features = torch.cat([a, b, torch.abs(a - b)], dim=-1)
        return self.network(features).squeeze(-1)

    def probability(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(a, b))


if __name__ == "__main__":
    torch.manual_seed(11)
    x = torch.randn(5, 6)
    model = PISA(latent_dim=48)
    output = model(x)
    print("Input shape:", tuple(x.shape))
    print("Canonical target shape:", tuple(output.target.shape))
    print("Latent shape:", tuple(output.latent.shape))
    print("Reconstruction shape:", tuple(output.reconstruction.shape))
    print("Predicted count:", float(output.predicted_count.item()))
