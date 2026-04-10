from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model import AutoencoderOutput


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class BaseSetAutoencoder(nn.Module):
    matching: str = "direct"

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, max_n: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_n = max_n
        self.size_predictor = _mlp(latent_dim, hidden_dim, 1)

    def canonicalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def predict_count(self, z: torch.Tensor) -> torch.Tensor:
        raw_count = self.size_predictor(z.unsqueeze(0)).squeeze(0)
        return torch.sigmoid(raw_count).squeeze(-1) * float(self.max_n)

    def infer_count(self, z: torch.Tensor) -> int:
        prediction = float(self.predict_count(z).item())
        return int(max(0, min(self.max_n, round(prediction))))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def decode(self, z: torch.Tensor, n: int) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        decode_count: int | None = None,
        decode_mode: str = "predicted",
    ) -> AutoencoderOutput:
        latent, target = self.encode(x)
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


class GRUSetAutoencoder(BaseSetAutoencoder):
    matching = "direct"

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, latent_dim: int = 96, max_n: int = 16):
        super().__init__(input_dim, hidden_dim, latent_dim, max_n)
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.query_embeddings = nn.Parameter(torch.randn(max_n, input_dim) * 0.05)
        self.decoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.canonicalize(x)
        if target.shape[0] == 0:
            latent = target.new_zeros(self.latent_dim)
            return latent, target
        _, hidden = self.encoder(target.unsqueeze(0))
        latent = self.latent_proj(hidden.squeeze(0).squeeze(0))
        return latent, target

    def decode(self, z: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0:
            return z.new_zeros((0, self.input_dim))
        query_tokens = self.query_embeddings[:n].unsqueeze(0)
        hidden = self.latent_to_hidden(z).unsqueeze(0).unsqueeze(0)
        decoded, _ = self.decoder(query_tokens, hidden)
        return self.output_proj(decoded.squeeze(0))


class DSPNSetAutoencoder(BaseSetAutoencoder):
    matching = "hungarian"

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        latent_dim: int = 96,
        max_n: int = 16,
        refinement_steps: int = 10,
        step_size: float = 0.45,
    ) -> None:
        super().__init__(input_dim, hidden_dim, latent_dim, max_n)
        self.refinement_steps = refinement_steps
        self.step_size = step_size
        self.phi = _mlp(input_dim, hidden_dim, latent_dim)
        self.initial_set = nn.Parameter(torch.randn(max_n, input_dim) * 0.05)

    def encode_elements(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_zeros(self.latent_dim)
        return self.phi(x).sum(dim=0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.canonicalize(x)
        return self.encode_elements(target), target

    def decode(self, z: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0:
            return z.new_zeros((0, self.input_dim))
        with torch.enable_grad():
            guess = self.initial_set[:n]
            latent_target = z.detach()
            for _ in range(self.refinement_steps):
                guess = guess.detach().requires_grad_(True)
                latent_guess = self.encode_elements(guess)
                match_loss = torch.mean((latent_guess - latent_target) ** 2)
                grad = torch.autograd.grad(match_loss, guess, create_graph=self.training)[0]
                guess = guess - self.step_size * grad
        return guess.detach() if not self.training else guess


class TSPNSetAutoencoder(BaseSetAutoencoder):
    matching = "hungarian"

    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, latent_dim: int = 96, max_n: int = 16):
        super().__init__(input_dim, hidden_dim, latent_dim, max_n)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.context_proj = nn.Linear(latent_dim, hidden_dim)
        self.query_embeddings = nn.Parameter(torch.randn(max_n, hidden_dim) * 0.05)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.canonicalize(x)
        if target.shape[0] == 0:
            return target.new_zeros(self.latent_dim), target
        encoded = self.encoder(self.input_proj(target).unsqueeze(0))
        latent = self.latent_proj(encoded.mean(dim=1).squeeze(0))
        return latent, target

    def decode(self, z: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 0:
            return z.new_zeros((0, self.input_dim))
        context = self.context_proj(z).unsqueeze(0)
        tokens = self.query_embeddings[:n].unsqueeze(0) + context.unsqueeze(1)
        if self.training:
            tokens = tokens + 0.01 * torch.randn_like(tokens)
        decoded = self.decoder(tokens)
        return self.output_proj(decoded.squeeze(0))
