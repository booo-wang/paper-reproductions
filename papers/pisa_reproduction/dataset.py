from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List

import torch


@dataclass
class FusionScene:
    agent_positions: torch.Tensor
    object_features: torch.Tensor
    communication_adjacency: torch.Tensor
    observation_mask: torch.Tensor

    @property
    def num_agents(self) -> int:
        return int(self.agent_positions.shape[0])

    @property
    def num_objects(self) -> int:
        return int(self.object_features.shape[0])

    def local_object_ids(self, agent_idx: int) -> torch.Tensor:
        return torch.nonzero(self.observation_mask[agent_idx], as_tuple=False).flatten()

    def local_observations(self, agent_idx: int) -> torch.Tensor:
        return self.object_features[self.local_object_ids(agent_idx)]


def sample_random_set(
    min_n: int = 0,
    max_n: int = 16,
    dim: int = 6,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if generator is None:
        n = random.randint(min_n, max_n)
        return torch.randn(n, dim)

    n = int(torch.randint(min_n, max_n + 1, (1,), generator=generator).item())
    return torch.randn(n, dim, generator=generator)


def _is_connected(adjacency: torch.Tensor) -> bool:
    num_agents = adjacency.shape[0]
    visited = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        neighbours = torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
        for neighbour in neighbours:
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)
    return len(visited) == num_agents


def _hop_neighbourhood(adjacency: torch.Tensor, start: int, hops: int) -> List[int]:
    visited = {start}
    frontier = {start}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            neighbours = torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
            next_frontier.update(neighbours)
        next_frontier.difference_update(visited)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return sorted(visited)


def hop_union_ids(scene: FusionScene, agent_idx: int, hops: int) -> torch.Tensor:
    reachable_agents = _hop_neighbourhood(scene.communication_adjacency, agent_idx, hops)
    mask = scene.observation_mask[reachable_agents].any(dim=0)
    return torch.nonzero(mask, as_tuple=False).flatten()


def hop_union_set(scene: FusionScene, agent_idx: int, hops: int) -> torch.Tensor:
    return scene.object_features[hop_union_ids(scene, agent_idx, hops)]


def sample_fusion_scene(
    num_agents: int = 7,
    num_objects: int = 10,
    communication_radius: float = 0.58,
    observation_radius: float = 0.36,
    max_tries: int = 512,
    generator: torch.Generator | None = None,
) -> FusionScene:
    for _ in range(max_tries):
        if generator is None:
            agent_positions = torch.rand(num_agents, 2)
            object_positions = torch.rand(num_objects, 2)
            colours = torch.rand(num_objects, 3)
            radii = 0.03 + 0.07 * torch.rand(num_objects, 1)
        else:
            agent_positions = torch.rand(num_agents, 2, generator=generator)
            object_positions = torch.rand(num_objects, 2, generator=generator)
            colours = torch.rand(num_objects, 3, generator=generator)
            radii = 0.03 + 0.07 * torch.rand(num_objects, 1, generator=generator)

        communication_distances = torch.cdist(agent_positions, agent_positions)
        communication_adjacency = communication_distances <= communication_radius
        communication_adjacency.fill_diagonal_(True)

        observation_distances = torch.cdist(agent_positions, object_positions)
        observation_mask = observation_distances <= observation_radius

        if not _is_connected(communication_adjacency):
            continue
        if not observation_mask.any(dim=0).all():
            continue
        if not observation_mask.any(dim=1).all():
            continue
        duplicate_count = int((observation_mask.sum(dim=0) > 1).sum().item())
        if duplicate_count == 0:
            continue

        object_features = torch.cat([object_positions, colours, radii], dim=1)
        return FusionScene(
            agent_positions=agent_positions,
            object_features=object_features,
            communication_adjacency=communication_adjacency,
            observation_mask=observation_mask,
        )

    raise RuntimeError("Unable to sample a connected fusion scene with observable objects.")


if __name__ == "__main__":
    torch.manual_seed(7)
    scene = sample_fusion_scene()
    print("Agents:", scene.num_agents)
    print("Objects:", scene.num_objects)
    print("Communication adjacency shape:", tuple(scene.communication_adjacency.shape))
    print("Observation mask shape:", tuple(scene.observation_mask.shape))
    for agent_idx in range(scene.num_agents):
        local_set = scene.local_observations(agent_idx)
        print(f"Agent {agent_idx}: local observations = {tuple(local_set.shape)}")
