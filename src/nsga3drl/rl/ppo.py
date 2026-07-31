from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import OperatorSelectionEnv


class PopulationActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_size: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.policy = nn.Linear(hidden_size, action_count)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        hidden = self.encoder(observations)
        return Categorical(logits=self.policy(hidden)), self.value(hidden).squeeze(-1)


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    epochs: int = 4
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5


class PPOTrainer:
    def __init__(
        self,
        observation_size: int,
        action_count: int,
        config: PPOConfig | None = None,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.seed = seed
        self.set_seed(seed)
        self.model = PopulationActorCritic(observation_size, action_count).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    @staticmethod
    def set_seed(seed: int) -> None:
        """Seed every RNG used by this CPU/GPU PPO implementation."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "observation_size": self.model.encoder[0].in_features,
            "action_count": self.model.policy.out_features,
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PPOTrainer":
        checkpoint = torch.load(path, map_location=device)
        trainer = cls(
            checkpoint["observation_size"],
            checkpoint["action_count"],
            PPOConfig(**checkpoint["config"]),
            device=device,
            seed=checkpoint.get("seed", 0),
        )
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return trainer

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> tuple[int, float, float]:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        distribution, value = self.model(tensor)
        action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    @torch.no_grad()
    def greedy_action(self, observation: np.ndarray) -> int:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        distribution, _ = self.model(tensor)
        return int(distribution.probs.argmax(dim=-1).item())

    def collect(self, environments: list[OperatorSelectionEnv]) -> dict[str, np.ndarray]:
        data = {key: [] for key in ("observations", "actions", "log_probs", "rewards", "values", "dones")}
        for index, environment in enumerate(environments):
            observation = environment.reset(seed=environment.seed + index)
            done = False
            while not done:
                action, log_prob, value = self.act(observation)
                next_observation, reward, done, _ = environment.step(action)
                data["observations"].append(observation)
                data["actions"].append(action)
                data["log_probs"].append(log_prob)
                data["rewards"].append(reward)
                data["values"].append(value)
                data["dones"].append(done)
                observation = next_observation
        return {key: np.asarray(value) for key, value in data.items()}

    def update(self, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        rewards = rollout["rewards"].astype(np.float32)
        values = rollout["values"].astype(np.float32)
        dones = rollout["dones"].astype(bool)
        advantages = np.zeros_like(rewards)
        next_advantage = 0.0
        next_value = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            if dones[index]:
                next_advantage, next_value = 0.0, 0.0
            delta = rewards[index] + self.config.gamma * next_value - values[index]
            next_advantage = delta + self.config.gamma * self.config.gae_lambda * next_advantage
            advantages[index] = next_advantage
            next_value = values[index]
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        observations = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        losses = []
        policy_losses = []
        value_losses = []
        entropies = []
        for _ in range(self.config.epochs):
            distribution, predicted_values = self.model(observations)
            ratio = torch.exp(distribution.log_prob(actions) - old_log_probs)
            policy_loss = -torch.minimum(ratio * advantages_t, torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages_t).mean()
            value_loss = nn.functional.mse_loss(predicted_values, returns_t)
            loss = policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * distribution.entropy().mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.item()))
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(distribution.entropy().mean().item()))
        return {
            "loss": float(np.mean(losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "mean_reward": float(rewards.mean()),
        }
