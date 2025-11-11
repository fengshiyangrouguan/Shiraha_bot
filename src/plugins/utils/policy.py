from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyOut:
    gate_logit: torch.Tensor   # [B]
    gate_prob: torch.Tensor    # [B]
    action_raw: torch.Tensor   # [B] pre-tanh
    action_tanh: torch.Tensor  # [B] in (-1,1)
    action_logp: torch.Tensor  # [B] with tanh jacobian
    value: torch.Tensor        # [B]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim), nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)


class GateHead(nn.Module):
    """Bernoulli gate. Outputs logit (float per sample)."""
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, h):
        logit = self.fc(h).squeeze(-1)
        prob = torch.sigmoid(logit)
        return logit, prob


class ActorHead(nn.Module):
    """Tanh-squashed Gaussian actor (1D position)."""
    def __init__(self, in_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
    def forward(self, h):
        mu = self.mu(h).squeeze(-1)  # raw (pre-tanh)
        std = torch.exp(self.log_std).clamp_min(1e-4)
        return mu, std


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.v = nn.Linear(in_dim, 1)
    def forward(self, h):
        return self.v(h).squeeze(-1)


class GateActorCritic(nn.Module):
    """Shared trunk -> gate + actor + critic."""
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.trunk = MLP(in_dim, hidden, hidden)
        self.gate = GateHead(hidden)
        self.actor = ActorHead(hidden)
        self.critic = Critic(hidden)

    def forward(self, state):
        h = self.trunk(state)
        gate_logit, gate_prob = self.gate(h)
        mu_raw, std = self.actor(h)
        value = self.critic(h)
        return gate_logit, gate_prob, mu_raw, std, value

