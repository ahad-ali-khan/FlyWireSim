"""Phase 9 recurrent, attention-based PPO actor-critic architecture.

This architecture is inspired by Global Workspace Theory as a computational design choice. It does not implement or claim to implement consciousness or awareness.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Normal


COGNITIVE_ARCH_ENABLED = True
SELF_MODEL_DIM = 64
ATTENTION_DIM = 64
BELIEF_GRU_SIZE = 64
WORKSPACE_DIM = 64
SELF_MODEL_LOSS_WEIGHT = 0.1
BELIEF_LOSS_WEIGHT = 0.1
SPECIALIST_NAMES = ("sensory", "proprioception", "self_model", "belief_state")


@dataclass
class CognitiveOutput:
    mean: torch.Tensor
    value: torch.Tensor
    hidden: torch.Tensor
    predicted_next_state: torch.Tensor
    belief_state: torch.Tensor
    attention_weights: torch.Tensor
    workspace_weights: torch.Tensor


def _mlp(input_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(),
                         nn.Linear(64, WORKSPACE_DIM), nn.Tanh())


class ObservationAttention(nn.Module):
    """One learned self-attention head over named scalar observation tokens."""

    def __init__(self, observation_dim: int):
        super().__init__()
        self.scalar_embedding = nn.Linear(1, ATTENTION_DIM)
        self.feature_embedding = nn.Parameter(torch.zeros(1, observation_dim, ATTENTION_DIM))
        nn.init.normal_(self.feature_embedding, std=0.02)
        self.attention = nn.MultiheadAttention(ATTENTION_DIM, 1, batch_first=True)
        self.norm = nn.LayerNorm(ATTENTION_DIM)

    def forward(self, observation):
        tokens = self.scalar_embedding(observation.unsqueeze(-1)) + self.feature_embedding
        attended, matrix = self.attention(tokens, tokens, tokens, need_weights=True,
                                          average_attn_weights=False)
        component_weights = matrix.mean(dim=1).mean(dim=1)
        return self.norm(attended).mean(dim=1), component_weights


class CognitiveActorCritic(nn.Module):
    """Four specialist heads compete for a single 64-value motor workspace."""

    def __init__(self, role: str, observation_dim: int, action_dim: int = 3):
        super().__init__()
        if role not in ("bat", "moth"):
            raise ValueError("role must be bat or moth")
        expected = 19 if role == "bat" else 23
        if observation_dim != expected:
            raise ValueError(f"Phase 9 expects {expected} {role} observation values, got {observation_dim}")
        self.role = role
        self.observation_dim = observation_dim
        self.attention = ObservationAttention(observation_dim)
        self.self_model = nn.Sequential(nn.Linear(6, SELF_MODEL_DIM), nn.Tanh(),
                                        nn.Linear(SELF_MODEL_DIM, SELF_MODEL_DIM), nn.Tanh())
        self.self_model_predictor = nn.Linear(SELF_MODEL_DIM, 6)
        self.belief_gru = nn.GRU(ATTENTION_DIM, BELIEF_GRU_SIZE, batch_first=True)
        self.belief_predictor = nn.Linear(BELIEF_GRU_SIZE, 6)
        sensory_dim = 9 if role == "bat" else 13
        self.sensory_specialist = _mlp(sensory_dim)
        self.proprioception_specialist = _mlp(9)
        self.self_model_specialist = _mlp(SELF_MODEL_DIM)
        self.belief_specialist = _mlp(BELIEF_GRU_SIZE)
        self.workspace_score = nn.Linear(WORKSPACE_DIM, 1)
        self.actor = nn.Sequential(nn.Linear(WORKSPACE_DIM, 64), nn.Tanh(),
                                   nn.Linear(64, 64), nn.Tanh(),
                                   nn.Linear(64, action_dim))
        self.critic = nn.Sequential(nn.Linear(WORKSPACE_DIM, 64), nn.Tanh(),
                                    nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.log_standard_deviation = nn.Parameter(torch.zeros(action_dim))

    def initial_hidden(self, batch_size=1, *, device=None):
        return torch.zeros(1, batch_size, BELIEF_GRU_SIZE,
                           device=device or next(self.parameters()).device)

    def forward_sequence(self, observations, hidden=None):
        """Run [batch,time,features], preserving recurrent state across time."""
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError(f"expected [batch,time,{self.observation_dim}] observations")
        batch, ticks, _ = observations.shape
        flat = observations.reshape(batch * ticks, self.observation_dim)
        attended, attention_weights = self.attention(flat)
        attended = attended.reshape(batch, ticks, ATTENTION_DIM)
        belief_hidden, next_hidden = self.belief_gru(attended, hidden)
        own_state = observations[..., :6]
        self_hidden = self.self_model(own_state.reshape(batch * ticks, 6)).reshape(
            batch, ticks, SELF_MODEL_DIM)
        predicted = self.self_model_predictor(self_hidden)
        belief = self.belief_predictor(belief_hidden)
        if self.role == "bat":
            sensory = torch.cat((observations[..., 9:17], observations[..., 18:19]), dim=-1)
        else:
            sensory = torch.cat((observations[..., 9:21], observations[..., 22:23]), dim=-1)
        sensory_head = self.sensory_specialist(sensory.reshape(batch * ticks, -1))
        proprio_head = self.proprioception_specialist(observations[..., :9].reshape(batch * ticks, 9))
        self_head = self.self_model_specialist(self_hidden.reshape(batch * ticks, SELF_MODEL_DIM))
        belief_head = self.belief_specialist(belief_hidden.reshape(batch * ticks, BELIEF_GRU_SIZE))
        heads = torch.stack((sensory_head, proprio_head, self_head, belief_head), dim=-2)
        scores = self.workspace_score(heads).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        workspace = torch.sum(heads * weights.unsqueeze(-1), dim=-2)
        workspace = workspace.reshape(batch, ticks, WORKSPACE_DIM)
        mean = self.actor(workspace)
        value = self.critic(workspace).squeeze(-1)
        attention_weights = attention_weights.reshape(batch, ticks, self.observation_dim)
        workspace_weights = weights.reshape(batch, ticks, len(SPECIALIST_NAMES))
        return CognitiveOutput(mean, value, next_hidden, predicted, belief,
                               attention_weights, workspace_weights)

    def act(self, observation, hidden=None, deterministic=False):
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        output = self.forward_sequence(observation.unsqueeze(1), hidden)
        mean = output.mean[:, 0]
        distribution = Normal(mean, self.log_standard_deviation.exp().expand_as(mean))
        action = mean if deterministic else distribution.sample()
        log_probability = distribution.log_prob(action).sum(dim=-1)
        return action, log_probability, output.value[:, 0], output

    def evaluate_sequence(self, observations, hidden=None):
        output = self.forward_sequence(observations, hidden)
        distribution = Normal(output.mean,
                              self.log_standard_deviation.exp().expand_as(output.mean))
        return output, distribution


def added_parameters(role: str, observation_dim: int) -> dict[str, int]:
    """Exact Phase 9 module parameter count versus a 64x64 flat PPO MLP."""
    cognitive = CognitiveActorCritic(role, observation_dim)
    flat_policy = nn.Sequential(nn.Linear(observation_dim, 64), nn.Tanh(),
                                nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
    flat_value = nn.Sequential(nn.Linear(observation_dim, 64), nn.Tanh(),
                               nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    flat_total = sum(p.numel() for p in flat_policy.parameters()) + \
        sum(p.numel() for p in flat_value.parameters()) + 3
    cognitive_total = sum(p.numel() for p in cognitive.parameters())
    return {"role": role, "flat_mlp_parameters": flat_total,
            "cognitive_parameters": cognitive_total,
            "added_parameters": cognitive_total - flat_total,
            "workspace_gate_parameters": sum(p.numel() for p in cognitive.workspace_score.parameters())}


def self_check():
    for role, size in (("bat", 19), ("moth", 23)):
        model = CognitiveActorCritic(role, size)
        obs = torch.zeros(1, 5, size)
        output = model.forward_sequence(obs)
        assert output.mean.shape == (1, 5, 3)
        assert output.value.shape == (1, 5)
        assert output.predicted_next_state.shape == (1, 5, 6)
        assert output.belief_state.shape == (1, 5, 6)
        assert output.attention_weights.shape == (1, 5, size)
        assert output.workspace_weights.shape == (1, 5, 4)
        assert torch.allclose(output.workspace_weights.sum(-1), torch.ones(1, 5))
        loss = output.mean.square().mean() + output.predicted_next_state.square().mean()
        loss.backward()
        assert model.workspace_score.weight.grad is not None
        assert model.self_model[0].weight.grad is not None


if __name__ == "__main__":
    self_check()
    for agent, dim in (("bat", 19), ("moth", 23)):
        print(added_parameters(agent, dim))
