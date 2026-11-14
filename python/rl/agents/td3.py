"""TD3 (Fujimoto et al. 2018): 비교 실험용 알고리즘."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent, mlp
from .sac import TwinQ


class DetActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden)

    def forward(self, obs):
        return torch.tanh(self.net(obs))


class TD3(Agent):
    def __init__(self, obs_dim: int, act_dim: int, cfg, device: torch.device):
        self.device = device
        self.gamma, self.tau = cfg.gamma, cfg.tau
        self.expl_noise, self.policy_noise = cfg.td3_exploration_noise, cfg.td3_policy_noise
        self.noise_clip, self.policy_delay = cfg.td3_noise_clip, cfg.td3_policy_delay
        self.act_dim = act_dim
        self.actor = DetActor(obs_dim, act_dim, cfg.hidden_units).to(device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = TwinQ(obs_dim, act_dim, cfg.hidden_units).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.updates = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).cpu().numpy()
        if not deterministic:
            a = a + np.random.normal(0.0, self.expl_noise, size=self.act_dim)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def update(self, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.updates += 1
        with torch.no_grad():
            noise = (torch.randn_like(b["act"]) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(b["next_obs"]) + noise).clamp(-1.0, 1.0)
            tq1, tq2 = self.critic_target(b["next_obs"], a2)
            target = b["rew"] + self.gamma * (1 - b["done"]) * torch.min(tq1, tq2)
        q1, q2 = self.critic(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        out = {"critic_loss": critic_loss.detach()}
        if self.updates % self.policy_delay == 0:
            actor_loss = -self.critic.q1(torch.cat([b["obs"], self.actor(b["obs"])], -1)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_target), (self.critic, self.critic_target)):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(1 - self.tau).add_(self.tau * p)
            out["actor_loss"] = actor_loss.detach()
        return out

    def policy_state_dict(self) -> dict:
        return {"actor": self.actor.state_dict()}

    def load_policy_state_dict(self, s: dict) -> None:
        self.actor.load_state_dict(s["actor"])

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(), "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "updates": self.updates}

    def load_state_dict(self, s: dict) -> None:
        self.actor.load_state_dict(s["actor"])
        self.actor_target.load_state_dict(s["actor_target"])
        self.critic.load_state_dict(s["critic"])
        self.critic_target.load_state_dict(s["critic_target"])
        self.actor_opt.load_state_dict(s["actor_opt"])
        self.critic_opt.load_state_dict(s["critic_opt"])
        self.updates = int(s["updates"])

