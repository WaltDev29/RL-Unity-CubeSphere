"""Soft Actor-Critic (자동 엔트로피 온도 조정). Haarnoja et al. 2018 (v2)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Agent, mlp

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # tanh 변환에 대한 log-prob 보정
        logp = (-0.5 * ((u - mu) / (std + 1e-8)).pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1, keepdim=True)
        return a, logp


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)


class SAC(Agent):
    def __init__(self, obs_dim: int, act_dim: int, cfg, device: torch.device):
        self.device = device
        self.gamma, self.tau = cfg.gamma, cfg.tau
        self.actor = GaussianActor(obs_dim, act_dim, cfg.hidden_units).to(device)
        self.critic = TwinQ(obs_dim, act_dim, cfg.hidden_units).to(device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(float(np.log(cfg.sac_init_alpha)), device=device, requires_grad=True)
        self.target_entropy = cfg.sac_target_entropy
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic)
        return a.squeeze(0).cpu().numpy()

    def update(self, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(b["next_obs"])
            tq1, tq2 = self.critic_target(b["next_obs"], a2)
            target = b["rew"] + self.gamma * (1 - b["done"]) * (torch.min(tq1, tq2) - alpha * logp2)
        q1, q2 = self.critic(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        a, logp = self.actor(b["obs"])
        q_pi = torch.min(*self.critic(b["obs"], a))
        actor_loss = (alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
        return {"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach(),
                "alpha": self.log_alpha.exp().detach()}

    def policy_state_dict(self) -> dict:
        return {"actor": self.actor.state_dict()}

    def load_policy_state_dict(self, s: dict) -> None:
        self.actor.load_state_dict(s["actor"])

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict()}

    def load_state_dict(self, s: dict) -> None:
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.critic_target.load_state_dict(s["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(s["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(s["actor_opt"])
        self.critic_opt.load_state_dict(s["critic_opt"])
        self.alpha_opt.load_state_dict(s["alpha_opt"])

