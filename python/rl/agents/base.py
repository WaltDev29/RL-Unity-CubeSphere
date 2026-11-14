"""공통 컴포넌트: 리플레이 버퍼, MLP, Agent 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda 이지만 CUDA 를 사용할 수 없습니다")
    return torch.device(name)


def mlp(in_dim: int, out_dim: int, hidden: int, layers: int = 2) -> nn.Sequential:
    dims = [in_dim] + [hidden] * layers
    mods: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        mods += [nn.Linear(a, b), nn.ReLU()]
    mods.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*mods)


class ReplayBuffer:
    """고정 크기 원형 버퍼. 소규모 데이터이므로 학습 장치(GPU)에 그대로 상주시켜 전송 비용을 없앤다."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity, self.device = capacity, device
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.act = torch.zeros((capacity, act_dim), device=device)
        self.rew = torch.zeros((capacity, 1), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.done = torch.zeros((capacity, 1), device=device)   # 종료(성공/추락)만 1, truncation 은 0 -> 부트스트랩 유지
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2, done: bool) -> None:
        i = self.ptr
        self.obs[i] = torch.as_tensor(o, device=self.device)
        self.act[i] = torch.as_tensor(a, device=self.device)
        self.rew[i] = float(r)
        self.next_obs[i] = torch.as_tensor(o2, device=self.device)
        self.done[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {"obs": self.obs[idx], "act": self.act[idx], "rew": self.rew[idx],
                "next_obs": self.next_obs[idx], "done": self.done[idx]}

    def state_dict(self) -> dict:
        n = self.size
        return {"ptr": self.ptr, "size": n,
                **{k: getattr(self, k)[:n].cpu().numpy() for k in ("obs", "act", "rew", "next_obs", "done")}}

    def load_state_dict(self, s: dict) -> None:
        n = int(s["size"])
        for k in ("obs", "act", "rew", "next_obs", "done"):
            getattr(self, k)[:n] = torch.as_tensor(s[k], device=self.device)
        self.size, self.ptr = n, int(s["ptr"]) % self.capacity


class Agent(ABC):
    """모든 알고리즘이 따르는 인터페이스. 새 알고리즘은 이 클래스를 구현하고 registry 에 등록한다."""

    device: torch.device

    @abstractmethod
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray: ...

    @abstractmethod
    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """gradient update 1회. 로그용 loss 텐서(detach)를 반환한다."""

    @abstractmethod
    def state_dict(self) -> dict: ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None: ...

    @abstractmethod
    def policy_state_dict(self) -> dict:
        """추론에 필요한 최소 가중치 (Best/Final 모델 저장용)."""

    @abstractmethod
    def load_policy_state_dict(self, state: dict) -> None: ...

