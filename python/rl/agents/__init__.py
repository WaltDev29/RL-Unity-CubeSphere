"""알고리즘 레지스트리: ALGORITHM 값 -> Agent 클래스. 새 알고리즘은 여기에 한 줄만 추가한다."""
from __future__ import annotations

from .base import Agent, ReplayBuffer, resolve_device
from .sac import SAC
from .td3 import TD3

REGISTRY: dict[str, type[Agent]] = {"sac": SAC, "td3": TD3}


def make_agent(name: str, obs_dim: int, act_dim: int, cfg, device) -> Agent:
    if name not in REGISTRY:
        raise ValueError(f"알 수 없는 알고리즘 '{name}' (사용 가능: {list(REGISTRY)})")
    return REGISTRY[name](obs_dim, act_dim, cfg, device)


__all__ = ["Agent", "ReplayBuffer", "resolve_device", "make_agent", "REGISTRY"]

