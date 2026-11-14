"""에피소드 1회 실행 로직 (학습/평가 공용)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class SpeedSync:
    """제어 파일의 속도 값이 바뀌면 Unity 에 config 로 재전송한다 (학습 중 실시간 속도 조절)."""

    def __init__(self, cfg, initial_sps: float):
        self.cfg, self.applied = cfg, initial_sps

    def apply_if_changed(self, env, run) -> None:
        sps = run.speed
        if sps is not None and sps != self.applied:
            env.configure(self.cfg.env_params(float(sps)))
            self.applied = sps


@dataclass
class EpisodeResult:
    reward: float
    length: int
    success: bool
    fell: bool


def run_episode(env,
                select_action: Callable[[np.ndarray], np.ndarray],
                on_transition: Callable[[np.ndarray, np.ndarray, float, np.ndarray, bool, bool], None] | None = None,
                step_hook: Callable[[], None] | None = None,
                seed: int | None = None) -> EpisodeResult:
    """step_hook 은 매 step 호출되며 예외(일시정지 대기/중지 요청 등)를 통해 흐름을 제어한다."""
    obs, _ = env.reset(seed)
    total, length = 0.0, 0
    info: dict = {}
    while True:
        if step_hook:
            step_hook()
        action = select_action(obs)
        obs2, reward, terminated, truncated, info = env.step(action)
        if on_transition:
            on_transition(obs, action, reward, obs2, terminated, truncated)
        obs = obs2
        total += reward
        length += 1
        if terminated or truncated:
            break
    return EpisodeResult(total, length, bool(info.get("success")), bool(info.get("fell")))

