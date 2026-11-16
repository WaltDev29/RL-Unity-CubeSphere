"""`.env` 로딩. 코드에 기본값을 두지 않고, 누락/형식 오류는 즉시 명확한 예외로 알린다."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, asdict
from pathlib import Path

from dotenv import dotenv_values

# python/rl/config.py -> 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    pass


@dataclass
class Config:
    # 학습
    total_episodes: int
    max_steps_per_episode: int
    seed: int
    algorithm: str
    device: str
    # 하이퍼파라미터
    learning_rate: float
    gamma: float
    tau: float
    batch_size: int
    buffer_size: int
    warmup_steps: int
    hidden_units: int
    update_every: int
    updates_per_cycle: int
    sac_init_alpha: float
    sac_target_entropy: float
    td3_exploration_noise: float
    td3_policy_noise: float
    td3_noise_clip: float
    td3_policy_delay: int
    # 환경
    floor_size: float
    min_spawn_distance: float
    spawn_margin: float
    max_force: float
    max_speed: float
    fall_y: float
    decision_interval: int
    fixed_dt: float
    sphere_mass: float
    sphere_drag: float
    sphere_angular_drag: float
    train_steps_per_sec: float
    eval_steps_per_sec: float
    # reward
    w_approach: float
    r_success: float
    r_fall: float
    r_wall_hit: float
    c_step: float
    c_action: float
    wall_width: float
    wall_thickness: float
    # 연결
    unity_host: str
    unity_port: int
    connect_timeout: float
    step_timeout: float
    reconnect_retries: int
    unity_exe: str
    unity_autolaunch: bool
    unity_headless: bool
    # 저장
    runs_dir: str
    checkpoint_every: int
    eval_every: int
    eval_episodes: int
    ma_window: int

    def to_dict(self) -> dict:
        return asdict(self)

    def env_params(self, steps_per_sec: float) -> dict:
        """Unity 의 EnvParams 필드명과 1:1 대응하는 dict (config 명령 페이로드)."""
        return {
            "floor_size": self.floor_size,
            "min_spawn_distance": self.min_spawn_distance,
            "spawn_margin": self.spawn_margin,
            "max_force": self.max_force,
            "max_speed": self.max_speed,
            "fall_y": self.fall_y,
            "decision_interval": self.decision_interval,
            "max_steps": self.max_steps_per_episode,
            "fixed_dt": self.fixed_dt,
            "sphere_mass": self.sphere_mass,
            "sphere_drag": self.sphere_drag,
            "sphere_angular_drag": self.sphere_angular_drag,
            "w_approach": self.w_approach,
            "r_success": self.r_success,
            "r_fall": self.r_fall,
            "r_wall_hit": self.r_wall_hit,
            "c_step": self.c_step,
            "c_action": self.c_action,
            "wall_width": self.wall_width,
            "wall_thickness": self.wall_thickness,
            "max_steps_per_sec": steps_per_sec,
        }

    def resolve_path(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _cast(name: str, raw: str, typ):
    try:
        if typ is bool:
            if raw.strip().lower() in ("1", "true", "yes", "on"):
                return True
            if raw.strip().lower() in ("0", "false", "no", "off"):
                return False
            raise ValueError("bool 은 1/0/true/false 만 허용")
        return typ(raw.strip())
    except ValueError as e:
        raise ConfigError(f".env 값 형식 오류: {name.upper()}='{raw}' ({typ.__name__} 필요): {e}") from e


def load_config(env_path: str | Path | None = None, overrides: dict | None = None) -> Config:
    path = Path(env_path) if env_path else PROJECT_ROOT / ".env"
    if not path.exists():
        raise ConfigError(f".env 파일이 없습니다: {path} (.env.example 을 복사하세요)")
    values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    # 프로세스 환경 변수가 있으면 .env 보다 우선 (GUI/CLI 가 실행 시 덮어쓸 수 있도록)
    for k in list(values):
        if k in os.environ:
            values[k] = os.environ[k]

    kwargs, missing = {}, []
    for f in fields(Config):
        key = f.name.upper()
        if key not in values or values[key] == "":
            missing.append(key)
            continue
        kwargs[f.name] = _cast(f.name, values[key], f.type if not isinstance(f.type, str) else eval(f.type))
    if missing:
        raise ConfigError(f".env 에 누락된 키: {', '.join(missing)}")
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    cfg = Config(**kwargs)
    _validate(cfg)
    return cfg


def _validate(c: Config) -> None:
    errs = []
    if c.total_episodes < 1: errs.append("TOTAL_EPISODES >= 1")
    if c.max_steps_per_episode < 1: errs.append("MAX_STEPS_PER_EPISODE >= 1")
    if c.algorithm not in ("sac", "td3"): errs.append("ALGORITHM 은 sac|td3")
    if c.device not in ("auto", "cuda", "cpu"): errs.append("DEVICE 는 auto|cuda|cpu")
    if not (0 < c.gamma <= 1): errs.append("0 < GAMMA <= 1")
    if not (0 < c.tau <= 1): errs.append("0 < TAU <= 1")
    if c.batch_size < 1 or c.buffer_size < c.batch_size: errs.append("BUFFER_SIZE >= BATCH_SIZE >= 1")
    if c.update_every < 1 or c.updates_per_cycle < 1: errs.append("UPDATE_EVERY, UPDATES_PER_CYCLE >= 1")
    if c.ma_window < 1: errs.append("MA_WINDOW >= 1")
    if c.checkpoint_every < 1 or c.eval_every < 1 or c.eval_episodes < 1: errs.append("CHECKPOINT_EVERY/EVAL_* >= 1")
    if errs:
        raise ConfigError("설정 검증 실패: " + "; ".join(errs))

