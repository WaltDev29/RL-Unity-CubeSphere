"""학습된 모델 평가 (학습과 완전히 분리된 모드: 업데이트/버퍼/탐색 노이즈 없음)."""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..agents import make_agent, resolve_device
from ..config import Config
from ..envs.session import EnvSession
from ..envs.unity_env import EnvConnectionError
from ..utils.checkpoint import load_torch, save_torch, seed_everything
from ..utils.plotting import save_plots
from ..utils.runio import RunDir
from .rollout import SpeedSync, run_episode
from .trainer import StopRequested

log = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: Config, run: RunDir, model_path: Path, episodes: int):
        self.cfg, self.run, self.model_path, self.episodes = cfg, run, model_path, episodes
        self.session = EnvSession(cfg, cfg.eval_steps_per_sec)
        self.speed_sync = SpeedSync(cfg, cfg.eval_steps_per_sec)
        self.env = None
        self._t0 = time.time()
        self._state = "starting"

    def evaluate(self) -> int:
        seed_everything(self.cfg.seed + 1)
        code = 0
        results = []
        try:
            self._status("starting", "Unity 연결 중")
            self.env = self.session.connect()
            device = resolve_device(self.cfg.device)
            agent = make_agent(self.cfg.algorithm, self.env.observation_space.shape[0],
                               self.env.action_space.shape[0], self.cfg, device)
            blob = load_torch(self.model_path, device)
            # 모델 파일("policy") 과 체크포인트("agent") 모두 actor 가중치 키를 공유한다
            agent.load_policy_state_dict(blob.get("policy", blob.get("agent")))
            log.info("평가 시작: %s (%d episodes)", self.model_path, self.episodes)
            self.run.write_control("run")
            rewards, succs = deque(maxlen=self.cfg.ma_window), deque(maxlen=self.cfg.ma_window)
            self._state = "evaluating"
            for ep in range(1, self.episodes + 1):
                res = run_episode(self.env, lambda o: agent.select_action(o, True), step_hook=self._hook)
                results.append(res)
                rewards.append(res.reward)
                succs.append(float(res.success))
                self.run.log_episode({
                    "episode": ep, "total_steps": 0, "reward": res.reward, "reward_ma": float(np.mean(rewards)),
                    "length": res.length, "success": int(res.success), "fell": int(res.fell),
                    "success_rate": float(np.mean(succs)), "critic_loss": float("nan"),
                    "actor_loss": float("nan"), "alpha": float("nan"), "elapsed_sec": round(time.time() - self._t0, 2)})
                self._status("evaluating", episode=ep)
                log.info("eval ep %d: reward %.2f success=%s len=%d", ep, res.reward, res.success, res.length)
            self._state = "finished"
        except StopRequested:
            self._state = "stopped"
        except Exception:
            log.exception("평가 중 오류")
            self._state, code = "error", 1
        finally:
            self.session.close()
        if results:
            summary = {"episodes": len(results), "mean_reward": float(np.mean([r.reward for r in results])),
                       "success_rate": float(np.mean([r.success for r in results])),
                       "mean_length": float(np.mean([r.length for r in results])), "model": str(self.model_path)}
            (self.run.root / "eval_summary.json").write_text(
                __import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            log.info("평가 요약: %s", summary)
            try:
                save_plots(self.run.root, self.run.plot_dir)
            except Exception:
                log.exception("그래프 저장 실패")
        self._status(self._state)
        return code

    def _hook(self) -> None:
        ctrl = self.run.read_control()
        self.speed_sync.apply_if_changed(self.env, self.run)
        if ctrl == "stop":
            raise StopRequested()
        while ctrl == "pause":
            self._status("paused")
            time.sleep(0.2)
            ctrl = self.run.read_control()
            if ctrl == "stop":
                raise StopRequested()

    def _status(self, state: str, message: str = "", episode: int = 0) -> None:
        self.run.write_status(
            state=state, message=message, episode=episode, total_episodes=self.episodes, total_steps=0,
            elapsed_sec=round(time.time() - self._t0, 2), unity_connected=bool(self.env and self.env.connected),
            algorithm=self.cfg.algorithm, device=self.cfg.device, last_reward=float("nan"),
            run_dir=str(self.run.root), updated=time.time(), mode="eval")

