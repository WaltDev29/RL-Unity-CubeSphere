"""학습 오케스트레이션: 에피소드 루프, 일시정지/중지, 체크포인트, 평가, Best/Final 모델, 오류 복구."""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from ..agents import ReplayBuffer, make_agent, resolve_device
from ..config import Config
from ..envs.session import EnvSession
from ..envs.unity_env import EnvConnectionError, EnvError
from ..utils.checkpoint import load_torch, rng_state, save_torch, seed_everything, set_rng_state
from ..utils.plotting import save_plots
from ..utils.runio import RunDir
from .rollout import SpeedSync, run_episode

log = logging.getLogger(__name__)


class StopRequested(Exception):
    """제어 파일에 stop 이 기록되었을 때 에피소드 루프를 빠져나오기 위한 신호."""


class Trainer:
    def __init__(self, cfg: Config, run: RunDir, resume: bool = False):
        self.cfg, self.run = cfg, run
        self.device = resolve_device(cfg.device)
        self.session = EnvSession(cfg, cfg.train_steps_per_sec)
        self.speed_sync = SpeedSync(cfg, cfg.train_steps_per_sec)
        self.env = None
        self.agent = None
        self.buffer = None
        self.episode = 0
        self.total_steps = 0
        self.elapsed_before = 0.0
        self.best_key = (-math.inf, -math.inf)       # (평가 성공률, 평가 평균 reward)
        self.rewards: deque[float] = deque(maxlen=cfg.ma_window)
        self.successes: deque[float] = deque(maxlen=cfg.ma_window)
        self.resume = resume
        self._t0 = time.time()
        self._last_status = 0.0
        self._state = "starting"
        self._last_reward = float("nan")

    # ---------------------------------------------------------------- 진입점
    def train(self) -> int:
        """0: 정상 종료(완료/사용자 중지), 1: 오류 종료."""
        seed_everything(self.cfg.seed)
        exit_code = 0
        try:
            self._status("starting", "Unity 연결 중")
            self.env = self.session.connect()
            self.agent = make_agent(self.cfg.algorithm, self.env.observation_space.shape[0],
                                    self.env.action_space.shape[0], self.cfg, self.device)
            self.buffer = ReplayBuffer(self.cfg.buffer_size, self.env.observation_space.shape[0],
                                       self.env.action_space.shape[0], self.device)
            if self.resume:
                self._load_latest()
            log.info("학습 시작: algo=%s device=%s episode=%d/%d", self.cfg.algorithm, self.device,
                     self.episode, self.cfg.total_episodes)
            self.run.write_control("run")
            self._loop()
            self._state = "finished"
            self._finalize("학습 완료")
        except StopRequested:
            self._state = "stopped"
            self._finalize("사용자 중지")
        except KeyboardInterrupt:
            self._state = "stopped"
            self._finalize("Ctrl+C 중지")
        except Exception as e:   # 어떤 오류든 체크포인트를 남기고 종료
            log.exception("학습 중 치명적 오류")
            self._state = "error"
            exit_code = 1
            self._finalize(f"오류: {e}")
        finally:
            self.session.close()
        return exit_code

    # ---------------------------------------------------------------- 메인 루프
    def _loop(self) -> None:
        cfg = self.cfg
        while self.episode < cfg.total_episodes:
            try:
                self._state = "running"
                res, losses = self._train_episode()
            except EnvConnectionError as e:
                log.warning("통신 오류로 에피소드 폐기: %s", e)
                self._recover()
                continue
            self.episode += 1
            self._record(res, losses)
            if self.episode % cfg.eval_every == 0:
                self._evaluate()
            if self.episode % cfg.checkpoint_every == 0:
                self._save_checkpoint(numbered=True)

    def _train_episode(self):
        cfg = self.cfg
        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}

        def select(obs):
            if self.total_steps < cfg.warmup_steps or self.buffer.size < cfg.batch_size:
                return np.random.uniform(-1, 1, self.env.action_space.shape).astype(np.float32)
            return self.agent.select_action(obs)

        def on_transition(o, a, r, o2, terminated, truncated):
            # truncation 은 시간 제한일 뿐이므로 done=0 으로 저장해 부트스트랩을 유지한다
            self.buffer.add(o, a, r, o2, terminated)
            self.total_steps += 1
            if (self.total_steps >= cfg.warmup_steps and self.buffer.size >= cfg.batch_size
                    and self.total_steps % cfg.update_every == 0):
                for _ in range(cfg.updates_per_cycle):
                    out = self.agent.update(self.buffer.sample(cfg.batch_size))
                    for k, v in out.items():
                        sums[k] = sums[k] + v if k in sums else v.clone()
                        counts[k] = counts.get(k, 0) + 1

        res = run_episode(self.env, select, on_transition, self._step_hook)
        losses = {k: (v / counts[k]).item() for k, v in sums.items()}   # 에피소드당 GPU 동기화 1회
        return res, losses

    def _step_hook(self) -> None:
        ctrl = self.run.read_control()
        self.speed_sync.apply_if_changed(self.env, self.run)
        if ctrl == "stop":
            raise StopRequested()
        if ctrl == "pause":
            self._state = "paused"
            self._status("paused", "일시정지")
            while True:
                time.sleep(0.2)
                ctrl = self.run.read_control()
                if ctrl == "stop":
                    raise StopRequested()
                if ctrl == "run":
                    break
            self._state = "running"
        if time.time() - self._last_status > 0.5:
            self._status(self._state)

    # ---------------------------------------------------------------- 기록
    def _elapsed(self) -> float:
        return self.elapsed_before + time.time() - self._t0

    def _record(self, res, losses: dict) -> None:
        self.rewards.append(res.reward)
        self.successes.append(1.0 if res.success else 0.0)
        self._last_reward = res.reward
        nan = float("nan")
        row = {
            "episode": self.episode, "total_steps": self.total_steps,
            "reward": res.reward, "reward_ma": float(np.mean(self.rewards)),
            "length": res.length, "success": int(res.success), "fell": int(res.fell),
            "success_rate": float(np.mean(self.successes)),
            "critic_loss": losses.get("critic_loss", nan), "actor_loss": losses.get("actor_loss", nan),
            "alpha": losses.get("alpha", nan), "elapsed_sec": round(self._elapsed(), 2),
        }
        self.run.log_episode(row)
        self._status(self._state)
        if self.episode % 10 == 0 or self.episode == 1:
            log.info("ep %d | R %.2f | MA %.2f | succ %.2f | len %d | Lq %.4f",
                     self.episode, res.reward, row["reward_ma"], row["success_rate"], res.length,
                     row["critic_loss"])

    def _status(self, state: str, message: str = "") -> None:
        self._last_status = time.time()
        self.run.write_status(
            state=state, message=message, episode=self.episode, total_episodes=self.cfg.total_episodes,
            total_steps=self.total_steps, elapsed_sec=round(self._elapsed(), 2),
            unity_connected=bool(self.env and self.env.connected), algorithm=self.cfg.algorithm,
            device=str(self.device), last_reward=self._last_reward, run_dir=str(self.run.root),
            updated=time.time(), mode="train",
        )

    # ---------------------------------------------------------------- 평가 / 모델 저장
    def _evaluate(self) -> None:
        """결정론적 정책으로 평가하고, 성공률 -> 평균 reward 순으로 비교해 Best Model 을 갱신한다."""
        self._state = "evaluating"
        results = []
        try:
            for _ in range(self.cfg.eval_episodes):
                results.append(run_episode(self.env, lambda o: self.agent.select_action(o, True),
                                           step_hook=self._step_hook))
        except EnvConnectionError as e:
            log.warning("평가 중 통신 오류: %s", e)
            self._recover()
            return
        finally:
            self._state = "running"
        mean_r = float(np.mean([r.reward for r in results]))
        succ = float(np.mean([r.success for r in results]))
        self.run.log_eval({"episode": self.episode, "mean_reward": mean_r, "success_rate": succ,
                           "mean_length": float(np.mean([r.length for r in results]))})
        key = (succ, mean_r)
        log.info("평가 ep %d: reward %.2f success %.2f", self.episode, mean_r, succ)
        if key > self.best_key:
            self.best_key = key
            save_torch({"policy": self.agent.policy_state_dict(), "episode": self.episode,
                        "eval_success_rate": succ, "eval_mean_reward": mean_r,
                        "algorithm": self.cfg.algorithm}, self.run.model_dir / "best.pt")
            log.info("Best Model 갱신 (episode %d)", self.episode)

    def _save_checkpoint(self, numbered: bool) -> None:
        core = {"agent": self.agent.state_dict(), "episode": self.episode, "total_steps": self.total_steps,
                "elapsed": self._elapsed(), "best_key": self.best_key, "rewards": list(self.rewards),
                "successes": list(self.successes), "algorithm": self.cfg.algorithm}
        if numbered:   # 번호 체크포인트는 용량 때문에 버퍼 제외
            save_torch(core, self.run.ckpt_dir / f"ckpt_ep{self.episode:06d}.pt")
        # latest 는 재개용이므로 버퍼/RNG 까지 포함
        save_torch({**core, "buffer": self.buffer.state_dict(), "rng": rng_state()},
                   self.run.ckpt_dir / "latest.pt")

    def _load_latest(self) -> None:
        path = self.run.ckpt_dir / "latest.pt"
        if not path.exists():
            raise FileNotFoundError(f"재개할 체크포인트가 없습니다: {path}")
        s = load_torch(path, self.device)
        if s["algorithm"] != self.cfg.algorithm:
            raise ValueError(f"체크포인트 알고리즘({s['algorithm']})과 설정({self.cfg.algorithm})이 다릅니다")
        self.agent.load_state_dict(s["agent"])
        self.buffer.load_state_dict(s["buffer"])
        set_rng_state(s["rng"])
        self.episode, self.total_steps = int(s["episode"]), int(s["total_steps"])
        self.elapsed_before, self.best_key = float(s["elapsed"]), tuple(s["best_key"])
        self.rewards.extend(s["rewards"])
        self.successes.extend(s["successes"])
        log.info("체크포인트 재개: episode=%d steps=%d", self.episode, self.total_steps)

    def _recover(self) -> None:
        """Unity 연결 복구를 RECONNECT_RETRIES 회 시도. 실패하면 예외를 던져 상위에서 체크포인트를 남긴다."""
        self._state = "reconnecting"
        for i in range(1, self.cfg.reconnect_retries + 1):
            self._status("reconnecting", f"재연결 시도 {i}/{self.cfg.reconnect_retries}")
            try:
                self.session.reconnect()
                log.info("재연결 성공")
                return
            except (EnvConnectionError, FileNotFoundError, EnvError) as e:
                log.warning("재연결 실패(%d): %s", i, e)
                time.sleep(min(2.0 * i, 10.0))
        raise EnvConnectionError("재연결 한도 초과")

    def _finalize(self, message: str) -> None:
        """종료 경로 공통 처리: 가능한 모든 산출물을 보존한다."""
        log.info("종료 처리: %s", message)
        if self.agent is not None:
            try:
                self._save_checkpoint(numbered=False)
                save_torch({"policy": self.agent.policy_state_dict(), "episode": self.episode,
                            "algorithm": self.cfg.algorithm}, self.run.model_dir / "final.pt")
            except Exception:
                log.exception("최종 모델/체크포인트 저장 실패")
        try:
            save_plots(self.run.root, self.run.plot_dir)
        except Exception:
            log.exception("그래프 저장 실패")
        self._status(self._state, message)

