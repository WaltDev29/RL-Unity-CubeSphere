"""환경 연결 생성 (필요 시 Unity 빌드 자동 실행)."""
from __future__ import annotations

import logging
import subprocess
import time

from ..config import Config
from .unity_env import EnvConnectionError, UnityEnv, launch_unity

log = logging.getLogger(__name__)


class EnvSession:
    """UnityEnv 와 (자동 실행했다면) Unity 프로세스의 수명주기를 함께 관리한다."""

    def __init__(self, cfg: Config, steps_per_sec: float):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.env = UnityEnv(cfg.unity_host, cfg.unity_port, cfg.connect_timeout, cfg.step_timeout,
                            cfg.env_params(steps_per_sec))

    def connect(self) -> UnityEnv:
        try:
            self.env.connect_timeout, orig = 1.5, self.env.connect_timeout   # 이미 떠 있는지 빠르게 확인
            try:
                self.env.connect()
                return self.env
            finally:
                self.env.connect_timeout = orig
        except EnvConnectionError:
            if not (self.cfg.unity_autolaunch and self.cfg.unity_exe):
                raise
        self._launch()
        self.env.connect()
        return self.env

    def _launch(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.proc = launch_unity(self.cfg.resolve_path(self.cfg.unity_exe), self.cfg.unity_port,
                                 self.cfg.unity_headless)
        time.sleep(1.0)

    def reconnect(self) -> None:
        """연결이 끊긴 뒤 복구. 자동 실행한 Unity 가 죽었다면 다시 실행한다."""
        if self.proc is not None and self.proc.poll() is not None:
            log.warning("Unity 프로세스가 종료됨 (code=%s). 재실행", self.proc.returncode)
            self._launch()
        self.env.connect()

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()

