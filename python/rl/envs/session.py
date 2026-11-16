"""환경 연결 생성 (필요 시 Unity 빌드 자동 실행)."""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from ..config import Config, PROJECT_ROOT
from .unity_env import EnvConnectionError, UnityEnv, launch_unity

log = logging.getLogger(__name__)


def ensure_unity_built(cfg: Config) -> None:
    """C# 스크립트(.cs)나 씬(.unity)이 .exe보다 최신이거나(브랜치 변경 등) .exe가 없으면 자동 빌드한다."""
    exe_path = cfg.resolve_path(cfg.unity_exe)
    assets_dir = PROJECT_ROOT / "Assets"
    
    needs_build = False
    if not exe_path.exists():
        needs_build = True
        log.info("Unity 실행 파일이 없어 자동 빌드를 시작합니다...")
    else:
        exe_mtime = exe_path.stat().st_mtime
        # Assets 내 모든 소스 파일의 mtime 검사
        for p in assets_dir.rglob("*"):
            if p.is_file() and p.suffix in (".cs", ".unity"):
                if p.stat().st_mtime > exe_mtime:
                    needs_build = True
                    log.info("C# 소스코드 변경 감지 (%s). Unity 자동 재빌드를 진행합니다...", p.name)
                    break

    if needs_build:
        unity_editor = Path(r"C:\Program Files\Unity\Hub\Editor\2022.3.62f3\Editor\Unity.exe")
        if not unity_editor.exists():
            log.warning("Unity 에디터 경로를 찾을 수 없어 자동 빌드를 건너뜁니다: %s", unity_editor)
            return

        cmd = [
            str(unity_editor),
            "-batchmode", "-nographics", "-quit",
            "-projectPath", str(PROJECT_ROOT),
            "-executeMethod", "RLPlatform.EditorTools.SceneBuilder.BuildPlayer",
            "-logFile", str(PROJECT_ROOT / "Logs" / "autobuild.log")
        ]
        log.info("Unity 빌드 실행 중 (잠시만 기다려주세요)...")
        res = subprocess.run(cmd, timeout=300)
        if res.returncode != 0:
            log.error("Unity 자동 빌드 실패 (code=%d). 로그: Logs/autobuild.log", res.returncode)
            raise RuntimeError("Unity 자동 빌드에 실패했습니다. Logs/autobuild.log를 확인하세요.")
        log.info("Unity 자동 빌드 완료!")


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
        ensure_unity_built(self.cfg)
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

