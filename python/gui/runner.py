"""학습/평가 프로세스 제어 (subprocess) 및 run 폴더 파일 읽기. GUI 와 학습은 파일로만 통신한다."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from rl.config import PROJECT_ROOT, Config
from rl.utils.runio import RunDir


class ProcessRunner:
    """Trainer/Evaluator 를 별도 프로세스로 실행하고 제어 파일로 일시정지/재개/종료를 전달한다."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.run: RunDir | None = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _spawn(self, args: list[str], run_dir: Path, speed: float | None = None) -> RunDir:
        if self.alive:
            raise RuntimeError("이미 실행 중인 프로세스가 있습니다")
        self.run = RunDir(run_dir)
        self.run.write_control("run", speed)   # speed=None 이면 .env 기본 속도 사용
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "rl.cli", *args, "--run-dir", str(run_dir)] if "--resume" not in args
            else [sys.executable, "-m", "rl.cli", *args],
            cwd=str(PROJECT_ROOT / "python"),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
        return self.run

    def _new_dir(self, prefix: str) -> Path:
        return self.cfg.resolve_path(self.cfg.runs_dir) / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

    def start_train(self, speed: float | None = None) -> RunDir:
        return self._spawn(["train"], self._new_dir(f"train_{self.cfg.algorithm}"), speed)

    def resume_train(self, run_dir: Path, speed: float | None = None) -> RunDir:
        return self._spawn(["train", "--resume", str(run_dir)], run_dir, speed)

    def start_eval(self, model: Path, episodes: int, speed: float | None = None) -> RunDir:
        return self._spawn(["eval", "--model", str(model), "--episodes", str(episodes)], self._new_dir("eval"), speed)

    def command(self, cmd: str) -> None:
        if self.run is not None:
            self.run.write_control(cmd)

    def set_speed(self, steps_per_sec: float) -> None:
        """실행 중인 학습/평가의 Unity 시뮬레이션 속도(steps/sec, 0=무제한)를 변경한다."""
        if self.run is not None:
            self.run.write_control(None, float(steps_per_sec))

    def stop(self, wait: float = 15.0) -> None:
        """정상 종료 요청 후 대기(체크포인트 저장 시간 확보), 응답이 없으면 강제 종료."""
        self.command("stop")
        if self.proc:
            try:
                self.proc.wait(timeout=wait)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MetricsTail:
    """metrics.jsonl 을 증분으로 읽는다 (매번 전체를 다시 읽지 않아 GUI 부하가 일정)."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "metrics.jsonl"
        self.offset = 0
        self._buf = b""

    def read_new(self) -> list[dict]:
        rows: list[dict] = []
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
                self.offset += len(data)
        except FileNotFoundError:
            return rows
        self._buf += data
        *lines, self._buf = self._buf.split(b"\n")
        for ln in lines:
            if ln.strip():
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
        return rows

