"""Run 디렉터리 입출력: 상태/제어 파일, 에피소드 로그. GUI 와 Trainer 가 파일로만 통신하도록 한다."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

EPISODE_FIELDS = ["episode", "total_steps", "reward", "reward_ma", "length", "success", "fell",
                  "success_rate", "critic_loss", "actor_loss", "alpha", "elapsed_sec"]
EVAL_FIELDS = ["episode", "mean_reward", "success_rate", "mean_length"]


def atomic_write_json(path: Path, data: dict) -> None:
    """GUI 가 쓰다 만 파일을 읽지 않도록 임시 파일 후 교체한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:   # Windows 에서 읽는 중이면 잠깐 실패할 수 있음
            time.sleep(0.02)
    tmp.replace(path)


class RunDir:
    def __init__(self, root: Path):
        self.root = root
        self.ckpt_dir = root / "checkpoints"
        self.model_dir = root / "models"
        self.plot_dir = root / "plots"
        for d in (self.root, self.ckpt_dir, self.model_dir, self.plot_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.status_path = root / "status.json"
        self.control_path = root / "control.json"
        self.episodes_csv = root / "episodes.csv"
        self.eval_csv = root / "eval.csv"
        self.metrics_jsonl = root / "metrics.jsonl"
        self.log_path = root / "train.log"
        self._control_mtime = 0.0
        self._control = "run"
        # 시뮬레이션 속도(steps/sec, 0=무제한). None 이면 .env 기본값을 그대로 사용
        self.speed: float | None = None

    # ----- 제어 -----
    def read_control(self) -> str:
        """run | pause | stop. mtime 이 바뀐 경우에만 파일을 읽어 step 마다 호출해도 부담이 없다."""
        try:
            m = self.control_path.stat().st_mtime
        except FileNotFoundError:
            return self._control
        if m != self._control_mtime:
            self._control_mtime = m
            try:
                data = json.loads(self.control_path.read_text(encoding="utf-8"))
                self._control = data.get("command", "run")
                self.speed = data.get("speed")
            except (OSError, json.JSONDecodeError):
                pass   # 쓰는 도중이면 다음 호출에 다시 읽는다
        return self._control

    def write_control(self, command: str | None = None, speed: float | None = None) -> None:
        """None 인 항목은 기존 파일 값을 유지한다 (명령 변경이 속도를 지우지 않도록)."""
        try:
            cur = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cur = {}
        new = {"command": command or cur.get("command", "run"),
               "speed": speed if speed is not None else cur.get("speed"), "time": time.time()}
        atomic_write_json(self.control_path, new)

    # ----- 상태 -----
    def write_status(self, **kw) -> None:
        atomic_write_json(self.status_path, kw)

    def read_status(self) -> dict:
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    # ----- 로그 -----
    def append_csv(self, path: Path, fields: list[str], row: dict) -> None:
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def append_metrics(self, row: dict) -> None:
        with open(self.metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_episode(self, row: dict) -> None:
        self.append_csv(self.episodes_csv, EPISODE_FIELDS, row)
        self.append_metrics({"type": "episode", **row})

    def log_eval(self, row: dict) -> None:
        self.append_csv(self.eval_csv, EVAL_FIELDS, row)
        self.append_metrics({"type": "eval", **row})

