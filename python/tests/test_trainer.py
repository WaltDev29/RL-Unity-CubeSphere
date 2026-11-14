"""Trainer 통합 테스트(가짜 서버): 학습 완료/산출물, 중지 후 재개, 연결 끊김 복구."""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_unity_server import FakeUnityServer  # noqa: E402
from rl.config import PROJECT_ROOT, load_config  # noqa: E402
from rl.train.trainer import Trainer  # noqa: E402
from rl.utils.runio import RunDir  # noqa: E402


def make_cfg(tmp_path, port, **kw):
    base = {"runs_dir": str(tmp_path), "hidden_units": 32, "batch_size": 16, "warmup_steps": 20,
            "buffer_size": 2000, "device": "cpu", "eval_every": 3, "eval_episodes": 2, "checkpoint_every": 3,
            "total_episodes": 6, "reconnect_retries": 3, "unity_port": port, "unity_autolaunch": False,
            "connect_timeout": 3.0, "step_timeout": 3.0, "max_steps_per_episode": 30}
    base.update(kw)
    return load_config(PROJECT_ROOT / ".env.example", base)


def test_train_produces_artifacts(tmp_path):
    srv = FakeUnityServer()
    try:
        run = RunDir(tmp_path / "run")
        assert Trainer(make_cfg(tmp_path, srv.port), run).train() == 0
    finally:
        srv.close()
    assert (run.model_dir / "final.pt").exists() and (run.model_dir / "best.pt").exists()
    assert (run.ckpt_dir / "latest.pt").exists() and list(run.ckpt_dir.glob("ckpt_ep*.pt"))
    assert (run.episodes_csv).exists() and (run.plot_dir / "episode_reward.png").exists()
    assert run.read_status()["state"] == "finished"


def test_stop_and_resume(tmp_path):
    srv = FakeUnityServer()
    try:
        run = RunDir(tmp_path / "run")
        cfg = make_cfg(tmp_path, srv.port, total_episodes=40)
        t = Trainer(cfg, run)
        result = {}
        th = threading.Thread(target=lambda: result.setdefault("code", t.train()))
        th.start()
        while run.read_status().get("episode", 0) < 5:
            time.sleep(0.05)
        run.write_control("pause"); time.sleep(0.5)
        ep_paused = run.read_status()["episode"]; time.sleep(0.5)
        assert run.read_status()["episode"] == ep_paused          # 일시정지 중에는 진행 없음
        run.write_control("run"); time.sleep(0.3)
        run.write_control("stop"); th.join(20)
        assert run.read_status()["state"] == "stopped"
        stopped_at = run.read_status()["episode"]

        cfg2 = make_cfg(tmp_path, srv.port, total_episodes=stopped_at + 3)
        t2 = Trainer(cfg2, run, resume=True)
        assert t2.train() == 0
        assert t2.episode == stopped_at + 3 and run.read_status()["state"] == "finished"
    finally:
        srv.close()


def test_recover_from_disconnect(tmp_path):
    srv = FakeUnityServer(drop_after_steps=60)
    try:
        run = RunDir(tmp_path / "run")
        t = Trainer(make_cfg(tmp_path, srv.port, total_episodes=8), run)
        assert t.train() == 0 and t.episode == 8       # 연결이 끊겨도 재연결 후 목표 에피소드 도달
    finally:
        srv.close()


def test_unreachable_unity_saves_error(tmp_path):
    run = RunDir(tmp_path / "run")
    t = Trainer(make_cfg(tmp_path, 1, connect_timeout=1.0), run)
    assert t.train() == 1 and run.read_status()["state"] == "error"

