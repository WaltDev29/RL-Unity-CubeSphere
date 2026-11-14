import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_unity_server import FakeUnityServer  # noqa: E402
from rl.agents import ReplayBuffer, make_agent  # noqa: E402
from rl.config import ConfigError, PROJECT_ROOT, load_config  # noqa: E402
from rl.envs.unity_env import EnvConnectionError, UnityEnv  # noqa: E402
from rl.utils.runio import RunDir  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return load_config(PROJECT_ROOT / ".env.example",
                       {"runs_dir": str(tmp_path), "hidden_units": 32, "batch_size": 16, "warmup_steps": 20,
                        "buffer_size": 1000, "device": "cpu", "eval_every": 2, "eval_episodes": 2,
                        "checkpoint_every": 2, "total_episodes": 4, "reconnect_retries": 2})


def test_env_example_loads():
    c = load_config(PROJECT_ROOT / ".env.example")
    assert c.total_episodes > 0 and c.algorithm in ("sac", "td3")


def test_missing_key_raises(tmp_path):
    p = tmp_path / ".env"
    p.write_text("TOTAL_EPISODES=3\n")
    with pytest.raises(ConfigError, match="누락"):
        load_config(p)


def test_bad_value_raises(tmp_path):
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").replace("TOTAL_EPISODES=1500", "TOTAL_EPISODES=abc")
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="형식 오류"):
        load_config(p)


def test_client_protocol_roundtrip(cfg):
    srv = FakeUnityServer()
    try:
        env = UnityEnv("127.0.0.1", srv.port, 3, 3, cfg.env_params(0))
        env.connect()
        assert env.observation_space.shape == (6,) and env.action_space.shape == (2,)
        obs, _ = env.reset(1)
        assert obs.shape == (6,)
        obs, r, term, trunc, info = env.step([0.1, 0.1])
        assert obs.shape == (6,) and isinstance(r, float) and "success" in info
        env.close()
    finally:
        srv.close()


def test_connection_failure_raises(cfg):
    env = UnityEnv("127.0.0.1", 1, 1, 1, cfg.env_params(0))
    with pytest.raises(EnvConnectionError):
        env.connect()


def test_disconnect_detected(cfg):
    srv = FakeUnityServer(drop_after_steps=2)
    try:
        env = UnityEnv("127.0.0.1", srv.port, 3, 3, cfg.env_params(0))
        env.connect(); env.reset()
        env.step([0, 0])
        with pytest.raises(EnvConnectionError):
            env.step([0, 0])
        assert not env.connected
    finally:
        srv.close()


@pytest.mark.parametrize("algo", ["sac", "td3"])
def test_agent_update_and_save(cfg, algo, tmp_path):
    cfg.algorithm = algo
    dev = torch.device("cpu")
    agent = make_agent(algo, 6, 2, cfg, dev)
    buf = ReplayBuffer(100, 6, 2, dev)
    for _ in range(50):
        buf.add(np.random.randn(6), np.random.uniform(-1, 1, 2), 1.0, np.random.randn(6), False)
    for _ in range(4):
        out = agent.update(buf.sample(16))
    assert all(torch.isfinite(v).all() for v in out.values())
    a = agent.select_action(np.zeros(6, np.float32), deterministic=True)
    assert a.shape == (2,) and np.abs(a).max() <= 1.0
    agent2 = make_agent(algo, 6, 2, cfg, dev)
    agent2.load_state_dict(agent.state_dict())
    assert np.allclose(agent2.select_action(np.zeros(6, np.float32), True), a, atol=1e-6)


def test_buffer_state_roundtrip():
    dev = torch.device("cpu")
    b = ReplayBuffer(10, 3, 2, dev)
    for i in range(15):
        b.add(np.full(3, i), np.zeros(2), i, np.full(3, i), i % 2 == 0)
    b2 = ReplayBuffer(10, 3, 2, dev)
    b2.load_state_dict(b.state_dict())
    assert b2.size == 10 and b2.ptr == b.ptr and torch.equal(b2.obs, b.obs)


def test_control_file(tmp_path):
    r = RunDir(tmp_path / "r")
    assert r.read_control() == "run"
    r.write_control("pause")
    assert r.read_control() == "pause"
    r.write_control("stop")
    assert r.read_control() == "stop"

