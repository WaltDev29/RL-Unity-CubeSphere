"""Unity TCP 서버용 클라이언트. Gymnasium 호환 인터페이스(reset/step/spaces)를 제공한다.

프로토콜: [uint32 big-endian 길이][UTF-8 JSON]. 환경 중립적이므로 obs/act 차원은 hello 응답에서 읽는다.
"""
from __future__ import annotations

import json
import logging
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from gymnasium import spaces

log = logging.getLogger(__name__)

MAX_FRAME = 1 << 20


class EnvConnectionError(Exception):
    """통신 실패(연결 끊김/타임아웃). 재연결로 복구 가능할 수 있다."""


class EnvError(Exception):
    """Unity 가 ok=false 로 응답한 환경 내부 오류 (재시도해도 같은 결과)."""


class UnityEnv:
    def __init__(self, host: str, port: int, connect_timeout: float, step_timeout: float,
                 env_params: dict):
        self.host, self.port = host, port
        self.connect_timeout, self.step_timeout = connect_timeout, step_timeout
        self.env_params = env_params
        self._sock: socket.socket | None = None
        self.observation_space: spaces.Box | None = None
        self.action_space: spaces.Box | None = None
        self.env_name = ""

    # ---------- 연결 ----------
    @property
    def connected(self) -> bool:
        return self._sock is not None

    def connect(self) -> None:
        """서버에 연결하고 hello/config 로 초기화한다. connect_timeout 동안 재시도."""
        self.close_socket()
        deadline = time.time() + self.connect_timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                s = socket.create_connection((self.host, self.port), timeout=2.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(self.step_timeout)
                self._sock = s
                break
            except OSError as e:
                last_err = e
                time.sleep(0.5)
        if self._sock is None:
            raise EnvConnectionError(f"Unity 연결 실패 {self.host}:{self.port} ({last_err})")
        try:
            hello = self._request({"cmd": "hello"})
            self.env_name = hello["env_name"]
            self.observation_space = spaces.Box(-1.0, 1.0, (hello["obs_dim"],), np.float32)
            self.action_space = spaces.Box(np.array(hello["act_low"], np.float32),
                                           np.array(hello["act_high"], np.float32), dtype=np.float32)
            self.configure(self.env_params)
        except EnvConnectionError:
            self.close_socket()
            raise
        log.info("Unity 연결됨: %s obs=%s act=%s", self.env_name,
                 self.observation_space.shape, self.action_space.shape)

    def configure(self, env_params: dict) -> None:
        self.env_params = env_params
        self._request({"cmd": "config", "env": env_params})

    def close_socket(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._request({"cmd": "close"})
            except (EnvConnectionError, EnvError):
                pass
        self.close_socket()

    # ---------- Gym API ----------
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        r = self._request({"cmd": "reset", "seed": -1 if seed is None else int(seed)})
        return np.asarray(r["obs"], np.float32), self._info(r)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        r = self._request({"cmd": "step", "action": a.tolist()})
        return (np.asarray(r["obs"], np.float32), float(r["reward"]),
                bool(r["done"]), bool(r["truncated"]), self._info(r))

    @staticmethod
    def _info(r: dict) -> dict:
        return {"success": r["success"], "fell": r["fell"], "step": r["step"], "distance": r["distance"]}

    # ---------- 저수준 ----------
    def _request(self, msg: dict) -> dict:
        if self._sock is None:
            raise EnvConnectionError("연결되어 있지 않음")
        data = json.dumps(msg).encode("utf-8")
        try:
            self._sock.sendall(struct.pack(">I", len(data)) + data)
            (n,) = struct.unpack(">I", self._recv_exact(4))
            if n <= 0 or n > MAX_FRAME:
                raise EnvConnectionError(f"잘못된 프레임 길이 {n}")
            resp = json.loads(self._recv_exact(n).decode("utf-8"))
        except (OSError, struct.error, json.JSONDecodeError) as e:
            # 프레임 동기화가 깨졌을 수 있으므로 소켓을 폐기하고 재연결을 유도
            self.close_socket()
            raise EnvConnectionError(f"{msg.get('cmd')} 요청 실패: {e}") from e
        if not resp.get("ok", False):
            raise EnvError(f"Unity 오류({msg.get('cmd')}): {resp.get('error')}")
        return resp

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("서버가 연결을 닫음")
            buf.extend(chunk)
        return bytes(buf)


def launch_unity(exe: Path, port: int, headless: bool) -> subprocess.Popen:
    """빌드된 Unity 실행 파일을 지정 포트로 실행한다."""
    if not exe.exists():
        raise FileNotFoundError(f"Unity 실행 파일이 없습니다: {exe}")
    args = [str(exe), "-rlport", str(port)]
    if headless:
        args += ["-batchmode", "-nographics"]
    else:
        args += ["-screen-width", "960", "-screen-height", "600", "-screen-fullscreen", "0"]
    log.info("Unity 실행: %s", " ".join(args))
    return subprocess.Popen(args)

