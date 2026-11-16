"""Unity 없이 프로토콜/클라이언트/학습 루프를 검증하기 위한 가짜 서버 (간단한 2D 점-목표 환경)."""
from __future__ import annotations

import json
import socket
import struct
import threading

import numpy as np


class FakeUnityServer:
    def __init__(self, port: int = 0, drop_after_steps: int | None = None):
        self.sock = socket.socket()
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.drop_after_steps = drop_after_steps   # 지정 step 수 후 연결을 끊어 장애 상황을 재현
        self.steps_total = 0
        self.running = True
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self):
        while self.running:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            self._serve(c)

    def _recv(self, c, n):
        buf = b""
        while len(buf) < n:
            ch = c.recv(n - len(buf))
            if not ch:
                raise ConnectionError
            buf += ch
        return buf

    def _serve(self, c):
        pos = np.zeros(2); goal = np.array([0.5, 0.5]); step = 0
        try:
            while True:
                (n,) = struct.unpack(">I", self._recv(c, 4))
                req = json.loads(self._recv(c, n))
                resp = {"ok": True, "error": "", "env_name": "Fake", "obs_dim": 10, "act_dim": 2,
                        "act_low": [-1, -1], "act_high": [1, 1], "obs": [0] * 10, "reward": 0.0,
                        "done": False, "truncated": False, "success": False, "fell": False, "step": 0,
                        "distance": 0.0}
                cmd = req["cmd"]
                if cmd == "reset":
                    pos = np.random.uniform(-1, 1, 2); step = 0
                elif cmd == "step":
                    self.steps_total += 1
                    if self.drop_after_steps and self.steps_total == self.drop_after_steps:
                        c.close()
                        return
                    pos = pos + 0.2 * np.clip(req["action"], -1, 1); step += 1
                    d = float(np.linalg.norm(goal - pos))
                    resp["success"] = d < 0.2
                    resp["fell"] = bool(np.abs(pos).max() > 2)
                    resp["done"] = resp["success"] or resp["fell"]
                    resp["truncated"] = (not resp["done"]) and step >= 30
                    resp["reward"] = -d
                    resp["step"] = step
                    resp["distance"] = d
                rel = goal - pos
                resp["obs"] = [rel[0], rel[1], 0, 0, pos[0], pos[1], 0, 0, 0, 0]
                data = json.dumps(resp).encode()
                c.sendall(struct.pack(">I", len(data)) + data)
        except (ConnectionError, OSError):
            pass
        finally:
            c.close()

    def close(self):
        self.running = False
        self.sock.close()

