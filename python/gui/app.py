"""PySide6 + pyqtgraph 대시보드. 학습 프로세스와는 파일(status.json / metrics.jsonl / control.json)로만 통신."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from rl.config import ConfigError, load_config
from rl.utils.runio import RunDir

from .runner import MetricsTail, ProcessRunner

STATE_KO = {"starting": "시작 중", "running": "학습 중", "paused": "일시정지", "evaluating": "평가 중",
            "reconnecting": "재연결 중", "stopped": "중지됨", "finished": "완료", "error": "오류", "idle": "대기"}


def fmt_time(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unity RL Dashboard")
        self.resize(1250, 820)
        try:
            self.cfg = load_config()
        except ConfigError as e:
            QtWidgets.QMessageBox.critical(self, "설정 오류", str(e))
            raise
        self.runner = ProcessRunner(self.cfg)
        self.tail: MetricsTail | None = None
        self.run: RunDir | None = None
        self._clear_data()
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(500)    # 0.5초 폴링: 학습 성능과 무관한 읽기 전용 갱신

    # ------------------------------------------------------------ UI
    def _clear_data(self) -> None:
        self.ep, self.rew, self.ma, self.loss_q, self.loss_pi, self.succ, self.length = [], [], [], [], [], [], []

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)
        # 상태 패널
        box = QtWidgets.QGroupBox("상태")
        form = QtWidgets.QFormLayout(box)
        self.lbl = {k: QtWidgets.QLabel("-") for k in
                    ("state", "unity", "episode", "reward", "ma", "loss", "succ", "length", "elapsed", "steps", "run")}
        names = {"state": "학습 상태", "unity": "Unity 연결", "episode": "Episode", "reward": "최근 Reward",
                 "ma": "Reward 이동평균", "loss": "Loss (critic)", "succ": "성공률", "length": "Episode Length",
                 "elapsed": "경과 시간", "steps": "Total Steps", "run": "Run 폴더"}
        for k, n in names.items():
            self.lbl[k].setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            form.addRow(n, self.lbl[k])
        self.lbl["run"].setWordWrap(True)
        left.addWidget(box)

        # 제어
        ctl = QtWidgets.QGroupBox("제어")
        g = QtWidgets.QGridLayout(ctl)
        self.btn_start = QtWidgets.QPushButton("학습 시작")
        self.btn_pause = QtWidgets.QPushButton("일시정지")
        self.btn_resume = QtWidgets.QPushButton("재개")
        self.btn_stop = QtWidgets.QPushButton("종료")
        self.btn_continue = QtWidgets.QPushButton("중단된 run 이어서 학습…")
        for i, b in enumerate((self.btn_start, self.btn_pause, self.btn_resume, self.btn_stop)):
            g.addWidget(b, i // 2, i % 2)
        g.addWidget(self.btn_continue, 2, 0, 1, 2)
        self.model_edit = QtWidgets.QLineEdit()
        self.model_edit.setPlaceholderText("평가할 모델 (.pt)")
        btn_model = QtWidgets.QPushButton("모델 선택…")
        self.eval_n = QtWidgets.QSpinBox(); self.eval_n.setRange(1, 1000); self.eval_n.setValue(20)
        self.eval_n.setPrefix("평가 에피소드 수: ")
        self.eval_n.setPrefix("평가 에피소드 수: ")
        self.btn_eval = QtWidgets.QPushButton("평가 실행")
        g.addWidget(self.model_edit, 3, 0, 1, 2)
        g.addWidget(btn_model, 4, 0)
        g.addWidget(self.eval_n, 4, 1)
        g.addWidget(self.btn_eval, 5, 0, 1, 2)
        # 시뮬레이션 속도: 초당 Unity step 수 (0 = 제한 없음/최고 속도). 실행 중에도 즉시 반영
        g.addWidget(QtWidgets.QLabel("시뮬레이션 속도 (steps/s, 0=최고속)"), 6, 0, 1, 2)
        self.speed_spin = QtWidgets.QSpinBox()
        self.speed_spin.setRange(0, 2000)
        self.speed_spin.setSingleStep(5)
        self.speed_spin.setValue(int(self.cfg.train_steps_per_sec))
        self.speed_spin.setToolTip("관찰은 10 전후(물리 1초 = 실시간 1초), 학습은 0(최고속) 권장")
        g.addWidget(self.speed_spin, 7, 0)
        presets = QtWidgets.QHBoxLayout()
        for label, v in (("실시간", 10), ("5배", 50), ("최고속", 0)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, val=v: self.speed_spin.setValue(val))
            presets.addWidget(b)
        g.addLayout(presets, 7, 1)
        self._speed_touched = False
        self._speed_timer = QtCore.QTimer(self)
        self._speed_timer.setSingleShot(True)
        self._speed_timer.timeout.connect(lambda: self.runner.set_speed(self.speed_spin.value()))
        self.speed_spin.valueChanged.connect(self._on_speed_changed)
        left.addWidget(ctl)

        # 주요 설정
        cbox = QtWidgets.QGroupBox("주요 설정 (.env)")
        cv = QtWidgets.QVBoxLayout(cbox)
        c = self.cfg
        text = "\n".join(f"{k}: {v}" for k, v in (
            ("TOTAL_EPISODES", c.total_episodes), ("MAX_STEPS", c.max_steps_per_episode), ("SEED", c.seed),
            ("ALGORITHM", c.algorithm), ("DEVICE", c.device), ("LEARNING_RATE", c.learning_rate),
            ("GAMMA", c.gamma), ("BATCH_SIZE", c.batch_size), ("WARMUP_STEPS", c.warmup_steps),
            ("UNITY", f"{c.unity_host}:{c.unity_port}"), ("RUNS_DIR", c.runs_dir)))
        lab = QtWidgets.QLabel(text)
        lab.setStyleSheet("font-family: Consolas;")
        cv.addWidget(lab)
        left.addWidget(cbox)
        left.addStretch(1)

        # 그래프
        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        grid = pg.GraphicsLayoutWidget()
        root.addWidget(grid, 1)
        self.p_rew = grid.addPlot(row=0, col=0, title="Episode Reward / Moving Average")
        self.p_loss = grid.addPlot(row=0, col=1, title="Loss")
        self.p_succ = grid.addPlot(row=1, col=0, title="Success Rate")
        self.p_len = grid.addPlot(row=1, col=1, title="Episode Length")
        for p in (self.p_rew, self.p_loss, self.p_succ, self.p_len):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setLabel("bottom", "Episode")
        self.p_rew.addLegend(); self.p_loss.addLegend()
        self.c_rew = self.p_rew.plot(pen=pg.mkPen((150, 150, 255), width=1), name="reward")
        self.c_ma = self.p_rew.plot(pen=pg.mkPen("r", width=2), name="moving avg")
        self.c_lq = self.p_loss.plot(pen=pg.mkPen("b", width=1.5), name="critic")
        self.c_lp = self.p_loss.plot(pen=pg.mkPen("g", width=1.5), name="actor")
        self.c_succ = self.p_succ.plot(pen=pg.mkPen("g", width=2))
        self.p_succ.setYRange(0, 1.05)
        self.c_len = self.p_len.plot(pen=pg.mkPen("m", width=1.5))

        self.btn_start.clicked.connect(self._on_start)
        self.btn_pause.clicked.connect(lambda: self.runner.command("pause"))
        self.btn_resume.clicked.connect(lambda: self.runner.command("run"))
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_continue.clicked.connect(self._on_continue)
        btn_model.clicked.connect(self._on_pick_model)
        self.btn_eval.clicked.connect(self._on_eval)

    # ------------------------------------------------------------ 동작
    def _attach(self, run: RunDir) -> None:
        self.run = run
        self.tail = MetricsTail(run.root)
        self._clear_data()

    def _guard(self, fn):
        try:
            self._attach(fn())
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "실행 실패", str(e))

    def _speed_arg(self):
        """사용자가 속도를 조작했을 때만 시작 시 전달 (아니면 .env 의 학습/평가 기본 속도 사용)."""
        return float(self.speed_spin.value()) if self._speed_touched else None

    def _on_speed_changed(self, _v: int) -> None:
        self._speed_touched = True
        self._speed_timer.start(300)   # 연속 조작 중에는 마지막 값만 전송

    def _on_start(self) -> None:
        self._guard(lambda: self.runner.start_train(self._speed_arg()))

    def _on_continue(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "이어서 학습할 run 폴더",
                                                       str(self.cfg.resolve_path(self.cfg.runs_dir)))
        if d:
            self._guard(lambda: self.runner.resume_train(Path(d), self._speed_arg()))

    def _on_stop(self) -> None:
        self.btn_stop.setEnabled(False)
        QtCore.QTimer.singleShot(0, lambda: (self.runner.stop(), self.btn_stop.setEnabled(True)))

    def _on_pick_model(self) -> None:
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "모델 선택", str(self.cfg.resolve_path(self.cfg.runs_dir)),
                                                     "Model (*.pt)")
        if f:
            self.model_edit.setText(f)

    def _on_eval(self) -> None:
        m = self.model_edit.text().strip()
        if not m or not Path(m).exists():
            QtWidgets.QMessageBox.warning(self, "모델", "유효한 모델 파일을 선택하세요")
            return
        self._guard(lambda: self.runner.start_eval(Path(m), self.eval_n.value(), self._speed_arg()))

    # ------------------------------------------------------------ 갱신
    def _refresh(self) -> None:
        if self.run is None or self.tail is None:
            return
        for r in self.tail.read_new():
            if r.get("type") != "episode":
                continue
            self.ep.append(r["episode"]); self.rew.append(r["reward"]); self.ma.append(r["reward_ma"])
            self.loss_q.append(r["critic_loss"]); self.loss_pi.append(r["actor_loss"])
            self.succ.append(r["success_rate"]); self.length.append(r["length"])
        if self.ep:
            self.c_rew.setData(self.ep, self.rew); self.c_ma.setData(self.ep, self.ma)
            self.c_lq.setData(self.ep, self.loss_q); self.c_lp.setData(self.ep, self.loss_pi)
            self.c_succ.setData(self.ep, self.succ); self.c_len.setData(self.ep, self.length)
            self.lbl["reward"].setText(f"{self.rew[-1]:.2f}")
            self.lbl["ma"].setText(f"{self.ma[-1]:.2f}")
            q = self.loss_q[-1]
            self.lbl["loss"].setText("-" if math.isnan(q) else f"{q:.4f}")
            self.lbl["succ"].setText(f"{self.succ[-1] * 100:.1f}%")
            self.lbl["length"].setText(str(int(self.length[-1])))
        st = self.run.read_status()
        if st:
            state = st.get("state", "idle")
            # 프로세스가 죽었는데 상태 파일이 running 인 경우를 구분
            if not self.runner.alive and state in ("running", "starting", "paused", "evaluating", "reconnecting"):
                state = "error"
            self.lbl["state"].setText(STATE_KO.get(state, state) + (f" - {st['message']}" if st.get("message") else ""))
            self.lbl["unity"].setText("연결됨" if st.get("unity_connected") and self.runner.alive else "끊김")
            self.lbl["episode"].setText(f"{st.get('episode', 0)} / {st.get('total_episodes', 0)}")
            self.lbl["elapsed"].setText(fmt_time(st.get("elapsed_sec", 0)))
            self.lbl["steps"].setText(str(st.get("total_steps", 0)))
            self.lbl["run"].setText(st.get("run_dir", ""))
        self.btn_start.setEnabled(not self.runner.alive)
        self.btn_continue.setEnabled(not self.runner.alive)
        self.btn_eval.setEnabled(not self.runner.alive)

    def closeEvent(self, e) -> None:
        # GUI 를 닫아도 학습 프로세스가 살아 있으면 사용자에게 선택권을 준다
        if self.runner.alive:
            ans = QtWidgets.QMessageBox.question(self, "종료", "실행 중인 학습을 종료하고 창을 닫을까요?\n(아니오: 학습은 계속 실행)")
            if ans == QtWidgets.QMessageBox.Yes:
                self.runner.stop()
        e.accept()


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    w = Dashboard()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

