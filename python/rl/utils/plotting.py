"""학습 결과 그래프 PNG 저장 및 run 비교."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # GUI 없이 파일 저장만 하므로 백엔드 고정
import matplotlib.pyplot as plt  # noqa: E402


def _read(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except (TypeError, ValueError):
                    cols.setdefault(k, []).append(float("nan"))
    return cols


def _line(ax, x, y, title, ylabel, color="tab:blue", label=None):
    ax.plot(x, y, color=color, linewidth=1.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def save_plots(run_root: Path, plot_dir: Path) -> list[Path]:
    csv_path = run_root / "episodes.csv"
    if not csv_path.exists():
        return []
    d = _read(csv_path)
    if not d.get("episode"):
        return []
    ep = d["episode"]
    saved: list[Path] = []

    def save(fig, name):
        p = plot_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=120)
        plt.close(fig)
        saved.append(p)

    fig, ax = plt.subplots(figsize=(8, 4)); _line(ax, ep, d["reward"], "Episode Reward", "Reward"); save(fig, "episode_reward.png")
    fig, ax = plt.subplots(figsize=(8, 4)); _line(ax, ep, d["reward_ma"], "Reward Moving Average", "Reward (MA)", "tab:orange"); save(fig, "reward_moving_average.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    _line(ax, ep, d["critic_loss"], "Loss", "Loss", "tab:red", "critic")
    ax.plot(ep, d["actor_loss"], color="tab:green", linewidth=1.0, label="actor"); ax.legend(); save(fig, "loss.png")
    fig, ax = plt.subplots(figsize=(8, 4)); _line(ax, ep, d["success_rate"], "Success Rate (moving window)", "Rate", "tab:green"); ax.set_ylim(-0.05, 1.05); save(fig, "success_rate.png")
    fig, ax = plt.subplots(figsize=(8, 4)); _line(ax, ep, d["length"], "Episode Length", "Steps", "tab:purple"); save(fig, "episode_length.png")
    return saved


def compare_runs(run_roots: list[Path], out: Path, window: int = 50) -> Path:
    """여러 run 의 Reward 이동 평균/성공률을 한 그래프에 겹쳐 알고리즘 비교에 사용한다."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for r in run_roots:
        p = r / "episodes.csv"
        if not p.exists():
            continue
        d = _read(p)
        axes[0].plot(d["episode"], d["reward_ma"], label=r.name)
        axes[1].plot(d["episode"], d["success_rate"], label=r.name)
    axes[0].set_title(f"Reward MA"); axes[1].set_title("Success Rate")
    for a in axes:
        a.set_xlabel("Episode"); a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out

