"""CLI: python -m rl.cli {train,eval,plot,compare}"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import ConfigError, load_config
from .utils.runio import RunDir
from .utils.plotting import compare_runs, save_plots


def _setup_logging(run: RunDir) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(run.log_path, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


def _new_run_dir(cfg, prefix: str) -> Path:
    return cfg.resolve_path(cfg.runs_dir) / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rl")
    ap.add_argument("--env-file", default=None, help=".env 경로 (기본: 프로젝트 루트)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="학습")
    t.add_argument("--run-dir", default=None, help="지정 시 해당 폴더 사용 (GUI 가 사용)")
    t.add_argument("--resume", default=None, help="재개할 run 폴더")
    t.add_argument("--episodes", type=int, default=None, help="TOTAL_EPISODES 덮어쓰기")
    t.add_argument("--algorithm", default=None, help="ALGORITHM 덮어쓰기")

    e = sub.add_parser("eval", help="평가")
    e.add_argument("--model", required=True)
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--run-dir", default=None)

    p = sub.add_parser("plot", help="run 의 그래프 재생성")
    p.add_argument("run_dir")

    c = sub.add_parser("compare", help="여러 run 비교 그래프")
    c.add_argument("run_dirs", nargs="+")
    c.add_argument("--out", default="compare.png")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "plot":
            for pth in save_plots(Path(a.run_dir), Path(a.run_dir) / "plots"):
                print(pth)
            return 0
        if a.cmd == "compare":
            print(compare_runs([Path(x) for x in a.run_dirs], Path(a.out)))
            return 0

        overrides = {}
        if a.cmd == "train":
            overrides = {"total_episodes": a.episodes, "algorithm": a.algorithm}
        cfg = load_config(a.env_file, overrides)

        if a.cmd == "train":
            root = Path(a.resume) if a.resume else (Path(a.run_dir) if a.run_dir else _new_run_dir(cfg, f"train_{cfg.algorithm}"))
            run = RunDir(root)
            _setup_logging(run)
            if not a.resume:
                (root / "config_snapshot.json").write_text(
                    __import__("json").dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
            from .train.trainer import Trainer   # torch import 지연: plot/compare 는 가볍게 실행
            return Trainer(cfg, run, resume=bool(a.resume)).train()

        if a.cmd == "eval":
            root = Path(a.run_dir) if a.run_dir else _new_run_dir(cfg, "eval")
            run = RunDir(root)
            _setup_logging(run)
            from .train.evaluator import Evaluator
            return Evaluator(cfg, run, Path(a.model), a.episodes).evaluate()
    except ConfigError as ex:
        print(f"[설정 오류] {ex}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

