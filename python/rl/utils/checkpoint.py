"""체크포인트/모델 저장·로드."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def save_torch(obj: dict, path: Path) -> None:
    """임시 파일에 쓴 뒤 교체하여, 저장 도중 중단돼도 기존 파일이 깨지지 않게 한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_torch(path: Path, device) -> dict:
    # 체크포인트에는 numpy 배열(리플레이 버퍼)이 포함되므로 weights_only=False (신뢰된 로컬 파일 전용)
    return torch.load(path, map_location=device, weights_only=False)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

