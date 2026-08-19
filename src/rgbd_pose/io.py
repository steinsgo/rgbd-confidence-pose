from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_rgbd_frame(
    directory: str | Path,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(rgb, np.uint8)).save(directory / "rgb.png")
    np.save(directory / "depth_m.npy", np.asarray(depth_m, np.float32))
    np.savetxt(directory / "intrinsics.txt", np.asarray(intrinsics, float), fmt="%.9f")


def load_rgbd_frame(directory: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = Path(directory)
    rgb = np.asarray(Image.open(directory / "rgb.png").convert("RGB"))
    depth = np.load(directory / "depth_m.npy")
    intrinsics = np.loadtxt(directory / "intrinsics.txt")
    return rgb, depth, intrinsics
