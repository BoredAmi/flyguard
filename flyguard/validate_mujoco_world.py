"""Numeric sanity check for the MuJoCo world (CLAUDE.md layer 2): confirms
looming trials expand and translation trials don't, from the rendered
pixels themselves rather than by visual inspection. Every trial in the
dataset is checked, not just one hand-picked example.

Usage:
    MUJOCO_GL=egl python -m flyguard.validate_mujoco_world --npz data/mujoco_dataset.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flyguard.mujoco_world import apparent_area_px


def main(npz_path: Path):
    z = np.load(npz_path)
    frames, condition, trial_id = z["frames"], z["condition"], z["trial_id"]
    print(f"{npz_path}: {frames.shape[0]} trials, {frames.shape[1]} frames, "
          f"{frames.shape[2]}x{frames.shape[3]} px\n")

    n_ok = 0
    for i in range(frames.shape[0]):
        sizes = apparent_area_px(frames[i])
        cond = str(condition[i])
        growth = sizes[-1] / max(sizes[0], 1.0)
        cv = sizes.std() / max(sizes.mean(), 1e-9)
        if cond == "looming":
            ok = growth > 1.5 and np.all(np.diff(sizes) >= -2)
            print(f"trial {trial_id[i]:2d} [{cond:11s}] growth={growth:5.2f}x "
                  f"monotonic={'yes' if ok else 'NO'}  {'OK' if ok else 'FAIL'}")
        else:
            ok = cv < 0.05
            print(f"trial {trial_id[i]:2d} [{cond:11s}] size CV={cv:.3f}  "
                  f"{'OK' if ok else 'FAIL'}")
        n_ok += ok

    print(f"\n{n_ok}/{frames.shape[0]} trials pass their condition's sanity check")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("data/mujoco_dataset.npz"))
    main(ap.parse_args().npz)
