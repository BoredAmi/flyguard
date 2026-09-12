"""Records one closed-loop obstacle-avoidance run for playback.

Companion to `record_live_demo.py`, which captures the *open-loop* escape
circuit (scene + spike raster + /cmd_vel). This one captures the thing that
recording cannot show: the loop closing. The agent's steering changes where
it goes, which changes what the camera sees, which changes the steering.

It records a real trial by calling `avoid.run_trial` with a `frame_sink`
rather than reimplementing the loop, so the recording is produced by exactly
the code path the benchmark measures. If this script grew its own copy of
the control loop, the recording would quietly stop being evidence about the
benchmark -- the same hazard flagged in `record_live_demo.py`.

Output is a single JSON bundle: the arena geometry, the agent's eye view per
tick as base64 JPEG, and every per-tick control signal on one shared
timeline. Frames are JPEG rather than PNG because these scenes are densely
textured (the checkerboard scenes in `record_live_demo.py` compress well as
PNG; a pillar corridor does not) -- roughly 5 KB versus 20 KB per frame.

Usage:
    MUJOCO_GL=egl python -m flyguard.record_avoid_demo --arena 0 \\
        --out avoid_recording.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flyguard.arena import ArenaConfig, sample_obstacles
from flyguard.avoid import (
    CONTROLLERS,
    BilateralEncoder,
    _default_columns_path,
    calibrate_controller,
    calibrate_drive,
    run_trial,
)


def frame_to_jpeg_b64(frame: np.ndarray, quality: int = 72) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main(out_path: Path, columns_csv: Path, arena_seed: int, controller_name: str,
         resolution: int, tick_hz: float, max_ticks: int, npz_path: Path,
         mirror: str, row_band: tuple[float, float], quality: int):
    cfg = ArenaConfig(seed=arena_seed)
    obstacles = sample_obstacles(cfg)

    print("loading T4/T5 columns and the real core circuit...")
    encoder = BilateralEncoder(columns_csv, mirror=mirror, row_band=row_band)

    cal_seeds = [900, 901, 902]
    print("calibrating (same procedure as the benchmark)...")
    cal = calibrate_drive(encoder, cfg, seeds=cal_seeds, resolution=resolution,
                          tick_hz=tick_hz, v_cruise=1.0, max_ticks=max_ticks)
    controller = CONTROLLERS[controller_name](
        drive_center=cal["drive_center"], drive_scale=cal["drive_scale"],
        tick_hz=tick_hz, npz_path=npz_path)
    cc = {}
    if controller.needs_vision:
        cc = calibrate_controller(controller, encoder, cfg, seeds=cal_seeds,
                                  resolution=resolution, tick_hz=tick_hz,
                                  max_ticks=max_ticks)
        print(f"  turn_offset={cc['turn_offset']:+.3f} "
              f"saccade_threshold={cc['saccade_threshold']:.3f} "
              f"estop_threshold={cc['estop_threshold']:.2f}")

    frames_b64: list[str] = []

    def frame_sink(_tick, frame, _state):
        frames_b64.append(frame_to_jpeg_b64(frame, quality))

    print(f"recording arena {arena_seed} with the {controller_name} controller...")
    result = run_trial(controller, cfg, encoder, resolution, tick_hz, max_ticks,
                       keep_telemetry=True, frame_sink=frame_sink)
    outcome = "collided" if result.collided else ("goal" if result.reached_goal else "timeout")
    print(f"  {outcome}: reached x={result.progress_x:.1f}/{cfg.goal_x:.0f} m "
          f"in {result.ticks} ticks, min clearance {result.min_clearance:.2f} m")

    # run_trial renders one frame per tick before commanding, so telemetry and
    # frames are the same length and share an index.
    telemetry = result.telemetry
    if len(frames_b64) != len(telemetry):
        # Defensive: a mismatch would silently desynchronise the playback.
        n = min(len(frames_b64), len(telemetry))
        print(f"  WARNING: {len(frames_b64)} frames vs {len(telemetry)} telemetry "
              f"rows; truncating both to {n}")
        frames_b64, telemetry = frames_b64[:n], telemetry[:n]

    bundle = {
        "meta": {
            "controller": controller_name,
            "arena_seed": arena_seed,
            "outcome": outcome,
            "progress_x": round(result.progress_x, 3),
            "goal_x": cfg.goal_x,
            "ticks": len(telemetry),
            "tick_hz": tick_hz,
            "resolution": resolution,
            "min_clearance": round(result.min_clearance, 3),
            "n_estop_ticks": result.n_estop_ticks,
            "mirror": mirror,
            "row_band": list(row_band),
            "saccade_threshold": cc.get("saccade_threshold"),
            "estop_threshold": cc.get("estop_threshold"),
            "turn_offset": cc.get("turn_offset"),
        },
        "arena": {
            "obstacles": [[round(float(x), 3), round(float(y), 3)] for x, y in obstacles],
            "obstacle_radius": cfg.obstacle_radius,
            "agent_radius": cfg.agent_radius,
            "lane_half_width": cfg.lane_half_width,
            "goal_x": cfg.goal_x,
        },
        "frames_jpeg_b64": frames_b64,
        "telemetry": telemetry,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB, "
          f"{len(frames_b64)} frames)")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("avoid_recording.json"))
    ap.add_argument("--columns", type=Path, default=Path(_default_columns_path()))
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--arena", type=int, default=0, help="arena seed to record")
    ap.add_argument("--controller", default="connectome", choices=sorted(CONTROLLERS))
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--max-ticks", type=int, default=320)
    ap.add_argument("--mirror", default="anatomical", choices=["none", "anatomical"])
    ap.add_argument("--row-band", type=float, nargs=2, default=(0.0, 0.6))
    ap.add_argument("--quality", type=int, default=72, help="JPEG quality for frames")
    a = ap.parse_args()
    main(a.out, a.columns, a.arena, a.controller, a.resolution, a.tick_hz,
         a.max_ticks, a.npz, a.mirror, tuple(a.row_band), a.quality)
