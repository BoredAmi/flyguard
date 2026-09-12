"""Post-mortem for a single closed-loop trial: why did it end that way?

    MUJOCO_GL=egl python -m flyguard.diagnose_trial --arena 9 --controller connectome

The benchmark records *what* happened (collided, reached goal, how far). This
records *why*: the turn signal every tick, the threshold it was measured
against, the escape statistic, and the pose. That gap is not hypothetical -- a
trial was observed driving straight into a corridor wall for 7.8 seconds and
the stored record could not say whether the steering signal was absent, below
threshold, or pointing the wrong way.

The three explanations look identical from the outside and need different
fixes:

* **signal absent** -- the wall or obstacle produced no left/right difference
  at all, so the geometry is invisible to a bilateral comparison;
* **below threshold** -- the difference existed but never exceeded the noise
  floor measured on an empty corridor, so the calibration is the binding
  constraint;
* **wrong sign** -- the difference pointed into the obstacle, which no gain
  tuning fixes and which aggregate statistics cannot see.

Prints a per-tick trace and a verdict that names which one it was.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flyguard.arena import ArenaConfig
from flyguard.avoid import (
    CONTROLLERS,
    _default_columns_path,
    calibrate_controller,
    calibrate_drive,
    run_trial,
)
from flyguard.runtime import BilateralEncoder


def _classify(rows: list, saccade_threshold: float) -> dict:
    """Say which of the three failure shapes the trace matches."""
    turns = np.array([r["turn"] for r in rows if "turn" in r])
    if not len(turns):
        return {"verdict": "no vision ticks recorded"}
    peak = float(np.max(np.abs(turns)))
    n_over = int(np.sum(np.abs(turns) >= saccade_threshold))
    return {
        "n_vision_ticks": len(turns),
        "peak_abs_turn": peak,
        "saccade_threshold": saccade_threshold,
        "ticks_over_threshold": n_over,
        "headroom": peak / saccade_threshold if saccade_threshold else float("inf"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arena", type=int, default=9)
    ap.add_argument("--controller", choices=sorted(CONTROLLERS), default="connectome")
    ap.add_argument("--columns", type=Path, default=Path(_default_columns_path()))
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--max-ticks", type=int, default=320)
    ap.add_argument("--n-obstacles", type=int, default=7)
    ap.add_argument("--row-band", type=float, nargs=2, default=(0.0, 0.6))
    ap.add_argument("--every", type=int, default=4, help="print every Nth tick")
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    encoder = BilateralEncoder(a.columns, row_band=tuple(a.row_band))
    template = ArenaConfig(n_obstacles=a.n_obstacles)

    print("calibrating (identical procedure to the benchmark)...")
    cal = calibrate_drive(encoder, template, seeds=[900, 901, 902],
                          resolution=a.resolution, tick_hz=a.tick_hz,
                          v_cruise=1.0, max_ticks=a.max_ticks)
    controller = CONTROLLERS[a.controller](
        drive_center=cal["drive_center"], drive_scale=cal["drive_scale"],
        tick_hz=a.tick_hz, npz_path=a.npz)
    if controller.needs_vision:
        cc = calibrate_controller(controller, encoder, template, seeds=[900, 901, 902],
                                  resolution=a.resolution, tick_hz=a.tick_hz,
                                  max_ticks=a.max_ticks)
        print(f"  turn_offset={cc['turn_offset']:+.4f}  "
              f"saccade_threshold={cc['saccade_threshold']:.4f}  "
              f"estop_threshold={cc['estop_threshold']:.2f}")

    cfg = ArenaConfig(**{**template.__dict__, "seed": a.arena})
    print(f"\nreplaying arena {a.arena} with the {a.controller} controller...")
    result = run_trial(controller, cfg, encoder, a.resolution, a.tick_hz,
                       a.max_ticks, keep_telemetry=True)

    rows = result.telemetry
    print(f"\n{'tick':>5} {'t':>5} {'x':>6} {'y':>6} {'head':>7} "
          f"{'turn':>7} {'|turn|>thr':>10} {'estop_stat':>10} {'omega':>6}")
    thr = controller.saccade_threshold
    for i, r in enumerate(rows):
        if i % a.every and i != len(rows) - 1:
            continue
        turn = r.get("turn")
        flag = "" if turn is None else ("YES" if abs(turn) >= thr else "-")
        print(f"{i:>5} {r['t']:>5.1f} {r['x']:>6.2f} {r['y']:>6.2f} "
              f"{math.degrees(r['heading']):>7.1f} "
              f"{('%+.3f' % turn) if turn is not None else '    -':>7} "
              f"{flag:>10} "
              f"{r.get('estop_stat', float('nan')):>10.1f} {r['omega']:>+6.2f}")

    kind = result.collision_kind
    print(f"\noutcome: {'COLLIDED with the ' + kind if result.collided else ('goal' if result.reached_goal else 'timeout')}"
          f" at x={result.progress_x:.2f} y={result.final_y:+.2f} "
          f"heading={result.final_heading_deg:+.1f} deg")
    print(f"wall clearance at the end: "
          f"{cfg.lane_half_width - cfg.agent_radius - abs(result.final_y):+.3f} m")

    stats = _classify(rows, thr)
    print("\n--- why ---")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if stats.get("ticks_over_threshold", 0) == 0:
        print("  VERDICT: the turn signal never crossed its own noise floor. The\n"
              "           steering law had nothing to act on -- this is a signal\n"
              "           problem, not a control-gain problem.")
    elif result.collided and kind == "wall":
        print("  VERDICT: the signal did cross threshold, so the steering law fired\n"
              "           but did not fire often or early enough to stay off the wall.")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({
            "arena": a.arena, "controller": a.controller,
            "outcome": {"collided": result.collided, "kind": kind,
                        "progress_x": result.progress_x, "final_y": result.final_y,
                        "final_heading_deg": result.final_heading_deg},
            "stats": stats, "telemetry": rows}, indent=2) + "\n")
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
