"""Does the arena's T4/T5 drive signal actually carry obstacle information?

Run this *before* trusting anything `flyguard.avoid` reports. A closed-loop
controller can look broken for two completely different reasons -- the
circuit mishandles a good signal, or the signal was never there -- and once
the loop is closed the two are very hard to tell apart. This measures the
signal open-loop, on scripted straight-line approaches, where ground truth
(distance to the obstacle) is exactly known.

Four sections, each of which caught something real:

1. **Self-motion floor and obstacle contrast.** An agent moving forward
   through a textured world generates expanding optic flow *everywhere*,
   from its own motion. That is the looming detector's classic blind spot:
   Klapoetke's paradigm is a stationary observer watching a moving object,
   and a robot is neither. Comparing an empty corridor against one obstacle
   dead ahead shows how much of the absolute drive level is actually about
   the obstacle -- and the answer is: not much.
2. **Left/right asymmetry.** Steering depends only on the difference between
   hemispheres, so it is measured separately rather than inferred from (1).
   It is a far better cue than the level, because self-motion flow is
   roughly symmetric and largely cancels in the difference.
3. **Row bands.** How much of the signal, and how much of the noise, comes
   from the ground plane. This is what justifies `BilateralEncoder`'s
   default of discarding the lower 40% of the image.
4. **Sign checks.** Whether the turn command actually points away from a
   wall and away from an obstacle. A sign inversion here is catastrophic and
   invisible in aggregate statistics -- this is the section that found
   Lucas-Kanade collapsing on close, fast-moving surfaces, which made a wall
   the agent was about to scrape read as *emptier* than open corridor.

Both drive variants are reported for (1) and (2) (unit-normalised rectified
cosine, the repo default for synthetic flow, vs magnitude-weighted) because
which one carries the signal on *real* flow is an empirical question -- see
`flyguard.encoder.encode_drive`.

Usage:
    MUJOCO_GL=egl python -m flyguard.validate_arena --columns ~/flywire/column_assignment.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flyguard.arena import AgentState, ArenaConfig, ArenaRenderer, step_agent
from flyguard.avoid import BilateralEncoder, bilateral_turn, _default_columns_path

OBSTACLE_X = 12.0
NEAR_RANGE = (1.0, 4.0)   # the band a controller must actually act in


def approach_scan(encoder: BilateralEncoder, obstacles: np.ndarray, cfg: ArenaConfig,
                   resolution: int, tick_hz: float, v: float, n_ticks: int) -> list[dict]:
    """Drive straight down the corridor, recording drive vs distance."""
    state = AgentState(0.0, 0.0, 0.0)
    prev = None
    rows = []
    with ArenaRenderer(obstacles, cfg, resolution) as renderer:
        for _ in range(n_ticks):
            frame = renderer.render(state)
            if prev is not None:
                p = encoder(prev, frame)
                rows.append({"x": round(state.x, 3), "distance": round(OBSTACLE_X - state.x, 3),
                             **{k: round(val, 5) for k, val in p.items()}})
            prev = frame
            state = step_agent(state, v, 0.0, 1.0 / tick_hz)
    return rows


def _near(rows):
    lo, hi = NEAR_RANGE
    return [r for r in rows if lo <= r["distance"] <= hi]


def _mean_drive(rows):
    return float(np.mean([(r["drive_left"] + r["drive_right"]) / 2 for r in rows]))


def _turns(rows):
    return np.array([bilateral_turn(r["drive_left"], r["drive_right"]) for r in rows])


def _mean_turn(rows):
    return float(_turns(rows).mean())


def _turn_spread(rows):
    """Standard deviation, which is the actual noise floor. The *mean* turn in
    an empty corridor is a bias estimate: it can sit near zero through
    cancellation while the tick-to-tick spread is large, and it is the spread
    that the saccade threshold has to clear."""
    return float(_turns(rows).std())


def section_signal(columns_csv, resolution, tick_hz, v, n_ticks) -> dict:
    cfg = ArenaConfig(n_obstacles=1, first_obstacle_x=OBSTACLE_X)
    scenes = {"empty": np.zeros((0, 2)),
              "ahead": np.array([[OBSTACLE_X, 0.0]]),
              "left": np.array([[OBSTACLE_X, 1.4]]),    # world +y = agent's left
              "right": np.array([[OBSTACLE_X, -1.4]])}
    out = {}
    for variant, mag in (("unit_normalised", False), ("magnitude_weighted", True)):
        enc = BilateralEncoder(columns_csv, magnitude_weighted=mag)
        scans = {name: approach_scan(enc, obs, cfg, resolution, tick_hz, v, n_ticks)
                 for name, obs in scenes.items()}
        floor = _mean_drive(_near(scans["empty"]))
        ahead = _mean_drive(_near(scans["ahead"]))
        summary = {
            "self_motion_floor": floor,
            "obstacle_ahead": ahead,
            "contrast": ahead / (floor + 1e-9),
            "turn_empty": _mean_turn(_near(scans["empty"])),
            "turn_obstacle_left": _mean_turn(_near(scans["left"])),
            "turn_obstacle_right": _mean_turn(_near(scans["right"])),
        }
        out[variant] = summary
        lo, hi = NEAR_RANGE
        print(f"\n=== {variant} ({lo:.0f}-{hi:.0f} m from the obstacle) ===")
        print(f"  absolute level : empty {floor:.4f} -> ahead {ahead:.4f}  "
              f"= {summary['contrast']:.2f}x contrast")
        print(f"  turn signal    : empty {summary['turn_empty']:+.3f}   "
              f"obstacle LEFT {summary['turn_obstacle_left']:+.3f} (want -)   "
              f"obstacle RIGHT {summary['turn_obstacle_right']:+.3f} (want +)")
        ok = summary["turn_obstacle_left"] < 0 < summary["turn_obstacle_right"]
        print(f"  -> side discrimination {'CORRECT' if ok else 'WRONG/ABSENT'}")
    return out


def section_row_bands(columns_csv, resolution, tick_hz, v, n_ticks) -> dict:
    """How much of the signal survives if the ground is excluded."""
    cfg = ArenaConfig(n_obstacles=1, first_obstacle_x=OBSTACLE_X)
    bands = {"full image": (0.0, 1.0), "upper 60%": (0.0, 0.6),
             "upper 40%": (0.0, 0.4), "lower 40%": (0.6, 1.0)}
    scenes = {"empty": np.zeros((0, 2)),
              "left": np.array([[OBSTACLE_X, 1.4]]),
              "right": np.array([[OBSTACLE_X, -1.4]])}
    print("\n=== row bands: turn signal by image region ===")
    print(f"{'band':12s} {'obst left':>10s} {'obst right':>11s} {'signal':>8s} "
          f"{'empty bias':>11s} {'empty sd':>9s} {'SNR':>6s}")
    out = {}
    for label, band in bands.items():
        enc = BilateralEncoder(columns_csv, magnitude_weighted=True, row_band=band)
        scans = {name: _near(approach_scan(enc, obs, cfg, resolution, tick_hz, v, n_ticks))
                 for name, obs in scenes.items()}
        vals = {name: _mean_turn(rows) for name, rows in scans.items()}
        signal = (abs(vals["left"]) + abs(vals["right"])) / 2
        noise = _turn_spread(scans["empty"])
        snr = signal / (noise + 1e-9)
        out[label] = {**vals, "signal": signal, "empty_sd": noise, "snr": snr}
        print(f"{label:12s} {vals['left']:10.3f} {vals['right']:11.3f} {signal:8.3f} "
              f"{vals['empty']:11.3f} {noise:9.3f} {snr:6.1f}")
    return out


def section_sign_checks(columns_csv, resolution, tick_hz, v) -> dict:
    """Does the turn command point away from things? Aggregate accuracy
    statistics cannot see a sign error; this can."""
    enc = BilateralEncoder(columns_csv)
    cfg = ArenaConfig(n_obstacles=0, corridor_length=25.5)
    print("\n=== sign checks: wall centring (empty corridor, heading 0) ===")
    print(f"{'y (m)':>7s} {'turn':>8s}   want")
    out = {"wall": {}, "obstacle": {}}
    ok_all = True
    with ArenaRenderer(np.zeros((0, 2)), cfg, resolution) as r:
        for y in (-2.5, -1.5, 0.0, 1.5, 2.5):
            s0 = AgentState(8.0, y, 0.0)
            s1 = step_agent(s0, v, 0.0, 1.0 / tick_hz)
            p = enc(r.render(s0), r.render(s1))
            turn = bilateral_turn(p["drive_left"], p["drive_right"])
            out["wall"][str(y)] = turn
            if y == 0.0:
                verdict, want = "-", "neutral"
            else:
                good = (turn > 0) if y < 0 else (turn < 0)
                ok_all &= good
                verdict = "OK" if good else "WRONG"
                want = "turn +y" if y < 0 else "turn -y"
            print(f"{y:7.1f} {turn:+8.3f}   {want:8s} {verdict}")

    print("sign checks: obstacle side")
    cfg2 = ArenaConfig(n_obstacles=1, first_obstacle_x=OBSTACLE_X)
    for label, oy, want_negative in (("obstacle LEFT  (+y)", 1.4, True),
                                      ("obstacle RIGHT (-y)", -1.4, False)):
        with ArenaRenderer(np.array([[OBSTACLE_X, oy]]), cfg2, resolution) as r:
            turns = []
            for x in (8.0, 9.0, 10.0):
                s0 = AgentState(x, 0.0, 0.0)
                s1 = step_agent(s0, v, 0.0, 1.0 / tick_hz)
                p = enc(r.render(s0), r.render(s1))
                turns.append(bilateral_turn(p["drive_left"], p["drive_right"]))
        mean = float(np.mean(turns))
        good = (mean < 0) if want_negative else (mean > 0)
        ok_all &= good
        out["obstacle"][label.strip()] = mean
        print(f"  {label}: turn {mean:+.3f}  {'OK' if good else 'WRONG'}")
    out["all_signs_correct"] = bool(ok_all)
    print(f"\nall sign checks {'PASS' if ok_all else 'FAIL'}")
    return out


def main(columns_csv: Path, resolution: int, tick_hz: float, v: float, n_ticks: int,
         json_out: Path | None, sections: list[str]):
    bundle = {}
    if "signal" in sections:
        bundle["signal"] = section_signal(columns_csv, resolution, tick_hz, v, n_ticks)
    if "bands" in sections:
        bundle["row_bands"] = section_row_bands(columns_csv, resolution, tick_hz, v, n_ticks)
    if "signs" in sections:
        bundle["sign_checks"] = section_sign_checks(columns_csv, resolution, tick_hz, v)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\nwrote {json_out}")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=Path, default=Path(_default_columns_path()))
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--v", type=float, default=1.0)
    ap.add_argument("--n-ticks", type=int, default=130)
    ap.add_argument("--sections", nargs="*", default=["signal", "bands", "signs"],
                     choices=["signal", "bands", "signs"])
    ap.add_argument("--json-out", type=Path, default=Path("data/arena_signal.json"))
    a = ap.parse_args()
    main(a.columns, a.resolution, a.tick_hz, a.v, a.n_ticks, a.json_out, a.sections)
