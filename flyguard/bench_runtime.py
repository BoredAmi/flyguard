"""Measure the per-tick cost of the robot runtime.

    python -m flyguard.bench_runtime --columns column_assignment.csv

Answers the only question that decides whether this can sit in a control loop:
how much of the tick budget does each stage eat, on one CPU core, with no GPU?

The two stages have very different scaling, which is why they are timed apart
rather than as one number:

* **perception** scales with pixel count, so halving the resolution roughly
  quarters it;
* **the circuit** scales with simulated time and neuron count and is completely
  independent of image size -- a 100 ms tick costs the same whether the camera
  is 64x64 or 4K.

So a loop that is too slow is fixed by lowering the *resolution* or the *tick
rate*, and knowing which one matters. Frames are synthetic texture here: the
timings depend on image size, not on image content.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from flyguard.runtime import BilateralEncoder, Calibration, CoreCircuit, FlyGuardPilot


def _texture(resolution: int, n: int, seed: int = 0) -> np.ndarray:
    """A drifting random-texture sequence: dense flow everywhere, so the
    timings are a worst case rather than a mostly-blank best case."""
    rng = np.random.default_rng(seed)
    field = (rng.random((resolution * 2, resolution * 2)) * 255).astype(np.uint8)
    return np.stack([field[r : r + resolution, r : r + resolution] for r in range(n)])


def _time(fn, n: int) -> float:
    """Median-of-n milliseconds, so one scheduling hiccup does not dominate."""
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples))


def benchmark(columns_csv: Path, resolution: int, tick_hz: float,
              npz_path: Path | None, n_ticks: int) -> dict:
    frames = _texture(resolution, n_ticks + 1)
    encoder = BilateralEncoder(columns_csv)

    counter = {"i": 0}

    def perceive():
        i = counter["i"] % n_ticks
        counter["i"] += 1
        encoder(frames[i], frames[i + 1])

    perception_ms = _time(perceive, n_ticks)

    circuit = CoreCircuit(npz_path, tick_hz=tick_hz, bilateral=True)
    circuit_ms = _time(lambda: circuit.step({"left": 0.5, "right": 0.7}), n_ticks)

    pilot = FlyGuardPilot(columns_csv, Calibration(drive_scale=0.05,
                                                   saccade_threshold=0.1,
                                                   estop_threshold=88.0),
                          npz_path=npz_path, tick_hz=tick_hz, encoder=encoder)
    counter["i"] = 0

    def full():
        i = counter["i"] % n_ticks
        counter["i"] += 1
        pilot.step(frames[i], t=i / tick_hz)

    pilot.step(frames[0], t=0.0)  # prime the frame history
    total_ms = _time(full, n_ticks)

    budget_ms = 1000.0 / tick_hz
    return {
        "resolution": resolution,
        "tick_hz": tick_hz,
        "n_neurons": circuit.n_neurons,
        "perception_ms": perception_ms,
        "circuit_ms": circuit_ms,
        "total_ms": total_ms,
        "budget_ms": budget_ms,
        "realtime_factor": budget_ms / total_ms,
        "headroom": total_ms <= budget_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--columns", type=Path, required=True)
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--resolutions", type=int, nargs="*", default=[64, 96, 128])
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--n-ticks", type=int, default=15)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    rows = [benchmark(a.columns, r, a.tick_hz, a.npz, a.n_ticks) for r in a.resolutions]

    print(f"\n{rows[0]['n_neurons']}-neuron circuit, {a.tick_hz:g} Hz tick "
          f"({rows[0]['budget_ms']:.0f} ms budget), median of {a.n_ticks} ticks\n")
    print(f"{'res':>6} {'perception':>11} {'circuit':>9} {'total':>8} {'vs budget':>10}")
    for r in rows:
        print(f"{r['resolution']:>4}px {r['perception_ms']:>9.1f}ms "
              f"{r['circuit_ms']:>7.1f}ms {r['total_ms']:>6.1f}ms "
              f"{r['realtime_factor']:>9.2f}x" + ("" if r["headroom"] else "  OVER BUDGET"))
    print("\nPerception scales with pixel count; the circuit does not -- it scales\n"
          "with simulated time. Too slow? Lower the resolution first.")

    if a.json_out:
        import json

        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
