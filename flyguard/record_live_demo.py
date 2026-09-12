"""Records a synchronized demo (rendered scene + spike raster + /cmd_vel)
for the visual simulation the project owner asked for -- and for README
GIF material.

Reuses the *exact same circuit logic* as `ros2_ws/src/flyguard_ros2/
{vision_node,looming_node}.py` (core circuit, real optical-flow encoder,
dual-channel LPLC2/LPi drive, DNp01 e-stop latch, the same
`DRIVE_SUM_LOW`/`DRIVE_SUM_HIGH` calibration) but runs it in one Python
process instead of over ROS2 topics, purely to capture a rich multi-modal
recording for offline visualization -- the ROS2 nodes already
independently prove the live messaging wiring works (see CLAUDE.md "Pilot
findings"). If this script's logic and the ROS2 nodes' logic ever drift
apart, the recording stops being an honest representation of what the
live system does; keep them in sync by hand until there's a shared module
to factor them into.

Output: a single JSON bundle with base64 PNG frames, per-sampled-neuron
spike events (real spike times, not a smoothed rate), and per-tick
drive/estop/cmd_vel arrays, all on one shared timeline -- everything an
HTML player needs, nothing it has to compute.

Usage:
    MUJOCO_GL=egl python -m flyguard.record_live_demo --out demo_recording.json
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

from flyguard.paths import default_columns_path
from flyguard.encoder import column_pixel_bounds, encode_drive, flow_from_frame_pair, load_column_assignment
from flyguard.mujoco_world import SCENE_XML
from flyguard.runtime import CoreCircuit
from flyguard.runtime.circuit import BASE_CORE_TYPES

import mujoco  # noqa: E402

CORE_TYPES = list(BASE_CORE_TYPES)
# Same empirical two-point calibration as vision_node.py -- see CLAUDE.md
# "Pilot findings" for why a single-ended normaliser broke the live e-stop.
DRIVE_SUM_LOW = 400.0
DRIVE_SUM_HIGH = 570.0


def _default_columns_path() -> str:
    """See `flyguard.paths` -- searches $FLYGUARD_COLUMNS, the working
    directory, ~/flywire/ and the packaged data dir, in that order."""
    return default_columns_path()


def _scene_xy(t, cruise_s, loom_s, depth0, radius, loom_speed):
    """Same trajectory math as vision_node.VisionNode._scene_xy."""
    cycle = cruise_s + loom_s
    phase = t % cycle
    if phase < cruise_s:
        half_width = depth0 * 0.4142
        max_excursion = 0.6 * half_width
        speed = (2 * max_excursion) / cruise_s
        y = speed * phase - speed * cruise_s / 2.0
        return depth0, y
    loom_t = phase - cruise_s
    x = max(depth0 - loom_speed * loom_t, radius * 1.5)
    return x, 0.0


def frame_to_png_b64(frame: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def sample_neuron_ids(groups_c: dict, n_lplc2: int, n_lpi: int) -> dict:
    """Pick a small, fixed, reproducible sample of real neuron indices per
    group for the raster -- recording *every* spike from all 530 neurons
    at 0.1ms resolution would be a huge, visually unreadable amount of
    data; DNp01 (2 neurons) is kept in full since it's the e-stop signal
    itself and there are only 2 of them."""
    rng = np.random.default_rng(0)
    lplc2_idx = groups_c["LPLC2"]
    lpi_types = [t for t in groups_c if str(t).startswith("LPi")]
    lpi_idx = np.concatenate([groups_c[t] for t in lpi_types]) if lpi_types else np.array([], dtype=int)
    dnp01_idx = groups_c["DNp01"]

    sampled_lplc2 = rng.choice(lplc2_idx, size=min(n_lplc2, len(lplc2_idx)), replace=False)
    sampled_lpi = rng.choice(lpi_idx, size=min(n_lpi, len(lpi_idx)), replace=False) if len(lpi_idx) else lpi_idx
    return {"LPLC2": sampled_lplc2, "LPi": sampled_lpi, "DNp01": dnp01_idx}


def main(out_path: Path, columns_csv: Path, side: str, resolution: int,
         duration_s: float, tick_hz: float, cruise_s: float, loom_s: float,
         depth0: float, radius: float, loom_speed: float,
         base_rate_hz: float, peak_weight: float, estop_cooldown_s: float, seed: int):
    print("loading real connectome core circuit (LPLC2+LC4+LPi+DNp01)...")
    # Same object the ROS2 node and the closed-loop benchmark run, so this
    # recording stays an honest picture of the live system rather than a
    # parallel implementation that has to be kept in step by hand.
    circuit = CoreCircuit(Path("data/looming.npz"), tick_hz=tick_hz,
                          base_rate_hz=base_rate_hz, peak_weight=peak_weight,
                          seed=seed, record=("LPLC2", "LPi", "DNp01"))
    print(circuit.describe())

    groups_c = {pop: circuit.indices(pop) for pop in ("LPLC2", "LPi", "DNp01")}
    sampled = sample_neuron_ids(groups_c, n_lplc2=24, n_lpi=16)
    # map sampled indices -> a stable 0..N-1 row id per group, for the raster
    row_id = {}
    for group, idx in sampled.items():
        for row, neuron_idx in enumerate(idx):
            row_id[int(neuron_idx)] = (group, row)

    print("loading T4/T5 columns...")
    columns = load_column_assignment(columns_csv, side=side)
    x_bounds, y_bounds = column_pixel_bounds(columns)

    record_idx = circuit.record_index
    dt = circuit.params.dt

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=resolution, width=resolution)

    frames_b64 = []
    drive_series, estop_series, cmdvel_series, time_series = [], [], [], []
    spike_events = []  # list of {t, group, row}

    prev_frame = None
    estop_until = -1.0
    n_ticks = int(duration_s * tick_hz)
    print(f"recording {n_ticks} ticks ({duration_s}s @ {tick_hz} Hz)...")

    try:
        for tick in range(n_ticks):
            t = tick / tick_hz
            x, y = _scene_xy(t, cruise_s, loom_s, depth0, radius, loom_speed)
            data.mocap_pos[0] = [x, y, 0.0]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="eye")
            frame = renderer.render()
            frames_b64.append(frame_to_png_b64(frame))

            if prev_frame is not None:
                flow_vx, flow_vy = flow_from_frame_pair(prev_frame, frame, columns, x_bounds, y_bounds)
                drive_sum = float(encode_drive(columns, flow_vx, flow_vy).sum())
            else:
                drive_sum = DRIVE_SUM_LOW
            prev_frame = frame

            span = DRIVE_SUM_HIGH - DRIVE_SUM_LOW
            drive = float(np.clip((drive_sum - DRIVE_SUM_LOW) / span, 0.0, 1.0))
            response = circuit.step(drive)
            spikes = response.spikes
            # spikes columns follow circuit.record_index == [LPLC2..., LPi..., DNp01...]
            fired_steps, fired_cols = np.nonzero(spikes)
            for step, col in zip(fired_steps, fired_cols):
                neuron_idx = int(record_idx[col])
                if neuron_idx in row_id:
                    group, row = row_id[neuron_idx]
                    spike_events.append({"t": round(t + step * dt, 4), "group": group, "row": row})

            dnp01_fired = response.any_spike("DNp01")
            if dnp01_fired:
                estop_until = t + estop_cooldown_s
            estop_active = t < estop_until

            drive_series.append(round(drive, 3))
            estop_series.append(bool(estop_active))
            cmdvel_series.append(0.0 if estop_active else 0.3)
            time_series.append(round(t, 4))

            if tick % 10 == 0:
                print(f"  tick {tick}/{n_ticks}  t={t:.2f}s  drive={drive:.2f}  estop={estop_active}")
    finally:
        renderer.close()

    bundle = {
        "meta": {
            "n_ticks": n_ticks, "tick_hz": tick_hz, "resolution": resolution,
            "cruise_s": cruise_s, "loom_s": loom_s,
            "n_lplc2_sampled": len(sampled["LPLC2"]), "n_lpi_sampled": len(sampled["LPi"]),
            "n_dnp01": len(sampled["DNp01"]),
        },
        "time": time_series,
        "frames_png_b64": frames_b64,
        "drive": drive_series,
        "estop": estop_series,
        "cmd_vel_linear_x": cmdvel_series,
        "spike_events": spike_events,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path} ({size_mb:.2f} MB), {len(spike_events)} spike events recorded")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("demo_recording.json"))
    ap.add_argument("--columns", type=Path, default=Path(_default_columns_path()))
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--resolution", type=int, default=112)
    ap.add_argument("--duration-s", type=float, default=12.0)
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--cruise-s", type=float, default=4.0)
    ap.add_argument("--loom-s", type=float, default=2.0)
    ap.add_argument("--depth0", type=float, default=5.0)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--loom-speed", type=float, default=2.0)
    ap.add_argument("--base-rate-hz", type=float, default=250.0)
    ap.add_argument("--peak-weight", type=float, default=30.0)
    ap.add_argument("--estop-cooldown-s", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    main(a.out, a.columns, a.side, a.resolution, a.duration_s, a.tick_hz,
         a.cruise_s, a.loom_s, a.depth0, a.radius, a.loom_speed,
         a.base_rate_hz, a.peak_weight, a.estop_cooldown_s, a.seed)
