"""Measure a `Calibration` for a real camera.

    python -m flyguard.calibrate \
        --clear clear_run.npz --obstacle obstacle_run.npz \
        --columns column_assignment.csv --out my_camera.json

You supply two recordings taken with the vehicle moving forward at its
intended cruise speed:

* `--clear`     -- open space, nothing worth avoiding ahead;
* `--obstacle`  -- the same motion toward and past real obstacles.

Neither needs a label per frame and neither needs ground truth. What they
have to be is *representative*: the same camera, the same resolution, the
same speed and the same kind of scene the robot will actually fly in.

## Why two recordings and not one

Every constant here is a threshold, and a threshold needs a reference. The
split between them is not arbitrary -- it was measured, and getting it wrong
broke the controller in both directions (CLAUDE.md records both failures):

* `drive_center`, `turn_offset` and `saccade_threshold` come from the
  **clear** recording, because with nothing ahead every signal the pipeline
  produces is by construction noise or bias, which is exactly what those
  three quantities are. Measuring the saccade threshold on obstacle footage
  counts genuine obstacle responses as noise and puts it an order of
  magnitude too high (0.30-0.43 against a true floor of ~0.02); the vehicle
  then almost never turns.

* `drive_scale` and `estop_threshold` come from the **obstacle** recording.
  Escape has to mean "closer than almost any moment of ordinary travel", not
  "something is visible at all". Referenced to clear footage it fired on 87%
  of ticks and left the vehicle permanently braking.

## Backends calibrate separately, on purpose

The connectome backend's turn signal is a ratio of LPLC2 population rates and
the flow backend's is a ratio of pooled drives. They do not share a scale, so
each gets its own calibration -- which is also what keeps the comparison
between them fair, since every other gain stays identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyguard.runtime import BilateralEncoder, Calibration, FlyGuardPilot
from flyguard.runtime.types import SIDES

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_frames(path: Path) -> np.ndarray:
    """Load a frame sequence from `.npy`, `.npz` or a directory of images.

    `.npz` is searched for a "frames" key, else its single array. A directory
    is read in sorted filename order, which is why zero-padded names matter.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"{path} contains no images matching {IMAGE_SUFFIXES}")
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "reading a directory of images needs Pillow (`pip install pillow`). "
                "Alternatively save the sequence as a .npy array of shape (T, H, W[, 3])."
            ) from exc
        return np.stack([np.asarray(Image.open(f)) for f in files])

    if path.suffix == ".npz":
        z = np.load(path)
        key = "frames" if "frames" in z else list(z.keys())[0]
        return z[key]
    return np.load(path)


def _percepts(frames: np.ndarray, encoder: BilateralEncoder) -> list:
    """Encode every consecutive frame pair. N frames give N-1 percepts."""
    if len(frames) < 2:
        raise ValueError(f"need at least 2 frames to estimate flow, got {len(frames)}")
    return [encoder(frames[i - 1], frames[i]) for i in range(1, len(frames))]


def _replay(pilot: FlyGuardPilot, percepts: list, tick_hz: float) -> tuple[list, list]:
    """Push percepts through the pilot with both thresholds at infinity, so
    nothing fires and the telemetry is the raw signal rather than the
    controller's reaction to it."""
    pilot.steering.saccade_threshold = float("inf")
    pilot.steering.estop_threshold = float("inf")
    pilot.steering.turn_offset = 0.0
    pilot.reset()
    turns, stats = [], []
    for i, p in enumerate(percepts):
        cmd = pilot.act(p, i / tick_hz)
        turns.append(cmd.telemetry["turn_raw"])
        stats.append(cmd.telemetry["estop_stat"])
    return turns, stats


def measure(
    clear_frames: np.ndarray,
    obstacle_frames: np.ndarray,
    encoder: BilateralEncoder,
    *,
    backend: str = "connectome",
    columns_csv: Path | None = None,
    npz_path: Path | None = None,
    tick_hz: float = 10.0,
    estop_percentile: float = 99.0,
    saccade_percentile: float = 99.0,
    drive_percentile: float = 90.0,
    seed: int = 1,
) -> tuple[Calibration, dict]:
    """Returns the measured `Calibration` and a report of the raw statistics.

    The report is not decoration: a calibration whose clear and obstacle
    distributions overlap completely is telling you the pipeline cannot see
    your obstacles, and that is worth knowing before the robot moves.
    """
    clear = _percepts(clear_frames, encoder)
    obstacle = _percepts(obstacle_frames, encoder)

    # -- drive centre from clear footage, scale from obstacle footage -------
    centers, deviations = {}, []
    for side in SIDES:
        clear_side = np.array([p[f"drive_{side}"] for p in clear])
        obst_side = np.array([p[f"drive_{side}"] for p in obstacle])
        centers[side] = float(np.median(clear_side))
        deviations.append(np.abs(obst_side - centers[side]))
    # One shared half-range, set so ~10% of obstacle samples clip. A narrower
    # band spends most of its time pinned at 0 or 1 and throws the left-right
    # difference away; a much wider one leaves LPi's (1 - drive) channel with
    # no contrast.
    drive_scale = float(np.percentile(np.concatenate(deviations), drive_percentile))
    if drive_scale <= 0:
        raise ValueError(
            "measured drive scale is zero: the obstacle recording produces the "
            "same pooled drive as the clear one. Either the two recordings are "
            "of the same scene, or the flow estimate is not picking up your "
            "obstacles (textureless surfaces defeat Lucas-Kanade outright)."
        )

    base = Calibration(drive_center=centers, drive_scale=drive_scale,
                       encoder=encoder.settings())
    pilot = FlyGuardPilot(columns_csv or Path("."), base, backend=backend,
                          npz_path=npz_path, tick_hz=tick_hz, encoder=encoder,
                          seed=seed)

    # -- turn zero-point and noise floor from the clear recording -----------
    clear_turns, clear_stats = _replay(pilot, clear, tick_hz)
    turn_offset = float(np.median(clear_turns))
    centred = np.abs(np.array(clear_turns) - turn_offset)
    saccade_threshold = float(np.percentile(centred, saccade_percentile))

    # -- escape threshold from the obstacle recording -----------------------
    _, obstacle_stats = _replay(pilot, obstacle, tick_hz)
    estop_threshold = float(np.percentile(obstacle_stats, estop_percentile))
    # If the statistic saturates, a percentile can land exactly on the ceiling
    # and would then fire on every clipped tick. Require it strictly above the
    # median of the same distribution.
    estop_threshold = max(estop_threshold, float(np.median(obstacle_stats)) + 1e-6)

    calibration = Calibration(
        drive_center=centers,
        drive_scale=drive_scale,
        turn_offset=turn_offset,
        saccade_threshold=saccade_threshold,
        estop_threshold=estop_threshold,
        encoder=encoder.settings(),
        source={
            "backend": backend,
            "tick_hz": tick_hz,
            "n_clear_pairs": len(clear),
            "n_obstacle_pairs": len(obstacle),
            "frame_shape": list(np.shape(clear_frames)[1:]),
            "estop_percentile": estop_percentile,
            "saccade_percentile": saccade_percentile,
            "drive_percentile": drive_percentile,
        },
    )
    report = {
        "drive_center": centers,
        "side_offset": float(centers["right"] - centers["left"]),
        "drive_scale": drive_scale,
        "turn_offset": turn_offset,
        "turn_centred_median": float(np.median(centred)),
        "saccade_threshold": saccade_threshold,
        "estop_threshold": estop_threshold,
        "estop_stat_median": float(np.median(obstacle_stats)),
        "estop_stat_max": float(np.max(obstacle_stats)),
        "clear_stat_median": float(np.median(clear_stats)),
    }
    return calibration, report


def _warn_on_weak_separation(report: dict) -> list[str]:
    """Flag the ways a calibration can be arithmetically valid and useless.

    Every one of these has actually happened in this project, and each is
    silent -- the constants come out finite and plausible-looking either way.
    """
    warnings = []
    if report["estop_threshold"] <= report["clear_stat_median"]:
        warnings.append(
            "the escape threshold sits at or below the level clear footage "
            "already produces -- it will fire constantly and the vehicle will "
            "brake permanently. The absolute drive level is not separating "
            "your obstacles from open space."
        )
    separation = report["estop_stat_median"] - report["clear_stat_median"]
    if separation <= 0:
        warnings.append(
            f"the escape statistic does not separate the two recordings at all "
            f"(obstacle median {report['estop_stat_median']:.2f} vs clear "
            f"{report['clear_stat_median']:.2f}). With the connectome backend "
            f"this is the expected DNp01 saturation: 2 Giant Fiber neurons, a "
            f"2.2 ms refractory period and a 100 ms tick cap the count near 90, "
            f"and 231 incoming edges carrying summed weight 4257 against a "
            f"~26-synapse threshold pin it there. In a point-neuron model the "
            f"Giant Fiber is a binary alarm: it can encode 'something', never "
            f"'how close'. Braking will still help -- it buys reaction time -- "
            f"but do not read the escape channel as a distance estimate."
        )
    if report["saccade_threshold"] >= 0.9:
        warnings.append(
            "the saccade threshold is near its maximum, so the turn signal in "
            "clear footage is already swinging nearly full-scale. That is the "
            "signature of rotational flow swamping the encoder -- check the "
            "clear recording was taken travelling straight."
        )
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clear", type=Path, required=True,
                    help="frames of forward motion through open space (.npy/.npz/dir)")
    ap.add_argument("--obstacle", type=Path, required=True,
                    help="frames of the same motion toward real obstacles")
    ap.add_argument("--columns", type=Path, required=True,
                    help="FlyWire column_assignment.csv")
    ap.add_argument("--out", type=Path, default=Path("calibration.json"))
    ap.add_argument("--backend", choices=("connectome", "flow"), default="connectome")
    ap.add_argument("--npz", type=Path, default=None, help="subnetwork .npz")
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--mirror", choices=("none", "anatomical"), default="anatomical")
    ap.add_argument("--row-band", type=float, nargs=2, default=(0.0, 0.6),
                    metavar=("TOP", "BOTTOM"),
                    help="keep only this vertical slice of the image; 0 1 uses all of it")
    ap.add_argument("--report-out", type=Path, default=None)
    a = ap.parse_args()

    encoder = BilateralEncoder(a.columns, mirror=a.mirror, row_band=tuple(a.row_band))
    print(f"T4/T5 columns: left {encoder.n_columns('left')}, "
          f"right {encoder.n_columns('right')} (mirror={a.mirror})")

    clear = load_frames(a.clear)
    obstacle = load_frames(a.obstacle)
    print(f"clear:    {len(clear)} frames {np.shape(clear)[1:]}")
    print(f"obstacle: {len(obstacle)} frames {np.shape(obstacle)[1:]}")
    if np.shape(clear)[1:] != np.shape(obstacle)[1:]:
        raise SystemExit(
            "the two recordings have different frame shapes; a calibration is "
            "only valid for one camera at one resolution"
        )

    calibration, report = measure(clear, obstacle, encoder, backend=a.backend,
                                  columns_csv=a.columns, npz_path=a.npz,
                                  tick_hz=a.tick_hz)

    print(f"\ndrive centre:      left={report['drive_center']['left']:.4f} "
          f"right={report['drive_center']['right']:.4f} "
          f"(hemisphere offset {report['side_offset']:+.4f})")
    print(f"drive scale:       {report['drive_scale']:.4f} (shared)")
    print(f"turn offset:       {report['turn_offset']:+.4f}")
    print(f"saccade threshold: {report['saccade_threshold']:.4f} "
          f"(clear centred median {report['turn_centred_median']:.4f})")
    print(f"escape threshold:  {report['estop_threshold']:.2f} "
          f"(obstacle median {report['estop_stat_median']:.2f}, "
          f"clear median {report['clear_stat_median']:.2f})")

    for warning in _warn_on_weak_separation(report):
        print(f"\nWARNING: {warning}")

    calibration.save(a.out)
    print(f"\nwrote {a.out}")
    if a.report_out:
        a.report_out.parent.mkdir(parents=True, exist_ok=True)
        a.report_out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {a.report_out}")


if __name__ == "__main__":
    main()
