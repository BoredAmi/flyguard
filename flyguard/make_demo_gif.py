"""Render a recorded closed-loop run as an animated GIF.

    python -m flyguard.make_demo_gif --recording data/avoid_recording.json \
        --out docs/corridor_run.gif

    python -m flyguard.make_demo_gif --compare data/rec_flow.json data/rec_conn.json \
        --out docs/flow_vs_connectome.gif

Why a GIF and not a hosted page: GitHub renders an animated GIF inline in the
README, so the run plays where someone is already reading, with no link to
follow, no external host to depend on, and nothing that can rot. The
interactive version (`make_run_page.py` -> `docs/corridor_run.html`) is
self-contained too and opens from the filesystem.

Each frame is composited from the recording rather than re-simulated, so what
you see is the trial the benchmark actually measured -- the agent's own
128x128 eye view on the left, a top-down corridor map on the right, and the
telemetry that produced every steering decision. Nothing is redrawn from a
smoothed summary.

`--compare` stacks two runs of the *same arena* on one timeline, which is the
only fair way to show the headline result: the flow baseline and the 530-neuron
circuit see identical obstacles under identical gains, and the difference
between the two strips is the circuit. Runs end at different times, so the
shorter one freezes on its final frame rather than disappearing -- a run that
vanished would read as "the recording stopped", not "the robot crashed".

Pillow only (already the `images` extra); no matplotlib, no ffmpeg.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

# Dark instrument-console palette, matching the HTML player.
BG = (12, 15, 22)
PANEL = (20, 25, 35)
GRID = (38, 46, 62)
WALL = (70, 82, 105)
OBSTACLE = (196, 88, 72)
AGENT = (232, 176, 84)
TRAIL = (94, 116, 150)
TEXT = (150, 165, 190)
TEXT_BRIGHT = (226, 232, 240)
BRAKE = (214, 96, 80)
SACCADE = (120, 190, 220)


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _map_projection(arena: dict, width: int, height: int, pad: int):
    """World (x, y) -> pixel, with x running left-to-right along the corridor."""
    goal_x = float(arena["goal_x"])
    half = float(arena["lane_half_width"])
    sx = (width - 2 * pad) / goal_x
    sy = (height - 2 * pad) / (2 * half)
    scale = min(sx, sy)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (pad + x * scale, height / 2 - y * scale)

    return to_px, scale


def render_frame(frame: Image.Image, arena: dict, telemetry: list, i: int,
                 meta: dict, eye_px: int, map_w: int, map_h: int,
                 label: str | None = None, frozen: bool = False) -> Image.Image:
    pad = 10
    gap = 10
    W = eye_px + gap + map_w
    H = max(eye_px, map_h) + 34
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # -- the agent's own view ------------------------------------------------
    canvas.paste(frame.resize((eye_px, eye_px), Image.NEAREST), (0, 0))
    draw.rectangle([0, 0, eye_px - 1, eye_px - 1], outline=GRID)
    # The encoder only reads the upper band; show where the cut falls.
    band = meta.get("row_band", [0.0, 0.6])
    y0, y1 = int(eye_px * band[0]), int(eye_px * band[1])
    draw.line([(0, y1), (eye_px - 1, y1)], fill=SACCADE, width=1)
    draw.line([(eye_px // 2, y0), (eye_px // 2, y1)], fill=(90, 100, 125), width=1)
    draw.text((4, 3), "eye view", fill=TEXT)
    draw.text((4, y1 + 3), "ground: not read", fill=(95, 108, 132))

    # -- top-down corridor ---------------------------------------------------
    ox = eye_px + gap
    draw.rectangle([ox, 0, ox + map_w - 1, map_h - 1], fill=PANEL, outline=GRID)
    to_px, scale = _map_projection(arena, map_w, map_h, pad)
    half = float(arena["lane_half_width"])

    for sign in (1, -1):
        _, wy = to_px(0.0, sign * half)
        draw.line([(ox + pad, wy), (ox + map_w - pad, wy)], fill=WALL, width=2)
    gx, _ = to_px(float(arena["goal_x"]), 0.0)
    draw.line([(ox + gx, pad), (ox + gx, map_h - pad)], fill=(120, 150, 110), width=2)

    r = max(2.0, float(arena["obstacle_radius"]) * scale)
    for obstacle in arena["obstacles"]:
        cx, cy = to_px(float(obstacle[0]), float(obstacle[1]))
        draw.ellipse([ox + cx - r, cy - r, ox + cx + r, cy + r], fill=OBSTACLE)

    trail = [to_px(t["x"], t["y"]) for t in telemetry[: i + 1]]
    if len(trail) > 1:
        draw.line([(ox + px, py) for px, py in trail], fill=TRAIL, width=2)

    row = telemetry[i]
    ax, ay = to_px(row["x"], row["y"])
    ar = max(3.0, float(arena["agent_radius"]) * scale)
    colour = BRAKE if row.get("estop") else AGENT
    draw.ellipse([ox + ax - ar, ay - ar, ox + ax + ar, ay + ar], fill=colour)
    heading = float(row["heading"])
    draw.line([(ox + ax, ay),
               (ox + ax + math.cos(heading) * ar * 2.6,
                ay - math.sin(heading) * ar * 2.6)], fill=TEXT_BRIGHT, width=2)

    # -- status line ---------------------------------------------------------
    y = max(eye_px, map_h) + 6
    state = "BRAKING" if row.get("estop") else ("saccade" if abs(row.get("omega", 0)) > 1e-6
                                                else "cruise")
    if frozen:
        outcome = meta.get("outcome", "")
        state = {"collision": "COLLIDED", "goal": "REACHED GOAL"}.get(outcome, outcome.upper())
    draw.text((2, y), f"t={row['t']:5.1f}s   x={row['x']:5.1f}/{arena['goal_x']:.0f} m",
              fill=TEXT_BRIGHT)
    name = label if label is not None else meta.get("controller", "?")
    draw.text((eye_px + gap, y), f"{name}  |  {state}",
              fill=BRAKE if (row.get("estop") or frozen) else TEXT)
    return canvas


def build(recording: Path, out: Path, *, step: int, eye_px: int, map_w: int,
          map_h: int, fps: float, loop_pause_ms: int) -> Path:
    data = json.loads(recording.read_text())
    frames_b64 = data["frames_jpeg_b64"]
    telemetry = data["telemetry"]
    arena = data["arena"]
    meta = data["meta"]

    n = min(len(frames_b64), len(telemetry))
    indices = list(range(0, n, step))
    frames = [render_frame(_decode(frames_b64[i]), arena, telemetry, i, meta,
                           eye_px, map_w, map_h) for i in indices]
    if not frames:
        raise ValueError(f"{recording} produced no frames")
    return _write_gif(frames, out, fps, loop_pause_ms,
                      note=(f"{meta.get('controller')} on arena {meta.get('arena_seed')}: "
                            f"{meta.get('outcome')} at {meta.get('progress_x')}"
                            f"/{meta.get('goal_x')} m  ({len(frames)} of {n} ticks)"))


def build_comparison(recordings: list[Path], out: Path, *, step: int, eye_px: int,
                     map_w: int, map_h: int, fps: float, loop_pause_ms: int,
                     labels: list[str] | None = None) -> Path:
    """Stack several runs of the same arena into one animation."""
    data = [json.loads(r.read_text()) for r in recordings]
    arenas = {json.dumps(d["arena"]["obstacles"], sort_keys=True) for d in data}
    if len(arenas) > 1:
        raise ValueError(
            "the recordings are of different arenas, so stacking them would "
            "invite a comparison that is not being made. Record each "
            "controller with the same --arena seed."
        )

    labels = labels or [d["meta"].get("controller", "?") for d in data]
    lengths = [min(len(d["frames_jpeg_b64"]), len(d["telemetry"])) for d in data]
    longest = max(lengths)
    indices = list(range(0, longest, step))

    strips = []
    for d, n, label in zip(data, lengths, labels):
        frames = []
        for i in indices:
            # Freeze a finished run on its last frame; a strip that vanished
            # would read as a recording glitch rather than an outcome.
            j = min(i, n - 1)
            frames.append(render_frame(_decode(d["frames_jpeg_b64"][j]), d["arena"],
                                       d["telemetry"], j, d["meta"], eye_px, map_w,
                                       map_h, label=label, frozen=i >= n))
        strips.append(frames)

    W = strips[0][0].size[0]
    H = sum(s[0].size[1] for s in strips) + 2 * (len(strips) - 1)
    composed = []
    for k in range(len(indices)):
        canvas = Image.new("RGB", (W, H), GRID)
        y = 0
        for strip in strips:
            canvas.paste(strip[k], (0, y))
            y += strip[k].size[1] + 2
        composed.append(canvas)

    return _write_gif(composed, out, fps, loop_pause_ms,
                      note=" vs ".join(labels) + f" on arena {data[0]['meta'].get('arena_seed')}")


def _write_gif(frames: list, out: Path, fps: float, loop_pause_ms: int,
               note: str = "") -> Path:
    if not frames:
        raise ValueError("no frames to write")
    # One shared palette; per-frame palettes make the animation shimmer.
    palette_source = frames[len(frames) // 2].quantize(colors=128, method=Image.MEDIANCUT)
    quantised = [f.quantize(palette=palette_source, dither=Image.NONE) for f in frames]
    durations = [int(1000 / fps)] * len(quantised)
    durations[-1] = loop_pause_ms
    out.parent.mkdir(parents=True, exist_ok=True)
    quantised[0].save(out, save_all=True, append_images=quantised[1:],
                      duration=durations, loop=0, optimize=True, disposal=2)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(quantised)} frames, {frames[0].size[0]}x"
          f"{frames[0].size[1]}, {size_mb:.2f} MB)")
    if note:
        print(f"  {note}")
    if size_mb > 10:
        print("  WARNING: over 10 MB; raise --step or lower --eye-px.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recording", type=Path, default=Path("data/avoid_recording.json"))
    ap.add_argument("--compare", type=Path, nargs="+", default=None,
                    metavar="RECORDING",
                    help="stack two or more runs of the SAME arena into one "
                         "animation, which is how the flow-vs-connectome "
                         "comparison is drawn")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="strip captions for --compare (default: controller names)")
    ap.add_argument("--out", type=Path, default=Path("docs/corridor_run.gif"))
    ap.add_argument("--step", type=int, default=2,
                    help="keep every Nth tick (2 = half the frames, twice as small)")
    ap.add_argument("--eye-px", type=int, default=192)
    ap.add_argument("--map-w", type=int, default=384)
    ap.add_argument("--map-h", type=int, default=150)
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--loop-pause-ms", type=int, default=1200)
    a = ap.parse_args()
    kw = dict(step=a.step, eye_px=a.eye_px, map_w=a.map_w, map_h=a.map_h,
              fps=a.fps, loop_pause_ms=a.loop_pause_ms)
    if a.compare:
        build_comparison(a.compare, a.out, labels=a.labels, **kw)
    else:
        build(a.recording, a.out, **kw)


if __name__ == "__main__":
    main()
