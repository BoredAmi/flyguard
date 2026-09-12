"""Tests for the README animation builder.

The GIF is a published artifact of the repository, so the things worth
guarding are the ones that would silently produce a misleading picture: frames
falling out of step with telemetry, an empty recording passing silently, or a
per-frame palette making the animation flicker.
"""

import base64
import io
import json

import pytest

pytest.importorskip("PIL", reason="Pillow is an optional extra (pip install flyguard[images])")

from PIL import Image  # noqa: E402

from flyguard.make_demo_gif import build, render_frame  # noqa: E402


def _jpeg_b64(value: int, size: int = 16) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (value, value, value)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _recording(n: int = 6) -> dict:
    return {
        "meta": {"controller": "connectome", "arena_seed": 0, "outcome": "collision",
                 "progress_x": 9.5, "goal_x": 29.0, "row_band": [0.0, 0.6]},
        "arena": {"obstacles": [[6.0, 0.5], [9.5, -1.2]], "obstacle_radius": 0.45,
                  "agent_radius": 0.35, "lane_half_width": 4.0, "goal_x": 29.0},
        "frames_jpeg_b64": [_jpeg_b64(20 + 10 * i) for i in range(n)],
        "telemetry": [{"t": i / 10, "x": float(i), "y": 0.1 * i, "heading": 0.0,
                       "v": 1.0, "omega": 0.0, "estop": i % 3 == 0} for i in range(n)],
    }


@pytest.fixture
def recording_file(tmp_path):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(_recording()))
    return p


def test_builds_a_looping_animation(tmp_path, recording_file):
    out = build(recording_file, tmp_path / "run.gif", step=1, eye_px=32,
                map_w=64, map_h=32, fps=10, loop_pause_ms=500)
    assert out.exists()
    with Image.open(out) as im:
        assert im.n_frames == 6
        assert im.info.get("loop") == 0


def test_step_subsamples_frames(tmp_path, recording_file):
    out = build(recording_file, tmp_path / "half.gif", step=2, eye_px=32,
                map_w=64, map_h=32, fps=10, loop_pause_ms=500)
    with Image.open(out) as im:
        assert im.n_frames == 3


def test_the_background_does_not_shift_between_frames(tmp_path, recording_file):
    """Quantising each frame against its own palette makes the whole image
    shimmer, which reads as a rendering bug in the simulation rather than in
    the encoder. A GIF stores one global palette and later frames inherit it,
    so the observable guarantee is that a pixel which never changes really
    never changes.
    """
    out = build(recording_file, tmp_path / "pal.gif", step=1, eye_px=32,
                map_w=64, map_h=32, fps=10, loop_pause_ms=500)
    corners = []
    with Image.open(out) as im:
        w, h = im.size
        for i in range(im.n_frames):
            im.seek(i)
            corners.append(im.convert("RGB").getpixel((w - 2, h - 2)))
    assert len(set(corners)) == 1, f"background drifts across frames: {set(corners)}"


def test_last_frame_holds_before_looping(tmp_path, recording_file):
    """The outcome is the point of the animation; it must not flash past."""
    out = build(recording_file, tmp_path / "hold.gif", step=1, eye_px=32,
                map_w=64, map_h=32, fps=10, loop_pause_ms=900)
    with Image.open(out) as im:
        im.seek(im.n_frames - 1)
        assert im.info["duration"] >= 800


def test_an_empty_recording_is_rejected(tmp_path):
    p = tmp_path / "empty.json"
    data = _recording()
    data["frames_jpeg_b64"], data["telemetry"] = [], []
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no frames"):
        build(p, tmp_path / "x.gif", step=1, eye_px=32, map_w=64, map_h=32,
              fps=10, loop_pause_ms=500)


def test_frames_are_truncated_to_the_shorter_of_the_two_streams(tmp_path):
    """Frames and telemetry are recorded separately; a mismatch must not index
    past the end, and must not silently pair frame i with tick i+k."""
    data = _recording(n=6)
    data["telemetry"] = data["telemetry"][:4]
    p = tmp_path / "ragged.json"
    p.write_text(json.dumps(data))
    out = build(p, tmp_path / "r.gif", step=1, eye_px=32, map_w=64, map_h=32,
                fps=10, loop_pause_ms=500)
    with Image.open(out) as im:
        assert im.n_frames == 4


def test_braking_is_drawn_differently_from_cruising():
    """The brake state is the one anatomical signal in the picture, so it has
    to be visible rather than implied."""
    data = _recording()
    frame = Image.new("RGB", (16, 16), (30, 30, 30))
    kw = dict(eye_px=32, map_w=64, map_h=32)
    braking = render_frame(frame, data["arena"], data["telemetry"], 0, data["meta"], **kw)
    cruising = render_frame(frame, data["arena"], data["telemetry"], 1, data["meta"], **kw)
    assert list(braking.getdata()) != list(cruising.getdata())


# --- the side-by-side comparison -------------------------------------------


def _write(tmp_path, name, **overrides):
    data = _recording(overrides.pop("n", 6))
    data["meta"].update(overrides.pop("meta", {}))
    for k, v in overrides.items():
        data[k] = v
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_comparison_stacks_runs_of_the_same_arena(tmp_path):
    from flyguard.make_demo_gif import build_comparison

    a = _write(tmp_path, "a.json", meta={"controller": "flow"})
    b = _write(tmp_path, "b.json", meta={"controller": "connectome"})
    out = build_comparison([a, b], tmp_path / "cmp.gif", step=1, eye_px=32,
                           map_w=64, map_h=32, fps=10, loop_pause_ms=500)
    with Image.open(out) as im:
        stacked_height = im.size[1]
    with Image.open(build_comparison([a], tmp_path / "one.gif", step=1, eye_px=32,
                                     map_w=64, map_h=32, fps=10,
                                     loop_pause_ms=500)) as im:
        single_height = im.size[1]
    assert stacked_height > single_height * 1.9


def test_comparison_refuses_different_arenas(tmp_path):
    """Stacking two different arenas invites a comparison that is not being
    made -- the whole point is that both controllers saw identical obstacles."""
    from flyguard.make_demo_gif import build_comparison

    a = _write(tmp_path, "a.json")
    other = _recording()
    other["arena"]["obstacles"] = [[3.0, 0.0]]
    b = tmp_path / "b.json"
    b.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="different arenas"):
        build_comparison([a, b], tmp_path / "x.gif", step=1, eye_px=32,
                         map_w=64, map_h=32, fps=10, loop_pause_ms=500)


def test_comparison_runs_to_the_longest_recording(tmp_path):
    """Runs end at different times. The shorter must freeze on its outcome,
    not vanish, or the animation reads as a recording glitch."""
    from flyguard.make_demo_gif import build_comparison

    long_run = _write(tmp_path, "long.json", n=8)
    short = _recording(n=3)
    short_path = tmp_path / "short.json"
    short_path.write_text(json.dumps(short))
    out = build_comparison([long_run, short_path], tmp_path / "cmp.gif", step=1,
                           eye_px=32, map_w=64, map_h=32, fps=10, loop_pause_ms=500)
    with Image.open(out) as im:
        assert im.n_frames == 8
