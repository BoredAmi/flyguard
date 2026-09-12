"""Builds the closed-loop playback page from a recording.

Keeps the project rule that nothing shown is hand-made: the page is a
template plus a JSON bundle written by `record_avoid_demo.py`, which in turn
comes from `avoid.run_trial` -- the same code path the benchmark measures.

Usage:
    MUJOCO_GL=egl python -m flyguard.record_avoid_demo --out avoid_recording.json
    python -m flyguard.make_avoid_artifact --recording avoid_recording.json \\
        --out docs/corridor_run.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("avoid_artifact_template.html")
PLACEHOLDER = "/*__DATA__*/"


def build(recording: Path, template: Path, out: Path) -> Path:
    html = template.read_text()
    if PLACEHOLDER not in html:
        raise ValueError(f"{template} has no {PLACEHOLDER} placeholder")
    data = json.loads(recording.read_text())

    # The bundle is injected into a <script type="application/json"> block, so
    # the only sequence that can break out of it is a literal "</script>".
    # json.dumps escapes nothing HTML-ish by default; escape the one that matters.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(PLACEHOLDER, payload))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) "
          f"from {recording} ({len(data['frames_jpeg_b64'])} frames)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", type=Path, default=Path("avoid_recording.json"))
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--out", type=Path, default=Path("docs/corridor_run.html"))
    a = ap.parse_args()
    build(a.recording, a.template, a.out)
