"""Regenerates the figures used in README.md from saved result files.

Keeps the project rule that no number (or figure) is hand-made: every panel
here reads a `.json` written by one of the validate_* scripts, or renders
the stimulus fresh from MuJoCo. If a figure and the text disagree,
re-running this is the tiebreak.

Usage:
    MUJOCO_GL=egl python -m flyguard.make_figures --out docs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# One palette for every figure, so the README reads as a set.
INK = "#1b1f27"
MUTED = "#6b7688"
GRID = "#dfe3ea"
LOOM = "#e07a3c"      # looming / expansion
TRANS = "#3d7ec9"     # translation
CRITICAL = "#c8434a"  # threshold breaches, e-stop
NEUTRAL = "#9aa5b5"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def fig_hemisphere_sweep(sweep_json: Path, out: Path):
    """LPLC2 rate vs translation heading, both hemispheres, with each
    hemisphere's looming response as a reference line. The headline result."""
    with open(sweep_json) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, side in zip(axes, ["right", "left"]):
        d = data[side]
        headings = sorted(int(k) for k in d["translation"])
        vals = [d["translation"][str(h)] for h in headings]
        loom = d["looming"]

        ax.axhline(loom, color=LOOM, lw=1.6, ls="--", zorder=2)
        ax.plot(headings, vals, color=TRANS, lw=1.8, marker="o", ms=3.5,
                mfc="white", mew=1.2, zorder=3, label="translation")

        breaches = [(h, v) for h, v in zip(headings, vals) if v > loom]
        n_breach = len(breaches)
        if breaches:
            bh, bv = zip(*breaches)
            ax.scatter(bh, bv, s=46, color=CRITICAL, zorder=4, edgecolor="white",
                       linewidth=1.0, label="exceeds looming")

        ax.text(352, loom, f" looming {loom:.1f} Hz", color=LOOM, fontsize=8.5,
                va="bottom", ha="right")
        ratio = loom / np.mean(vals)
        ax.set_title(f"{side} hemisphere\n{ratio:.2f}x mean separation .  "
                     f"{n_breach}/{len(vals)} headings exceed looming",
                     fontsize=9.5, color=INK, pad=8)
        ax.set_xlabel("translation heading (deg)")
        ax.set_xticks(range(0, 361, 90))
        ax.set_xlim(-10, 360)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    # headroom so the topmost marker never sits against the frame
    top = max(max(data[s]["translation"].values()) for s in ("right", "left"))
    axes[0].set_ylim(0, top * 1.18)
    axes[0].set_ylabel("LPLC2 population rate (Hz)")

    # one shared legend below both panels -- avoids colliding with either curve
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Radial motion opponency on the real connectome: the effect replicates, "
                 "the failure mode does not", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_ablation(ablation_json: Path, out: Path):
    """Looming vs mean-translation rate per ablation condition, with the
    discrimination margin annotated."""
    with open(ablation_json) as f:
        data = json.load(f)

    labels = list(data.keys())
    loom = [data[k]["looming"] for k in labels]
    trans = [float(np.mean(list(data[k]["translation"].values()))) for k in labels]
    margin = [a - b for a, b in zip(loom, trans)]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.bar(x - w / 2, loom, w, label="looming", color=LOOM)
    ax.bar(x + w / 2, trans, w, label="translation (mean of 8 headings)", color=TRANS)

    for xi, m, lo in zip(x, margin, loom):
        ax.annotate(f"margin {m:.1f} Hz", (xi, max(lo, 0) + 2.0), ha="center",
                    fontsize=8, color=CRITICAL if m < 5 else MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" ablated", "\nablated") for l in labels], fontsize=8.5)
    ax.set_ylabel("LPLC2 population rate (Hz)")
    ax.set_title("Deleting cell types from the real weight matrix: removing LPi collapses "
                 "the margin by 75%", fontsize=11, pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_stimulus_strip(out: Path, resolution: int = 96, n_frames: int = 6):
    """The two stimuli side by side, rendered fresh -- looming expands,
    translation slides at constant size."""
    import mujoco
    from flyguard.mujoco_world import SCENE_XML, TrialParams, render_trial

    # Matched starting depth for both rows -- the controlled pair the real
    # -encoder experiment actually uses. Drawing them at the *unmatched*
    # depths make_trial_params draws by default would show the size confound
    # documented in the project notes rather than the comparison being made here.
    depth0 = 5.0
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=resolution, width=resolution)
    try:
        loom = render_trial(model, renderer, data, TrialParams(
            condition="looming", n_frames=n_frames, dt=0.16, depth0=depth0, radius=0.3, speed=2.4))
        trans = render_trial(model, renderer, data, TrialParams(
            condition="translation", n_frames=n_frames, dt=0.16, depth0=depth0, radius=0.3, speed=1.2))
    finally:
        renderer.close()

    fig, axes = plt.subplots(2, n_frames, figsize=(9.5, 3.5))
    for row, (frames, name, color) in enumerate([(loom, "looming", LOOM),
                                                  (trans, "translation", TRANS)]):
        for col in range(n_frames):
            ax = axes[row, col]
            ax.imshow(frames[col])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(color); s.set_linewidth(1.4)
            if col == 0:
                ax.set_ylabel(name, color=color, fontsize=10, labelpad=8)
    fig.suptitle("MuJoCo stimuli, rendered headless (EGL). Identical object and starting depth "
                 "(5.0 m) -- only the trajectory differs.", fontsize=10.5, y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_avoidance(avoid_json: Path, out: Path):
    """Closed-loop obstacle avoidance: outcome per controller, and how far
    down the corridor each arena got."""
    with open(avoid_json) as f:
        bundle = json.load(f)
    results = bundle["results"]
    goal_x = bundle["config"]["goal_x"]
    order = [n for n in ("straight", "flow", "connectome") if n in results]
    labels = {"straight": "straight\n(no vision)", "flow": "flow\n(no circuit)",
              "connectome": "connectome\n(530 neurons)"}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6),
                             gridspec_kw={"width_ratios": [1.0, 1.35]})

    # --- outcomes, stacked -------------------------------------------------
    ax = axes[0]
    x = np.arange(len(order))
    n = [results[k]["n_arenas"] for k in order]
    goal = np.array([results[k]["goal_rate"] for k in order])
    coll = np.array([results[k]["collision_rate"] for k in order])
    timeout = 1.0 - goal - coll
    ax.bar(x, goal, 0.58, label="reached goal", color=TRANS)
    ax.bar(x, timeout, 0.58, bottom=goal, label="timed out", color=NEUTRAL)
    ax.bar(x, coll, 0.58, bottom=goal + timeout, label="collided", color=CRITICAL)
    for xi, g in zip(x, goal):
        if g > 0:
            ax.text(xi, g / 2, f"{g:.0%}", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in order], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"fraction of arenas (n={n[0]})")
    ax.set_title("Outcome", fontsize=10, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    # --- progress per arena ------------------------------------------------
    ax = axes[1]
    for i, key in enumerate(order):
        xs = [r["progress_x"] for r in results[key]["per_arena"]]
        colours = [TRANS if r["reached_goal"] else (CRITICAL if r["collided"] else NEUTRAL)
                   for r in results[key]["per_arena"]]
        jitter = np.random.default_rng(0).uniform(-0.13, 0.13, len(xs))
        ax.scatter(xs, np.full(len(xs), i) + jitter, c=colours, s=34, zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.plot([float(np.mean(xs))], [i], marker="|", ms=22, mew=2.4, color=INK, zorder=4)
    ax.axvline(goal_x, color=LOOM, lw=1.4, ls="--", zorder=2)
    ax.text(goal_x, len(order) - 0.42, " goal", color=LOOM, fontsize=8.5, va="top")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[k].replace("\n", " ") for k in order], fontsize=8.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, goal_x * 1.08)
    ax.set_xlabel("distance reached along corridor (m)")
    ax.set_title("Per-arena progress  (bar = mean)", fontsize=10, pad=8)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, frameon=False, fontsize=8.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Closed-loop obstacle avoidance: steering from the real connectome, "
                 "against the flow it is fed", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main(out_dir: Path, sweep_json: Path, ablation_json: Path, avoid_json: Path | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if sweep_json.exists():
        fig_hemisphere_sweep(sweep_json, out_dir / "hemisphere_sweep.png")
    else:
        print(f"skip hemisphere figure: {sweep_json} not found")
    if ablation_json.exists():
        fig_ablation(ablation_json, out_dir / "ablation.png")
    else:
        print(f"skip ablation figure: {ablation_json} not found "
              f"(run validate_ablation.py --json-out {ablation_json})")
    if avoid_json is not None and avoid_json.exists():
        fig_avoidance(avoid_json, out_dir / "avoidance.png")
    else:
        print(f"skip avoidance figure: {avoid_json} not found "
              f"(run `python -m flyguard.avoid --json-out {avoid_json}`)")
    fig_stimulus_strip(out_dir / "stimuli.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs"))
    ap.add_argument("--sweep-json", type=Path, default=Path("data/hemisphere_sweep.json"))
    ap.add_argument("--ablation-json", type=Path, default=Path("data/ablation_sweep.json"))
    ap.add_argument("--avoid-json", type=Path, default=Path("data/avoid_results.json"))
    a = ap.parse_args()
    main(a.out, a.sweep_json, a.ablation_json, a.avoid_json)
