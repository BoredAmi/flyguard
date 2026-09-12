"""Real-data validation of the retinotopic encoder (CLAUDE.md task 1).

Drives the REAL T4/T5 neurons in the extracted subnetwork (not the abstract
9-neuron ring circuit in `flyguard.stimuli`) with synthetic looming and
translational flow fields via `flyguard.encoder.poisson_stim_fn`, and
measures the real LPLC2 population's firing rate. This is the step that
checks whether radial motion opponency, proven exactly on the idealized
ring model, survives contact with the real, unevenly-weighted, recurrent
connectome.

Requires the raw `column_assignment.csv` (not committed -- see CLAUDE.md
"Actual Codex file schema"), pointed to via `--columns`.

Usage:
    python -m flyguard.validate_encoder --npz data/looming.npz \
        --columns ~/flywire/column_assignment.csv --side right

    # per-hemisphere replication control, fine heading resolution (CLAUDE.md
    # "Pilot findings" FOLLOW-UP -- regenerates the two-hemisphere chart data):
    python -m flyguard.validate_encoder --npz data/looming.npz \
        --columns ~/flywire/column_assignment.csv --both-sides --heading-step 15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flyguard.encoder import (
    expanding_flow_2d,
    load_column_assignment,
    poisson_stim_fn,
    translational_flow_2d,
)
from flyguard.extract import load_subnetwork
from flyguard.lif import LIFNetwork, LIFParams, rates


def run_condition(W, lplc2_idx, stim_fn, params, burn_steps, measure_steps, seed):
    net = LIFNetwork(W, params, seed=seed)
    net.run(burn_steps, stim_fn=stim_fn)  # discard transient before opponency settles
    spikes = net.run(measure_steps, stim_fn=stim_fn, record=lplc2_idx)
    return float(rates(spikes, params.dt).mean())


def sweep_side(W, meta, groups, columns_path: Path, side: str, headings_deg,
                base_rate_hz: float, peak_weight: float,
                burn_steps: int, measure_steps: int, seed: int) -> dict:
    """Run the looming + translation-heading sweep for one hemisphere.
    Returns {"looming": Hz, "translation": {heading: Hz, ...}, "n_lplc2": int}."""
    lplc2_idx = np.flatnonzero(((meta.cell_type == "LPLC2") & (meta.side == side)).to_numpy())
    if len(lplc2_idx) == 0:
        raise ValueError(f"no LPLC2 neurons on side={side!r}")

    columns = load_column_assignment(columns_path, side=side)
    sub_ids = meta.root_id.to_numpy()
    x, y = columns["x"].to_numpy(), columns["y"].to_numpy()
    p = LIFParams(dt=1e-4)

    def measure(vx, vy):
        stim_fn = poisson_stim_fn(columns, vx, vy, sub_ids, dt=p.dt,
                                   base_rate_hz=base_rate_hz, peak_weight=peak_weight, seed=seed)
        return run_condition(W, lplc2_idx, stim_fn, p, burn_steps, measure_steps, seed)

    r_loom = measure(*expanding_flow_2d(x, y))
    trans_rates = {}
    for d in headings_deg:
        vx, vy = translational_flow_2d(x, float(d))
        trans_rates[d] = measure(vx, vy)

    return {
        "looming": r_loom,
        "translation": trans_rates,
        "n_lplc2": int(len(lplc2_idx)),
        "n_columns_present": int(columns["root_id"].isin(sub_ids).sum()),
        "n_columns_total": int(len(columns)),
    }


def print_side_report(side: str, result: dict, exc_inh_ratio: float | None):
    r_loom = result["looming"]
    trans = result["translation"]
    print(f"\n=== {side} hemisphere ===")
    print(f"LPLC2: {result['n_lplc2']} neurons | "
          f"T4/T5 present: {result['n_columns_present']} / {result['n_columns_total']}")
    print(f"looming: LPLC2 mean rate = {r_loom:.1f} Hz")
    for d, r in trans.items():
        flag = " <-- exceeds looming" if r > r_loom else ""
        print(f"translation({int(d):3d} deg): LPLC2 mean rate = {r:.1f} Hz{flag}")

    vals = list(trans.values())
    mean_trans = float(np.mean(vals))
    worst_trans = max(vals)
    n_exceeding = sum(r > r_loom for r in vals)
    print(f"\nlooming vs mean(translation): {r_loom:.1f} Hz vs {mean_trans:.1f} Hz "
          f"({r_loom / mean_trans:.2f}x)")
    print(f"looming vs worst-case translation heading: {r_loom:.1f} Hz vs {worst_trans:.1f} Hz")
    print(f"headings where translation exceeds looming: {n_exceeding} / {len(vals)}")
    if exc_inh_ratio is not None:
        print(f"real T4/T5->LPLC2 : LPi->LPLC2 total weight ratio = {exc_inh_ratio:.2f} "
              f"(idealized ring model used exactly 1:1)")


def exc_inh_weight_ratio(W, meta, groups) -> float | None:
    """Real anatomical exc:inh imbalance onto LPLC2 (both hemispheres combined,
    since the connectome doesn't split by side for this) -- context for why
    perfect cancellation, exact in the idealized ring model, doesn't hold
    exactly on the real wiring."""
    t4t5 = np.concatenate([groups.get(t, np.array([], dtype=int))
                            for t in meta.cell_type.unique() if str(t).startswith(("T4", "T5"))])
    lpi = np.concatenate([groups.get(t, np.array([], dtype=int))
                           for t in meta.cell_type.unique() if str(t).upper().startswith("LPI")])
    lplc2_all = groups.get("LPLC2", np.array([], dtype=int))
    if not (len(t4t5) and len(lpi) and len(lplc2_all)):
        return None
    block_e = W[np.ix_(t4t5, lplc2_all)]
    block_i = W[np.ix_(lpi, lplc2_all)]
    exc_sum = block_e.data[block_e.data > 0].sum() if block_e.nnz else 0.0
    inh_sum = -block_i.data[block_i.data < 0].sum() if block_i.nnz else 0.0
    return exc_sum / inh_sum if inh_sum > 0 else None


def main(npz_path: Path, columns_path: Path, side: str, both_sides: bool, heading_step: int,
         base_rate_hz: float, peak_weight: float,
         burn_steps: int, measure_steps: int, seed: int, json_out: Path | None):
    W, meta, groups = load_subnetwork(npz_path)
    headings = list(range(0, 360, heading_step))
    ratio = exc_inh_weight_ratio(W, meta, groups)

    sides = ["right", "left"] if both_sides else [side]
    results = {}
    for s in sides:
        results[s] = sweep_side(W, meta, groups, columns_path, s, headings,
                                 base_rate_hz, peak_weight, burn_steps, measure_steps, seed)
        print_side_report(s, results[s], ratio)

    if both_sides:
        print("\n=== summary ===")
        for s in sides:
            vals = list(results[s]["translation"].values())
            n_exc = sum(v > results[s]["looming"] for v in vals)
            print(f"{s:5s}: {n_exc}/{len(vals)} headings exceed looming "
                  f"(looming {results[s]['looming']:.1f} Hz)")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {json_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("data/looming.npz"))
    ap.add_argument("--columns", type=Path, required=True,
                     help="path to the raw column_assignment.csv (not committed)")
    ap.add_argument("--side", default="right", choices=["left", "right"],
                     help="ignored if --both-sides is given")
    ap.add_argument("--both-sides", action="store_true",
                     help="run the sweep on both hemispheres (per-hemisphere replication control)")
    ap.add_argument("--heading-step", type=int, default=45,
                     help="translation heading resolution in degrees, e.g. 15 for a fine sweep")
    ap.add_argument("--base-rate-hz", type=float, default=250.0)
    ap.add_argument("--peak-weight", type=float, default=30.0)
    ap.add_argument("--burn-steps", type=int, default=2000)
    ap.add_argument("--measure-steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json-out", type=Path, default=None,
                     help="optionally dump the raw per-heading results as JSON")
    a = ap.parse_args()
    main(a.npz, a.columns, a.side, a.both_sides, a.heading_step, a.base_rate_hz, a.peak_weight,
         a.burn_steps, a.measure_steps, a.seed, a.json_out)
