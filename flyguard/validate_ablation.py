"""Ablation study on the real extracted connectome (CLAUDE.md task 5):
delete individual cell types from the wiring matrix and measure what
breaks in looming-vs-translation discrimination. Extends the abstract
ring-circuit LPi ablation already proven in `flyguard.stimuli`
(`ablate_lpi=True`) to the real anatomy, and adds individual T4/T5 subtype
deletions the ring model can't represent (it only has 4 abstract "arm"
channels, not real per-subtype populations).

"Deletion" means zeroing both the row and column of the real weight matrix
`W` for every neuron of the target cell type(s): no outgoing synapses (the
neuron can't influence anything downstream any more) and no incoming
synapses (it can't be driven by anything else in the network). Ablated
T4/T5 neurons still receive their normal external Poisson drive from
`encoder.poisson_stim_fn` (that's injected as `i_ext`, not through `W`) --
this specifically tests "what if this population's synaptic OUTPUT is
gone," which is the meaningful ablation question here, not "what if this
population never got visual input."

Usage:
    python -m flyguard.validate_ablation --npz data/looming.npz \
        --columns ~/flywire/column_assignment.csv --side right
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from flyguard.extract import load_subnetwork
from flyguard.validate_encoder import sweep_side

ABLATION_CONDITIONS = {
    "intact": [],
    "LPi ablated": ["LPi"],
    "T4a ablated": ["T4a"],
    "T4b ablated": ["T4b"],
    "T4c ablated": ["T4c"],
    "T4d ablated": ["T4d"],
}


def ablate_cell_types(W, meta, prefixes: list[str]):
    """Zero the row and column of every neuron whose `cell_type` starts with
    one of `prefixes` (e.g. "LPi" matches all 11 real LPi subtypes;
    "T4a" matches only T4a, since no other real type shares that prefix).
    Returns (ablated W, number of neurons ablated).
    """
    mask = np.zeros(len(meta), dtype=bool)
    for p in prefixes:
        mask |= meta.cell_type.str.startswith(p).to_numpy()
    idx = np.flatnonzero(mask)
    Wc = sp.lil_matrix(W)
    Wc[idx, :] = 0
    Wc[:, idx] = 0
    return sp.csr_matrix(Wc), int(mask.sum())


def main(npz_path: Path, columns_path: Path, side: str, heading_step: int,
         base_rate_hz: float, peak_weight: float,
         burn_steps: int, measure_steps: int, seed: int,
         json_out: Path | None = None):
    W, meta, groups = load_subnetwork(npz_path)
    headings = list(range(0, 360, heading_step))

    results = {}
    print(f"{'condition':14s} {'n_ablated':>10s}  {'looming':>8s}  {'mean_trans':>11s}  {'exceeding':>10s}")
    for label, prefixes in ABLATION_CONDITIONS.items():
        if prefixes:
            W_use, n_ablated = ablate_cell_types(W, meta, prefixes)
        else:
            W_use, n_ablated = W, 0
        r = sweep_side(W_use, meta, groups, columns_path, side, headings,
                        base_rate_hz, peak_weight, burn_steps, measure_steps, seed)
        results[label] = r
        vals = list(r["translation"].values())
        n_exceed = sum(v > r["looming"] for v in vals)
        print(f"{label:14s} {n_ablated:10d}  {r['looming']:8.1f}  {np.mean(vals):11.1f}  "
              f"{n_exceed:3d} / {len(vals)}")

    intact = results["intact"]
    print(f"\nintact discrimination margin (looming - mean translation): "
          f"{intact['looming'] - np.mean(list(intact['translation'].values())):.1f} Hz")
    lpi = results["LPi ablated"]
    print(f"LPi-ablated margin: "
          f"{lpi['looming'] - np.mean(list(lpi['translation'].values())):.1f} Hz")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {json_out}")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("data/looming.npz"))
    ap.add_argument("--columns", type=Path, required=True,
                     help="path to the raw column_assignment.csv (not committed)")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--heading-step", type=int, default=45)
    ap.add_argument("--base-rate-hz", type=float, default=250.0)
    ap.add_argument("--peak-weight", type=float, default=30.0)
    ap.add_argument("--burn-steps", type=int, default=2000)
    ap.add_argument("--measure-steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json-out", type=Path, default=None,
                     help="optionally dump per-condition results as JSON (used for the README figures)")
    a = ap.parse_args()
    main(a.npz, a.columns, a.side, a.heading_step, a.base_rate_hz, a.peak_weight,
         a.burn_steps, a.measure_steps, a.seed, a.json_out)
