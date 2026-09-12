"""Sanity check on an extracted subnetwork.

Prints composition, checks whether LC4<->LPLC2 inhibitory connections
actually exist in the data, and benchmarks LIF speed on the real matrix
(not a synthetic one). Run after `python -m flyguard.extract`.

Usage:
    python -m flyguard.inspect_subnetwork --npz data/looming.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from flyguard.extract import load_subnetwork
from flyguard.lif import LIFNetwork, LIFParams, rates


def main(npz_path: Path):
    W, meta, groups = load_subnetwork(npz_path)
    N, nnz = W.shape[0], W.nnz
    print(f"subnetwork: {N} neurons, {nnz} connections, "
          f"{100 * nnz / N**2:.2f}% density\n")

    print("cell type composition (top 15, excluding 'unknown'):")
    counts = meta.loc[meta.cell_type != "unknown", "cell_type"].value_counts()
    for t, c in counts.head(15).items():
        print(f"  {t:12s} {c:5d}")
    n_unknown = (meta.cell_type == "unknown").sum()
    print(f"  {'unknown':12s} {n_unknown:5d}  ({100*n_unknown/N:.0f}% of subnetwork)\n")

    # --- signed connectivity between LC4 and LPLC2 ---
    lc4 = groups.get("LC4", np.array([], dtype=int))
    lplc2 = groups.get("LPLC2", np.array([], dtype=int))
    # --- signed connectivity: the ACTUAL radial-opponency circuit (Klapoetke et al. 2017) ---
    # LC4<->LPLC2 turned out to be a red herring (see below); the real mechanism is
    # T4/T5 (outward-tuned, excitatory) vs LPi (inward-tuned, inhibitory) both
    # converging onto LPLC2's dendrites within the same lobula-plate layer.
    t4t5 = np.concatenate([groups.get(t, np.array([], dtype=int))
                            for t in meta.cell_type.unique() if str(t).startswith(("T4", "T5"))])
    lpi = np.concatenate([groups.get(t, np.array([], dtype=int))
                           for t in meta.cell_type.unique() if str(t).upper().startswith("LPI")])

    def summarize(src, dst, label):
        if len(src) == 0 or len(dst) == 0:
            print(f"  {label}: not found in this subnetwork")
            return
        block = W[np.ix_(src, dst)]
        pos = block.data[block.data > 0].sum() if block.nnz else 0
        neg = block.data[block.data < 0].sum() if block.nnz else 0
        print(f"  {label}: {block.nnz} edges, "
              f"excitatory weight sum {pos:.0f}, inhibitory weight sum {neg:.0f}")

    print("T4/T5 -> LPLC2 and LPi -> LPLC2 (the actual radial-opponency circuit):")
    summarize(t4t5, lplc2, "T4/T5 -> LPLC2")
    summarize(lpi, lplc2, "LPi   -> LPLC2")
    print()

    if len(lc4) and len(lplc2):
        print("LC4 <-> LPLC2 signed connectivity (expected weak -- parallel pathways):")
        summarize(lc4, lplc2, "LC4 -> LPLC2")
        summarize(lplc2, lc4, "LPLC2 -> LC4")
    else:
        print("WARNING: LC4 or LPLC2 not found in this subnetwork "
              "(check --types / --hops when extracting).")
    print()

    # --- real-data LIF speed ---
    print("LIF benchmark on the actual extracted matrix:")
    net = LIFNetwork(W, LIFParams(dt=1e-4), seed=0)
    stim_targets = groups.get("LPLC2", np.arange(min(50, N)))
    net.run(200, stim_fn=lambda k: net.poisson(stim_targets, 100.0))  # warm-up
    steps = 5000  # 0.5 s simulated
    t0 = time.perf_counter()
    spikes = net.run(steps, stim_fn=lambda k: net.poisson(stim_targets, 100.0))
    elapsed = time.perf_counter() - t0
    print(f"  {N} neurons, {nnz} synapses, dt=0.1ms")
    print(f"  {elapsed:.2f} s wall-clock for {steps*1e-4:.2f} s simulated "
          f"-> {steps*1e-4/elapsed:.1f}x realtime")
    print(f"  {elapsed/steps*1e6:.1f} us/step")
    print(f"  mean firing rate: {rates(spikes, 1e-4).mean():.1f} Hz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("data/looming.npz"))
    main(ap.parse_args().npz)