"""Extract a subnetwork from the full FlyWire connectome into a light .npz.

Input:  connections.csv + classification.csv from snapshot 783 (Codex).
Output: sparse matrix + metadata, ~2 MB, small enough to commit to the repo.

Usage:
    python -m flyguard.extract --data ~/flywire --out data/looming.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# ACh excitatory; GABA and glutamate inhibitory (in insects GluCl is a chloride
# channel); biogenic amines act by modulation, so they get 0 in a current model.
NT_SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "DA": 0, "SER": 0, "OCT": 0, "UNK": 0}

# Core of the escape circuit. LPLC2 and LC4 are looming detectors, DNp01 is the
# Giant Fiber, the rest are descending neurons for locomotion.
LOOMING_SEEDS = ["LPLC2", "LC4", "DNp01", "DNp02", "DNp04", "DNp11",
                 "DNp09", "DNa01", "DNa02", "MDN", "DNg11"]


def load_connectome(data_dir: Path, connections_file: str = "connections.csv"):
    """Load the CSVs and return (signed sparse matrix, metadata table).

    Codex splits cell-type info across two files: classification.csv carries
    side/super_class/hemilineage but NOT the specific cell type; the type
    itself (e.g. "LPLC2", "DNp01") lives in consolidated_cell_types.csv under
    the column `primary_type`. We merge them here so downstream code can just
    use `meta.cell_type`.
    """
    con = pd.read_csv(data_dir / connections_file)
    cls = pd.read_csv(data_dir / "classification.csv")

    required = {"pre_root_id", "post_root_id", "syn_count", "nt_type"}
    missing = required - set(con.columns)
    if missing:
        raise ValueError(
            f"{connections_file} is missing columns {missing}. "
            f"Found: {list(con.columns)}. Check the snapshot version."
        )

    ids = np.union1d(con.pre_root_id.values, con.post_root_id.values)
    pos = pd.Series(np.arange(len(ids)), index=ids)

    sign = con.nt_type.astype(str).str.upper().map(NT_SIGN).fillna(0).to_numpy()
    W = sp.csr_matrix(
        (con.syn_count.to_numpy() * sign,
         (pos[con.pre_root_id].to_numpy(), pos[con.post_root_id].to_numpy())),
        shape=(len(ids), len(ids)), dtype=np.float32,
    )

    meta = pd.DataFrame({"root_id": ids}).merge(cls, on="root_id", how="left")

    types_path = data_dir / "consolidated_cell_types.csv"
    if types_path.exists():
        types = pd.read_csv(types_path)[["root_id", "primary_type"]]
        meta = meta.merge(types, on="root_id", how="left")
        meta["cell_type"] = meta["primary_type"]
        meta = meta.drop(columns=["primary_type"])
    else:
        meta["cell_type"] = np.nan

    meta["cell_type"] = meta["cell_type"].fillna("unknown").astype(str)
    return W, meta


def grow(W, seed_idx, hops: int = 2, min_syn: int = 5) -> np.ndarray:
    """Grow the set by synaptic partners (both directions), `hops` times."""
    strong = abs(W) >= min_syn
    keep = set(int(i) for i in seed_idx)
    for _ in range(hops):
        cur = np.fromiter(keep, dtype=np.int64)
        out = strong[cur].tocoo().col
        inp = strong.T.tocsr()[cur].tocoo().col
        keep |= set(out.tolist()) | set(inp.tolist())
    return np.array(sorted(keep), dtype=np.int64)


def extract(data_dir: Path, out: Path, types=None, hops=1, min_syn=5,
            connections_file: str = "connections.csv", side: str | None = None):
    """`side`, if given ("left"/"right"), restricts extraction to one
    hemisphere: seed neurons are drawn only from that side, and after
    growth the subnetwork is filtered back down to that side plus any
    midline ("center") neurons, dropping the few cross-hemisphere synapses
    `grow()` would otherwise pull in. This is the per-hemisphere
    replication control (CLAUDE.md) -- FAFB is a single fly, so any result
    could be an artefact of one animal's specific reconstruction; running
    the same experiment on each hemisphere's largely-independent circuit
    checks whether an effect is structural or a one-sided fluke.
    """
    types = types or LOOMING_SEEDS
    W, meta = load_connectome(data_dir, connections_file)
    print(f"full connectome: {W.shape[0]} neurons, {W.nnz} connections")

    mask = meta.cell_type.str.fullmatch("|".join(types), case=False, na=False)
    if side is not None:
        mask &= meta.side == side
    seed_idx = np.flatnonzero(mask.to_numpy())
    if len(seed_idx) == 0:
        raise ValueError(f"no neurons found for cell types {types}" +
                          (f" on side={side!r}" if side else ""))
    print(f"seed neurons: {len(seed_idx)}" + (f" (side={side})" if side else ""))
    for t in types:
        tmask = meta.cell_type.str.fullmatch(t, case=False, na=False)
        if side is not None:
            tmask &= meta.side == side
        print(f"  {t:8s} {int(tmask.sum()):4d}")

    nodes = grow(W, seed_idx, hops=hops, min_syn=min_syn)
    if side is not None:
        before = len(nodes)
        keep = meta.iloc[nodes].side.isin([side, "center"]).to_numpy()
        nodes = nodes[keep]
        print(f"side={side} filter: {before} -> {len(nodes)} neurons "
              f"(dropped {before - len(nodes)} cross-hemisphere partners pulled in by growth)")

    Wsub = sp.csr_matrix(W[nodes][:, nodes])
    print(f"subnetwork: {len(nodes)} neurons, {Wsub.nnz} connections "
          f"({100 * Wsub.nnz / len(nodes) ** 2:.1f}% density)")
    frac = len(nodes) / W.shape[0]
    if frac > 0.15:
        print(f"WARNING: subnetwork is {100*frac:.0f}% of the whole brain. "
              f"Connectomes are small-world graphs -- hops={hops} may already "
              f"reach nearly everything. Consider hops={max(0, hops-1)} or a "
              f"higher --min-syn if you want a genuinely local circuit.")

    sub = meta.iloc[nodes]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        data=Wsub.data, indices=Wsub.indices, indptr=Wsub.indptr, shape=Wsub.shape,
        root_id=sub.root_id.to_numpy(),
        cell_type=sub.cell_type.to_numpy().astype("U32"),
        side=sub.get("side", pd.Series(["?"] * len(sub))).to_numpy().astype("U8"),
        is_seed=np.isin(nodes, seed_idx),
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def load_subnetwork(path: Path):
    """Load the .npz. Returns (W, metadata table, dict cell_type -> indices)."""
    z = np.load(path, allow_pickle=False)
    W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    meta = pd.DataFrame({
        "root_id": z["root_id"], "cell_type": z["cell_type"],
        "side": z["side"], "is_seed": z["is_seed"],
    })
    groups = {t: np.flatnonzero((meta.cell_type == t).to_numpy())
              for t in np.unique(meta.cell_type)}
    return W, meta, groups


def subset_subnetwork(W, meta, types):
    """Restrict an already-loaded subnetwork to just the given cell types
    and the real synapses directly among them -- no re-growth, no fresh
    Codex extraction. This is how the ~300-neuron "core control circuit"
    (LPLC2 + LC4 + DNp01, see CLAUDE.md "Real-data performance") is built
    for the ROS2 node: the full periphery subnetwork (`data/looming.npz`,
    18k neurons, 0.8x realtime) is too slow for a live control loop, but
    restricting it down to just the looming-detector and escape-command
    neurons -- and driving LPLC2 externally instead of through T4/T5 --
    gives a circuit small enough to run several times faster than
    real-time on one CPU core.
    """
    mask = meta.cell_type.isin(types).to_numpy()
    idx = np.flatnonzero(mask)
    Wsub = sp.csr_matrix(W[idx][:, idx])
    meta_sub = meta.iloc[idx].reset_index(drop=True)
    groups_sub = {t: np.flatnonzero((meta_sub.cell_type == t).to_numpy())
                  for t in np.unique(meta_sub.cell_type)}
    return Wsub, meta_sub, groups_sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="directory holding the Codex CSVs")
    ap.add_argument("--out", type=Path, default=Path("data/looming.npz"))
    ap.add_argument("--hops", type=int, default=1,
                     help="1 = direct synaptic partners only. 2+ tends to "
                          "reach most of the brain (small-world graph).")
    ap.add_argument("--min-syn", type=int, default=5)
    ap.add_argument("--types", nargs="*", default=None)
    ap.add_argument("--connections-file", default="connections.csv",
                     help="filename inside --data, e.g. connections_unfiltered.csv")
    ap.add_argument("--side", choices=["left", "right"], default=None,
                     help="restrict to one hemisphere's circuit (plus midline "
                          "'center' neurons) for the per-hemisphere replication "
                          "control. Default: both hemispheres, unfiltered.")
    a = ap.parse_args()
    extract(a.data, a.out, a.types, a.hops, a.min_syn, a.connections_file, a.side)