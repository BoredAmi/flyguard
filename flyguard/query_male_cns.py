"""Check whether LPLC2/LC4 reach steering descending neurons through the
ventral nerve cord, using the male whole-CNS connectome.

Why this script exists
-----------------------
Every closed-loop result in this project rests on a finding from
`inspect_subnetwork.py`: in the FAFB hops=1 subnetwork, LPLC2/LC4 have
*zero* direct edges to DNa01, DNa02 or MDN (the steering/reverse descending
neurons) and hundreds of edges to DNp01 (the Giant Fiber, escape-only).
`avoid.py` therefore treats steering as "an engineering addition, not
connectome-derived" and says so everywhere.

That finding could never be more than "not found in this crop of the
graph", because FAFB is brain-only -- descending neurons are defined by
where they end up, in the ventral nerve cord (VNC), and FAFB simply does
not contain the VNC. A zero count there is consistent both with "these
neurons don't talk" and "the wiring is there, just outside the crop".

male-cns:v1.0 (Janelia FlyEM / Google Research, released 2026-09-03,
https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
is the first dataset this project has access to that includes the VNC:
166,000 neurons, 125 million synapses, brain and cord together in one
animal. It is therefore the first real test of whether "steering is not
connectome-derived" is a fact about the fly or an artifact of FAFB's crop.

This does not retire the FAFB-based results elsewhere in this project --
male-cns is a different individual and a different sex, so a positive
result here is a hypothesis about anatomy, not a correction to any number
already reported. See CLAUDE.md's "male whole-CNS connectome" entry.

Access
------
Queried live via NeuPrint (https://neuprint.janelia.org, dataset
"male-cns:v1.0"), not downloaded as static CSVs -- male-cns does not ship
a Codex-style bulk CSV export the way FlyWire does. This requires an API
token bound to a Google account, obtained by logging in at
https://neuprint.janelia.org and copying the token from the Account menu.
There is no way to obtain this programmatically; set it as
NEUPRINT_APPLICATION_CREDENTIALS before running.

Usage:
    pip install flyguard[neuprint]
    export NEUPRINT_APPLICATION_CREDENTIALS=<token from neuprint.janelia.org>
    python -m flyguard.query_male_cns
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import pandas as pd

DATASET = "male-cns:v1.0"
SERVER = "neuprint.janelia.org"

LOOMING_SOURCE_TYPES = ["LPLC2", "LC4"]
ESCAPE_TYPES = ["DNp01"]
STEERING_TARGET_TYPES = ["DNa01", "DNa02", "MDN"]


def require_client(token: str | None = None):
    """Build a neuprint Client, failing with an actionable message rather
    than a bare ImportError or a bare 401."""
    try:
        from neuprint import Client
    except ImportError as exc:
        raise RuntimeError(
            "neuprint-python is not installed. Run: pip install flyguard[neuprint]"
        ) from exc

    token = token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError(
            "No NeuPrint API token found. This cannot be generated "
            "programmatically -- log in at https://neuprint.janelia.org with "
            "a Google account, open Account -> copy your auth token, and set "
            "NEUPRINT_APPLICATION_CREDENTIALS to it."
        )
    return Client(SERVER, dataset=DATASET, token=token)


def fetch_type_to_type_edges(client, source_types: list[str], target_types: list[str]) -> pd.DataFrame:
    """All synapses from any neuron of `source_types` to any neuron of
    `target_types`, as a bodyId-level edge list with a `type_pre`/`type_post`
    column each. Thin wrapper around neuprint's adjacency fetch so the rest
    of this module never touches the client directly and can be tested
    against a synthetic DataFrame instead."""
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_adjacencies

    _, conn_df = fetch_adjacencies(
        NC(type=source_types, client=client),
        NC(type=target_types, client=client),
        client=client,
    )
    return conn_df


def fetch_downstream_types(client, source_types: list[str]) -> pd.DataFrame:
    """Every postsynaptic neuron of any neuron in `source_types`, with its
    own type and weight -- used to expand one hop at a time without
    assuming in advance which types are reachable."""
    from neuprint import NeuronCriteria as NC
    from neuprint import fetch_adjacencies

    _, conn_df = fetch_adjacencies(NC(type=source_types, client=client), NC(client=client), client=client)
    return conn_df


@dataclass
class EdgeSummary:
    label: str
    n_edges: int
    total_weight: int
    found: bool = True

    def __str__(self) -> str:
        if not self.found:
            return f"  {self.label}: not found in male-cns"
        return f"  {self.label}: {self.n_edges} edges, weight {self.total_weight}"


def summarize_edges(conn_df: pd.DataFrame, label: str) -> EdgeSummary:
    """Pure summary of an edge-list DataFrame with a `weight` column --
    no neuprint call, so this is what the tests exercise directly."""
    if conn_df is None or len(conn_df) == 0:
        return EdgeSummary(label, 0, 0, found=False)
    return EdgeSummary(label, len(conn_df), int(conn_df["weight"].sum()))


def types_reached(conn_df: pd.DataFrame, target_types: list[str]) -> dict[str, int]:
    """For each target type, how many edges from the expanded source set
    land on it -- used to check a hop's frontier against the steering
    types without assuming which ones will show up."""
    if conn_df is None or len(conn_df) == 0 or "type_post" not in conn_df.columns:
        return {t: 0 for t in target_types}
    counts = conn_df.groupby("type_post")["weight"].sum()
    return {t: int(counts.get(t, 0)) for t in target_types}


def bfs_reaches(conn_df_by_hop: list[pd.DataFrame], target_types: list[str]) -> dict[str, int]:
    """Which hop (1-indexed) first reaches each target type, given a list
    of per-hop adjacency DataFrames already fetched by the caller. 0 means
    not reached within the hops provided. Pure function, independent of
    neuprint, so it is what the multi-hop test checks."""
    first_hop = {t: 0 for t in target_types}
    for hop_idx, conn_df in enumerate(conn_df_by_hop, start=1):
        reached = types_reached(conn_df, target_types)
        for t, weight in reached.items():
            if weight > 0 and first_hop[t] == 0:
                first_hop[t] = hop_idx
    return first_hop


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=None, help="NeuPrint token (default: $NEUPRINT_APPLICATION_CREDENTIALS)")
    parser.add_argument("--max-hops", type=int, default=2, help="how far to expand from LPLC2/LC4 looking for steering types")
    args = parser.parse_args(argv)

    client = require_client(args.token)

    print(f"male-cns:v1.0 via {SERVER}\n")

    print("Direct LPLC2/LC4 -> steering descending neurons (the FAFB question, re-asked with the VNC present):")
    direct = fetch_type_to_type_edges(client, LOOMING_SOURCE_TYPES, STEERING_TARGET_TYPES)
    print(summarize_edges(direct, "LPLC2/LC4 -> DNa01/DNa02/MDN (direct)"))

    print("\nDirect LPLC2/LC4 -> DNp01 (the known FAFB result, as a sanity check that male-cns agrees):")
    gf = fetch_type_to_type_edges(client, LOOMING_SOURCE_TYPES, ESCAPE_TYPES)
    print(summarize_edges(gf, "LPLC2/LC4 -> DNp01 (direct)"))

    print(f"\nExpanding up to {args.max_hops} hops from LPLC2/LC4 looking for DNa01/DNa02/MDN "
          "(now reachable in principle, since the VNC is in this graph):")
    frontier_types = list(LOOMING_SOURCE_TYPES)
    conn_by_hop = []
    for hop in range(1, args.max_hops + 1):
        conn_df = fetch_downstream_types(client, frontier_types)
        conn_by_hop.append(conn_df)
        frontier_types = sorted(set(conn_df.get("type_post", pd.Series(dtype=str)).dropna().unique()))
        print(f"  hop {hop}: {len(frontier_types)} distinct downstream types")

    reached_at = bfs_reaches(conn_by_hop, STEERING_TARGET_TYPES)
    for t, hop in reached_at.items():
        status = f"reached at hop {hop}" if hop else f"not reached within {args.max_hops} hops"
        print(f"  {t}: {status}")

    print(
        "\nIf DNa01/DNa02/MDN are reached here, `avoid.py`'s bilateral turn law "
        "-- currently an engineering addition on top of only-escape wiring -- has a "
        "real anatomical pathway it could instead be built from. If not, the "
        "'DNp01-only' finding is a real property of this circuit's descending "
        "output, not an artifact of FAFB's brain-only crop, and steering should "
        "keep being reported as ours, not the connectome's."
    )


if __name__ == "__main__":
    main()
