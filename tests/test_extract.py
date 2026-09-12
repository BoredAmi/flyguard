import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from flyguard.extract import extract, load_subnetwork, subset_subnetwork


def _write_tiny_connectome(data_dir):
    """A tiny synthetic connectome with two independent (mostly) hemispheres
    plus one deliberate cross-hemisphere edge, so the --side filter has
    something real to drop.

        1 (A, right) -> 2 (B, right)     same-hemisphere, kept for side=right
        3 (A, left)  -> 4 (B, left)      same-hemisphere, kept for side=left
        3 (A, left)  -> 5 (X, right)     cross-hemisphere, dropped by side=left
    """
    connections = pd.DataFrame({
        "pre_root_id": [1, 3, 3],
        "post_root_id": [2, 4, 5],
        "syn_count": [10, 10, 10],
        "nt_type": ["ACH", "ACH", "ACH"],
    })
    classification = pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5],
        "side": ["right", "right", "left", "left", "right"],
    })
    types = pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5],
        "primary_type": ["A", "B", "A", "B", "X"],
    })
    connections.to_csv(data_dir / "connections.csv", index=False)
    classification.to_csv(data_dir / "classification.csv", index=False)
    types.to_csv(data_dir / "consolidated_cell_types.csv", index=False)


def test_extract_without_side_keeps_cross_hemisphere_edge(tmp_path):
    _write_tiny_connectome(tmp_path)
    out = tmp_path / "out.npz"
    extract(tmp_path, out, types=["A", "B", "X"], hops=1, min_syn=5)
    W, meta, groups = load_subnetwork(out)
    assert set(meta.root_id) == {1, 2, 3, 4, 5}


def test_extract_with_side_drops_cross_hemisphere_partner(tmp_path):
    _write_tiny_connectome(tmp_path)
    out = tmp_path / "out_left.npz"
    extract(tmp_path, out, types=["A", "B", "X"], hops=1, min_syn=5, side="left")
    W, meta, groups = load_subnetwork(out)
    # seed is neuron 3 (A, left); its same-hemisphere partner 4 (B, left) is
    # kept, but 5 (X, right), pulled in by growth, must be dropped.
    assert set(meta.root_id) == {3, 4}
    assert (meta.side == "left").all()


def test_extract_with_side_raises_if_no_seed_on_that_side(tmp_path):
    _write_tiny_connectome(tmp_path)
    out = tmp_path / "out.npz"
    with pytest.raises(ValueError):
        extract(tmp_path, out, types=["X"], hops=1, min_syn=5, side="left")


def _tiny_subnetwork():
    meta = pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5],
        "cell_type": ["LPLC2", "LC4", "DNp01", "T4a", "LPLC2"],
        "side": ["right"] * 5,
        "is_seed": [True] * 5,
    })
    n = 5
    W = sp.csr_matrix(np.arange(1, n * n + 1).reshape(n, n).astype(np.float32))
    return W, meta


def test_subset_subnetwork_keeps_only_requested_types():
    W, meta = _tiny_subnetwork()
    Wc, meta_c, groups_c = subset_subnetwork(W, meta, ["LPLC2", "LC4", "DNp01"])
    assert set(meta_c.cell_type) == {"LPLC2", "LC4", "DNp01"}
    assert len(meta_c) == 4  # everything except the T4a neuron
    assert Wc.shape == (4, 4)


def test_subset_subnetwork_preserves_real_weights_among_kept_neurons():
    W, meta = _tiny_subnetwork()
    Wfull = W.toarray()
    Wc, meta_c, groups_c = subset_subnetwork(W, meta, ["LPLC2", "LC4", "DNp01"])
    Wc = Wc.toarray()
    # original indices 0,1,2,4 are kept (root_id 1,2,3,5); check one edge survives unchanged
    kept_orig_idx = np.flatnonzero(meta.cell_type.isin(["LPLC2", "LC4", "DNp01"]).to_numpy())
    i, j = 0, 1  # positions within kept_orig_idx
    assert Wc[i, j] == Wfull[kept_orig_idx[i], kept_orig_idx[j]]


def test_subset_subnetwork_groups_index_into_the_new_smaller_matrix():
    W, meta = _tiny_subnetwork()
    Wc, meta_c, groups_c = subset_subnetwork(W, meta, ["LPLC2", "LC4", "DNp01"])
    for t, idx in groups_c.items():
        assert (meta_c.iloc[idx].cell_type == t).all()
    assert len(groups_c["LPLC2"]) == 2  # root_id 1 and 5
