import numpy as np
import pandas as pd
import scipy.sparse as sp

from flyguard.validate_ablation import ablate_cell_types


def _tiny_network():
    """5 neurons: A, B (type X), C (type Y), D, E (type X), fully connected
    with distinct weights so zeroed entries are easy to spot."""
    meta = pd.DataFrame({
        "root_id": [1, 2, 3, 4, 5],
        "cell_type": ["X", "X", "Y", "Z", "X"],
        "side": ["right"] * 5,
        "is_seed": [False] * 5,
    })
    n = 5
    W = sp.csr_matrix(np.arange(1, n * n + 1).reshape(n, n).astype(np.float32))
    return W, meta


def test_ablate_zeroes_rows_and_columns_of_matching_neurons():
    W, meta = _tiny_network()
    Wc, n_ablated = ablate_cell_types(W, meta, ["X"])
    Wc = Wc.toarray()
    x_idx = [0, 1, 4]  # neurons 1, 2, 5 are type X
    assert n_ablated == 3
    for i in x_idx:
        assert np.all(Wc[i, :] == 0), f"row {i} should be fully zeroed"
        assert np.all(Wc[:, i] == 0), f"column {i} should be fully zeroed"


def test_ablate_leaves_non_matching_connections_untouched():
    W, meta = _tiny_network()
    Wc, _ = ablate_cell_types(W, meta, ["X"])
    Wc = Wc.toarray()
    Worig = W.toarray()
    # neuron 3 (type Y, idx 2) <-> neuron 4 (type Z, idx 3): neither ablated
    assert Wc[2, 3] == Worig[2, 3]
    assert Wc[3, 2] == Worig[3, 2]
    assert Wc[2, 3] != 0  # sanity: original weight was actually nonzero


def test_ablate_prefix_matches_multiple_subtypes():
    """Mirrors the real LPi case: many numbered subtypes share a prefix."""
    meta = pd.DataFrame({
        "root_id": [1, 2, 3],
        "cell_type": ["LPi02", "LPi09", "T4a"],
        "side": ["right"] * 3,
        "is_seed": [False] * 3,
    })
    W = sp.csr_matrix(np.ones((3, 3), dtype=np.float32))
    Wc, n_ablated = ablate_cell_types(W, meta, ["LPi"])
    assert n_ablated == 2
    Wc = Wc.toarray()
    assert np.all(Wc[0, :] == 0) and np.all(Wc[:, 0] == 0)
    assert np.all(Wc[1, :] == 0) and np.all(Wc[:, 1] == 0)
    assert Wc[2, 2] == 1.0  # T4a untouched


def test_ablate_exact_subtype_does_not_match_sibling_subtypes():
    """"T4a" should not accidentally ablate "T4b" or vice versa."""
    meta = pd.DataFrame({
        "root_id": [1, 2],
        "cell_type": ["T4a", "T4b"],
        "side": ["right"] * 2,
        "is_seed": [False] * 2,
    })
    W = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    Wc, n_ablated = ablate_cell_types(W, meta, ["T4a"])
    assert n_ablated == 1
    Wc = Wc.toarray()
    assert Wc[1, 1] == 4.0  # T4b's self-weight untouched


def test_ablate_empty_prefix_list_is_a_noop():
    W, meta = _tiny_network()
    Wc, n_ablated = ablate_cell_types(W, meta, [])
    assert n_ablated == 0
    np.testing.assert_array_equal(Wc.toarray(), W.toarray())


def test_ablate_nonexistent_type_is_a_noop():
    W, meta = _tiny_network()
    Wc, n_ablated = ablate_cell_types(W, meta, ["DoesNotExist"])
    assert n_ablated == 0
    np.testing.assert_array_equal(Wc.toarray(), W.toarray())
