"""Tests for the pure logic in query_male_cns.py -- no network, no neuprint
token, no neuprint-python required. The functions that actually call
NeuPrint (require_client, fetch_type_to_type_edges, fetch_downstream_types)
are thin wrappers exercised only by hand, against the live server, since
there is no way to fake a connectome query service meaningfully here;
everything decision-relevant (how an edge list becomes a verdict) is pure
and covered below.
"""

import pandas as pd
import pytest

from flyguard.query_male_cns import (
    EdgeSummary,
    bfs_reaches,
    require_client,
    summarize_edges,
    types_reached,
)


def _edges(rows):
    return pd.DataFrame(rows, columns=["bodyId_pre", "bodyId_post", "type_post", "weight"])


class TestSummarizeEdges:
    def test_empty_reports_not_found(self):
        s = summarize_edges(pd.DataFrame(), "X -> Y")
        assert s.found is False
        assert "not found" in str(s)

    def test_none_reports_not_found(self):
        s = summarize_edges(None, "X -> Y")
        assert s.found is False

    def test_sums_weight_and_counts_edges(self):
        df = _edges([
            (1, 10, "DNa01", 5),
            (1, 11, "DNa01", 3),
            (2, 10, "DNa01", 7),
        ])
        s = summarize_edges(df, "LPLC2 -> DNa01")
        assert s.found is True
        assert s.n_edges == 3
        assert s.total_weight == 15
        assert "3 edges" in str(s)
        assert "weight 15" in str(s)

    def test_str_of_edge_summary_is_stable_shape(self):
        s = EdgeSummary("A -> B", 2, 9)
        assert str(s) == "  A -> B: 2 edges, weight 9"


class TestTypesReached:
    def test_empty_gives_zero_for_every_target(self):
        result = types_reached(pd.DataFrame(), ["DNa01", "DNa02", "MDN"])
        assert result == {"DNa01": 0, "DNa02": 0, "MDN": 0}

    def test_missing_type_post_column_gives_zero(self):
        df = pd.DataFrame({"bodyId_post": [1, 2]})
        result = types_reached(df, ["DNa01"])
        assert result == {"DNa01": 0}

    def test_counts_weight_per_target_type_only(self):
        df = _edges([
            (1, 10, "DNa01", 4),
            (1, 11, "DNa01", 6),
            (2, 12, "T4a", 100),  # not a target type -- must not leak in
            (2, 13, "MDN", 2),
        ])
        result = types_reached(df, ["DNa01", "DNa02", "MDN"])
        assert result == {"DNa01": 10, "DNa02": 0, "MDN": 2}


class TestBfsReaches:
    def test_not_reached_within_hops_is_zero(self):
        hop1 = _edges([(1, 10, "T4a", 5)])
        hop2 = _edges([(10, 20, "T4b", 5)])
        result = bfs_reaches([hop1, hop2], ["DNa01", "DNa02", "MDN"])
        assert result == {"DNa01": 0, "DNa02": 0, "MDN": 0}

    def test_records_first_hop_a_target_is_reached(self):
        hop1 = _edges([(1, 10, "T4a", 5)])  # DNa01 not yet reached
        hop2 = _edges([(10, 20, "DNa01", 7)])  # reached at hop 2
        result = bfs_reaches([hop1, hop2], ["DNa01"])
        assert result == {"DNa01": 2}

    def test_does_not_overwrite_first_hop_with_a_later_one(self):
        hop1 = _edges([(1, 10, "DNa01", 3)])  # reached at hop 1
        hop2 = _edges([(10, 20, "DNa01", 99)])  # must not overwrite to 2
        result = bfs_reaches([hop1, hop2], ["DNa01"])
        assert result == {"DNa01": 1}

    def test_empty_hop_list_reaches_nothing(self):
        result = bfs_reaches([], ["DNa01", "MDN"])
        assert result == {"DNa01": 0, "MDN": 0}


class TestRequireClient:
    def test_missing_token_raises_actionable_error(self, monkeypatch):
        monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
        pytest.importorskip("neuprint")
        with pytest.raises(RuntimeError, match="neuprint.janelia.org"):
            require_client(token=None)

    def test_missing_package_raises_actionable_error(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "neuprint":
                raise ImportError("no module named neuprint")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="pip install flyguard\\[neuprint\\]"):
            require_client(token="fake-token")
