import json
from pathlib import Path


import pytest

# Skip the whole module rather than failing collection: an optional
# dependency missing must not abort the entire test run.
pytest.importorskip("torch", reason="the offline benchmark imports the CNN baseline")

from flyguard.offline_benchmark import connectome_accuracy

REAL_SWEEP = Path(__file__).resolve().parents[1] / "data" / "hemisphere_sweep.json"


def test_connectome_accuracy_on_synthetic_sweep():
    """Isolates the threshold/accuracy formula from any specific real
    numbers: 1 looming trial correctly above threshold, 3 translation
    headings below (correct) and 1 above (misclassified) -> 4/5 accuracy."""
    data = {
        "right": {
            "looming": 100.0,
            "translation": {"0": 10.0, "90": 10.0, "180": 10.0, "270": 200.0},
        }
    }
    path = Path("/tmp") / "synthetic_sweep_test.json"
    path.write_text(json.dumps(data))
    try:
        result = connectome_accuracy(path)
    finally:
        path.unlink()
    r = result["right"]
    # threshold = (100 + mean(10,10,10,200)) / 2 = (100 + 57.5) / 2 = 78.75
    assert r["threshold_hz"] == pytest.approx(78.75)
    # looming (100) > threshold: correct. translations 10,10,10 <= threshold: correct.
    # translation 200 > threshold: misclassified as looming.
    assert r["accuracy"] == pytest.approx(4 / 5)
    assert r["n_conditions"] == 5


def test_connectome_accuracy_perfect_when_well_separated():
    data = {"left": {"looming": 100.0, "translation": {str(i): 5.0 for i in range(10)}}}
    path = Path("/tmp") / "synthetic_sweep_perfect.json"
    path.write_text(json.dumps(data))
    try:
        result = connectome_accuracy(path)
    finally:
        path.unlink()
    assert result["left"]["accuracy"] == pytest.approx(1.0)


@pytest.mark.skipif(not REAL_SWEEP.exists(), reason="data/hemisphere_sweep.json not present")
def test_connectome_accuracy_matches_known_reference_numbers():
    """Regression check against the numbers reported in CLAUDE.md 'Pilot
    findings' for the offline benchmark -- right hemisphere's exceeding
    band (5/24 headings) costs real accuracy relative to the left."""
    result = connectome_accuracy(REAL_SWEEP)
    assert result["right"]["accuracy"] == pytest.approx(0.76, abs=0.01)
    assert result["left"]["accuracy"] == pytest.approx(0.92, abs=0.01)
    assert result["right"]["accuracy"] < result["left"]["accuracy"]
