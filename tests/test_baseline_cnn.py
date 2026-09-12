import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is an optional extra (pip install flyguard[cnn])")

from flyguard.baseline_cnn import (  # noqa: E402
    TinyLoomNet,
    evaluate,
    inference_latency_ms,
    static_shortcut_accuracy,
    train_model,
)


def _synthetic_dataset(n=40, h=16, w=16, seed=0):
    """Frame pairs where channel 1 - channel 0 encodes the label directly
    (label 1 = grows brighter, label 0 = stays the same) -- enough signal
    for a tiny CNN to learn in a handful of epochs without needing MuJoCo."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    base = rng.uniform(0.2, 0.5, size=(n, h, w)).astype(np.float32)
    second = base.copy()
    for i in range(n):
        if y[i] == 1:
            second[i] += 0.3
    X = np.stack([base, second], axis=1).astype(np.float32)
    return X, y


def test_tinyloomnet_forward_shape():
    model = TinyLoomNet()
    x = torch.zeros(5, 2, 32, 32)
    out = model(x)
    assert out.shape == (5, 2)


def test_tinyloomnet_is_small():
    model = TinyLoomNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 50_000, f"baseline should stay tiny, got {n_params} parameters"


def test_train_model_reduces_loss():
    X, y = _synthetic_dataset()
    losses = [loss for _, loss, _ in train_model(X, y, epochs=5, device="cpu", seed=0)]
    assert losses[-1] < losses[0]


def test_train_model_learns_separable_task():
    X, y = _synthetic_dataset(n=200, seed=1)
    model = None
    for _, _, model in train_model(X, y, epochs=15, device="cpu", seed=1):
        pass
    acc = evaluate(model, X, y, device="cpu")
    assert acc > 0.9, f"should easily learn a linearly-separable brightness cue, got {acc:.2f}"


def test_static_shortcut_accuracy_near_chance_when_cue_is_real_motion():
    """If the label is genuinely carried by the frame-to-frame change (as in
    the synthetic dataset here), duplicating frame 0 into frame 1 removes
    all signal, so accuracy on the resulting static pairs should collapse
    toward chance."""
    X, y = _synthetic_dataset(n=200, seed=2)
    model = None
    for _, _, model in train_model(X, y, epochs=15, device="cpu", seed=2):
        pass
    normal_acc = evaluate(model, X, y, device="cpu")
    shortcut_acc = static_shortcut_accuracy(model, X, y, device="cpu")
    assert normal_acc > 0.9
    assert shortcut_acc < 0.65, f"shortcut accuracy {shortcut_acc:.2f} should be near chance"


def test_static_shortcut_accuracy_flags_a_real_shortcut():
    """Sanity check on the check itself: if the label instead leaks into
    frame 0's absolute brightness (a deliberate shortcut, unrelated to
    motion), the static-duplicated test should stay high, not collapse --
    proving the metric actually detects a shortcut when one exists."""
    rng = np.random.default_rng(3)
    n, h, w = 200, 16, 16
    y = rng.integers(0, 2, size=n).astype(np.int64)
    base = rng.uniform(0.2, 0.3, size=(n, h, w)).astype(np.float32)
    for i in range(n):
        if y[i] == 1:
            base[i] += 0.4  # absolute brightness alone carries the label
    X = np.stack([base, base.copy()], axis=1).astype(np.float32)  # frame1 == frame0 already

    model = None
    for _, _, model in train_model(X, y, epochs=15, device="cpu", seed=3):
        pass
    shortcut_acc = static_shortcut_accuracy(model, X, y, device="cpu")
    assert shortcut_acc > 0.9, "the check should reveal a real shortcut, not just always pass"


def test_inference_latency_is_positive_and_finite():
    model = TinyLoomNet()
    X = np.zeros((4, 2, 32, 32), dtype=np.float32)
    lat = inference_latency_ms(model, X, device="cpu", n_reps=5)
    assert 0 < lat < 1000
