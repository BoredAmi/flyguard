import numpy as np
import scipy.sparse as sp

from flyguard.lif import LIFNetwork, LIFParams, rates


def net(W, **kw):
    return LIFNetwork(sp.csr_matrix(np.asarray(W, dtype=np.float32)),
                      LIFParams(**kw))


def test_silent_without_input():
    """With no stimulation the network is silent and stays at rest."""
    n = net(np.zeros((10, 10)))
    s = n.run(2000)
    assert not s.any()
    assert np.allclose(n.V, -52.0, atol=1e-3)


def _single_psp(weight, n_steps=3000):
    """PSP amplitude in neuron 1 after EXACTLY one spike from neuron 0."""
    W = np.zeros((2, 2), dtype=np.float32)
    W[0, 1] = weight
    n = net(W)
    n.force_spike([0])
    n.step()
    dev = 0.0
    for _ in range(n_steps):
        n.step()
        if abs(n.V[1] + 52.0) > abs(dev):
            dev = n.V[1] + 52.0
    return dev


def test_single_synapse_psp_amplitude():
    """KEY TEST: one spike through one synapse peaks at 0.275 mV.

    Verifies the psp_scale() normalisation. Without it the amplitude would
    depend on the arbitrary choice of tau_s/tau_m and the published parameter
    would be meaningless.
    """
    psp = _single_psp(1.0)
    assert abs(psp - 0.275) < 0.005, f"PSP = {psp:.4f} mV, expected 0.275"


def test_inhibition_mirrors_excitation():
    """A negative synapse yields an IPSP of equal magnitude, opposite sign."""
    exc, inh = _single_psp(+1.0), _single_psp(-1.0)
    assert exc > 0 and inh < 0
    assert abs(exc + inh) < 1e-3, f"asymmetry: {exc:.4f} vs {inh:.4f}"


def test_refractory_period_caps_rate():
    """At saturation the firing rate never exceeds 1/t_refractory."""
    n = net(np.zeros((1, 1)), t_refractory=0.0022)
    s = n.run(10000, stim_fn=lambda k: np.array([500.0], dtype=np.float32))
    f = rates(s, n.p.dt)[0]
    assert 0 < f <= 1.0 / 0.0022 + 1.0, f"f = {f:.1f} Hz"
    assert f > 300, f"neuron should fire fast, got {f:.1f} Hz"


def test_axonal_delay():
    """The postsynaptic neuron responds no earlier than `delay` after the spike."""
    W = np.zeros((2, 2), dtype=np.float32)
    W[0, 1] = 200.0  # strong connection, guaranteed to drive a spike
    p = LIFParams(delay=0.0018, dt=1e-4)
    n = LIFNetwork(sp.csr_matrix(W), p)
    n.force_spike([0])
    s = n.run(400)
    t_pre = np.argmax(s[:, 0])
    t_post = np.argmax(s[:, 1])
    assert s[:, 0].any() and s[:, 1].any(), "both neurons should fire"
    assert t_post - t_pre >= int(p.delay / p.dt), \
        f"delay was {t_post - t_pre} steps, minimum {int(p.delay / p.dt)}"


def test_synaptic_summation_is_linear():
    """N synapses give an N-fold PSP, as long as threshold is not crossed.

    Threshold sits 7 mV above rest, so up to ~25 synapses we stay in the
    subthreshold linear regime.
    """
    base = _single_psp(1.0)
    for w in (3.0, 10.0, 24.0):
        assert abs(_single_psp(w) / base - w) < 0.05 * w, f"nonlinearity at w={w}"


def test_threshold_triggers_spike():
    """26 synapses (26 * 0.275 = 7.15 mV) cross the 7 mV threshold and fire."""
    W = np.zeros((2, 2), dtype=np.float32)
    W[0, 1] = 26.0
    n = net(W)
    n.force_spike([0])
    n.step()
    s = n.run(2000)
    assert s[:, 1].any(), "the postsynaptic neuron should fire"


def test_determinism():
    """Same seed -> same result."""
    W = sp.random(50, 50, density=0.1, format="csr", dtype=np.float32) * 10
    a = LIFNetwork(W, seed=42)
    b = LIFNetwork(W, seed=42)
    tgt = np.arange(10)
    sa = a.run(500, stim_fn=lambda k: a.poisson(tgt, 200.0))
    sb = b.run(500, stim_fn=lambda k: b.poisson(tgt, 200.0))
    assert np.array_equal(sa, sb)
