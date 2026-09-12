"""LIF engine for FlyWire connectome subnetworks.

Current-based model with exponential synapses, parameters following
Shiu et al. 2024. Key implementation difference: weights are normalised so that
ONE synapse produces a PSP of exactly the configured peak amplitude (0.275 mV by
default). This makes `psp_mv` physically meaningful and unit-testable.

Dynamics (forward Euler, dt <= 0.1 ms):

    I[t+1] = I[t] * exp(-dt/tau_s) + k * (W.T @ s[t - delay])
    V[t+1] = V[t] + dt/tau_m * (V_rest - V[t] + I[t])

where k is chosen so that a unit impulse yields a PSP peak of exactly psp_mv.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class LIFParams:
    """Parameters from Shiu et al. 2024. Times in seconds, voltages in mV."""

    v_rest: float = -52.0
    v_reset: float = -52.0
    v_thresh: float = -45.0
    tau_m: float = 0.020
    tau_s: float = 0.005
    t_refractory: float = 0.0022
    delay: float = 0.0018
    psp_mv: float = 0.275
    dt: float = 1e-4

    def psp_scale(self) -> float:
        """Scaling factor k: normalises a unit impulse to a peak of psp_mv.

        For the bi-exponential kernel the peak occurs at
            t* = tau_m*tau_s/(tau_m - tau_s) * ln(tau_m/tau_s)
        """
        tm, ts = self.tau_m, self.tau_s
        if abs(tm - ts) < 1e-12:
            raise ValueError("tau_m == tau_s: degenerate case")
        t_peak = tm * ts / (tm - ts) * np.log(tm / ts)
        peak = ts / (tm - ts) * (np.exp(-t_peak / tm) - np.exp(-t_peak / ts))
        return self.psp_mv / peak


class LIFNetwork:
    """LIF network over a sparse weight matrix.

    Parameters
    ----------
    W : scipy.sparse matrix [N, N]
        W[i, j] = influence of neuron i on neuron j, in units of "signed synapse
        count" (positive = excitatory, negative = inhibitory).
    params : LIFParams
    seed : int | None
        Seed for Poisson stimulation. None = non-deterministic.
    """

    def __init__(self, W, params: LIFParams | None = None, seed: int | None = 0):
        self.p = params or LIFParams()
        self.W = sp.csr_matrix(W, dtype=np.float32)
        if self.W.shape[0] != self.W.shape[1]:
            raise ValueError(f"W must be square, got {self.W.shape}")
        self.N = self.W.shape[0]
        self.WT = sp.csr_matrix(self.W.T)  # transpose once, not inside the loop
        self.k = np.float32(self.p.psp_scale())
        self._decay_s = np.float32(np.exp(-self.p.dt / self.p.tau_s))
        self._n_delay = max(1, int(round(self.p.delay / self.p.dt)))
        self._rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        p, N = self.p, self.N
        self.V = np.full(N, p.v_rest, dtype=np.float32)
        self.I = np.zeros(N, dtype=np.float32)
        self.refr = np.zeros(N, dtype=np.float32)
        # ring buffer for axonal delays
        self._buf = np.zeros((self._n_delay, N), dtype=np.float32)
        self._forced = np.zeros(N, dtype=bool)
        self._head = 0
        self.t = 0.0

    def force_spike(self, targets) -> None:
        """Force a spike in the given neurons on the next step, bypassing the
        threshold. This is the optogenetic-click equivalent - useful for tests
        and for ablation experiments."""
        self._forced[np.asarray(targets)] = True

    def step(self, i_ext: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt. Returns a bool mask [N] of neurons that fired.

        i_ext : external drive [N], scaled like synaptic input, i.e. 1.0 is
                equivalent to one synapse before normalisation.
        """
        p = self.p
        # 1. spikes emitted `delay` ago arrive at synapses
        delayed = self._buf[self._head]
        self.I *= self._decay_s
        if delayed.any():
            self.I += self.k * self.WT.dot(delayed)
        if i_ext is not None:
            self.I += self.k * i_ext.astype(np.float32)

        # 2. membrane integration, only outside the refractory period
        active = self.refr <= 0.0
        dv = (p.dt / p.tau_m) * (p.v_rest - self.V + self.I)
        self.V[active] += dv[active]
        self.V[~active] = p.v_reset

        # 3. thresholding
        fired = (active & (self.V >= p.v_thresh)) | self._forced
        self._forced[:] = False
        self.V[fired] = p.v_reset
        self.refr[fired] = p.t_refractory
        self.refr -= p.dt

        # 4. write into the delay buffer
        self._buf[self._head] = fired.astype(np.float32)
        self._head = (self._head + 1) % self._n_delay
        self.t += p.dt
        return fired

    def poisson(self, targets: np.ndarray, rate_hz: float) -> np.ndarray:
        """External drive modelling optogenetic activation.

        Returns a vector [N] with unit-weight impulses at random steps such
        that the mean rate equals rate_hz.
        """
        out = np.zeros(self.N, dtype=np.float32)
        if rate_hz <= 0 or len(targets) == 0:
            return out
        pr = min(1.0, rate_hz * self.p.dt)
        hit = self._rng.random(len(targets)) < pr
        out[np.asarray(targets)[hit]] = 1.0
        return out

    def run(
        self,
        n_steps: int,
        stim_fn=None,
        record: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run n_steps steps.

        stim_fn : callable(step_index) -> i_ext [N] or None
        record  : neuron indices to record; None = all
        Returns a bool array [n_steps, len(record)].
        """
        idx = np.arange(self.N) if record is None else np.asarray(record)
        out = np.zeros((n_steps, len(idx)), dtype=bool)
        for n in range(n_steps):
            ext = stim_fn(n) if stim_fn is not None else None
            out[n] = self.step(ext)[idx]
        return out


def rates(spikes: np.ndarray, dt: float) -> np.ndarray:
    """Mean firing rate [Hz] per neuron from a [n_steps, N] spike array."""
    return spikes.sum(axis=0) / (spikes.shape[0] * dt)
