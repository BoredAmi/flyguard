"""Synthetic optic-flow stimuli and the radial-motion-opponency test.

Layer 1 of the architecture (see the project notes): pure numpy, no connectome, no
MuJoCo. This tests the core hypothesis -- that LPLC2's four-arm radial
motion opponency discriminates looming from self-motion translation --
*before* any image encoder or rendering is involved.

Channel naming: T4/T5 subtypes a/b/c/d each prefer one cardinal direction.
The a/b/c/d -> direction mapping is now resolved (the project notes "RESOLVED"):
a=front-to-back, b=back-to-front, c=upward, d=downward. This module's ring
model is still a 1-D abstraction (four evenly-spaced channels at arbitrary
angles 0/90/180/270), not a real 2-D visual field, so the angle assignment
below is a bookkeeping choice, not a geometric claim -- but the *labels* are
now the real subtype letters rather than placeholders. Per the project notes, the
front-to-back/up/down convention is body-centric and it is not yet confirmed
whether it is mirrored between the left and right optic lobe; that is
settled by the per-hemisphere replication test, not assumed here.

Model of one LPLC2 arm (Klapoetke et al. 2017, radial motion opponency):
each arm sits in a retinotopic zone of the visual field and receives
excitation from T4/T5 units locally tuned to the outward direction (away
from the visual field center) and inhibition from LPi units locally tuned
to the inward direction. A uniformly expanding flow field is, by
definition, outward everywhere -- so it drives every arm's excitatory
channel and none of the inhibitory channels: the four arms agree. A
uniform translational flow field points in one fixed direction regardless
of retinotopic position, so it is outward for some arms and inward for
others: the four arms conflict, and summed opponency drive is exactly zero
by symmetry (see `test_stimuli.py`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from flyguard.lif import LIFNetwork, LIFParams, rates

# T4/T5 subtype letters, four evenly spaced channels in the ring model.
# Direction labels per the project notes "RESOLVED -- a/b/c/d to cardinal direction
# mapping" (Maisak et al. 2013, via Fisher et al. and others citing it
# directly). Angle assignment (0/90/180/270) is an arbitrary but fixed
# bookkeeping choice for this abstraction, not a geometric claim.
SUBTYPE_NAMES = ("a", "b", "c", "d")
SUBTYPE_DIRECTIONS = {
    "a": "front_to_back",
    "b": "back_to_front",
    "c": "upward",
    "d": "downward",
}
SUBTYPE_DIRS_DEG = (0.0, 90.0, 180.0, 270.0)


def visual_field_positions(n: int = 360) -> np.ndarray:
    """`n` retinotopic columns evenly spaced around the visual field, in
    degrees. Each position's "outward" direction (away from the field
    center) is, by construction, equal to its own angle."""
    return np.linspace(0.0, 360.0, n, endpoint=False)


def expanding_flow(positions_deg: np.ndarray) -> np.ndarray:
    """Local flow direction for a uniformly expanding (looming) stimulus:
    outward everywhere, i.e. equal to the position's own angle."""
    return positions_deg.copy()


def translational_flow(positions_deg: np.ndarray, direction_deg: float) -> np.ndarray:
    """Local flow direction for pure self-motion translation: the same
    fixed direction at every retinotopic position."""
    return np.full_like(positions_deg, direction_deg)


def noise_flow(positions_deg: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Local flow direction for spatially incoherent motion noise: an
    independent, uniformly random direction at every retinotopic position.
    Neither looming nor translation -- a negative control. Radial opponency
    predicts arms see uncorrelated exc/inh drive and the net response stays
    near the translation floor, well below looming."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 360.0, size=positions_deg.shape)


def _angle_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    """Signed shortest difference a - b, wrapped to (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def arm_responses(
    flow_deg: np.ndarray,
    positions_deg: np.ndarray,
    subtype_dirs_deg=SUBTYPE_DIRS_DEG,
    zone_width_deg: float = 90.0,
) -> list[tuple[float, float]]:
    """Excitatory / inhibitory drive for each of the four LPLC2 arms.

    Each arm `k` pools over the retinotopic zone within `zone_width_deg/2`
    of its own preferred direction (its dendritic layer's territory).
    Within that zone: excitation is rectified-cosine tuned to the outward
    (preferred) direction, inhibition is rectified-cosine tuned to the
    opposite (inward) direction.

    Returns a list of (exc, inh) pairs, one per arm.
    """
    out = []
    for d_k in subtype_dirs_deg:
        zone = np.abs(_angle_diff_deg(positions_deg, d_k)) <= zone_width_deg / 2.0
        if not zone.any():
            out.append((0.0, 0.0))
            continue
        local_flow = flow_deg[zone]
        exc = np.clip(np.cos(np.radians(_angle_diff_deg(local_flow, d_k))), 0, None).mean()
        inh = np.clip(np.cos(np.radians(_angle_diff_deg(local_flow, d_k + 180.0))), 0, None).mean()
        out.append((float(exc), float(inh)))
    return out


def net_opponency_drive(arm_drives: list[tuple[float, float]]) -> float:
    """Summed (excitation - inhibition) across all arms."""
    return sum(exc - inh for exc, inh in arm_drives)


@dataclass(frozen=True)
class OpponencyIndices:
    t4t5: np.ndarray
    lpi: np.ndarray
    lplc2: int


def build_opponency_network(w_exc: float = 8.0, w_inh: float = 8.0):
    """A minimal 9-neuron circuit embodying radial motion opponency:

        T4T5_k --(+w_exc)--> LPLC2   for k in 0..3   (outward-tuned excitation)
        LPi_k  --(-w_inh)--> LPLC2   for k in 0..3   (inward-tuned inhibition)

    Each T4T5_k / LPi_k neuron stands in for one arm's local population
    (already pooled by `arm_responses`), not an individual real neuron.
    Weights are chosen so that all four arms firing in agreement (~4*w_exc)
    clears the ~26-synapse / 7 mV threshold (see the project notes "Known gotcha"),
    while a single arm alone does not.
    """
    n = 9
    idx = OpponencyIndices(t4t5=np.arange(0, 4), lpi=np.arange(4, 8), lplc2=8)
    W = sp.lil_matrix((n, n), dtype=np.float32)
    for k in range(4):
        W[idx.t4t5[k], idx.lplc2] = w_exc
        W[idx.lpi[k], idx.lplc2] = -w_inh
    return sp.csr_matrix(W), idx


def simulate_condition(
    flow_deg: np.ndarray,
    positions_deg: np.ndarray,
    n_steps: int = 20000,
    base_rate_hz: float = 250.0,
    w_exc: float = 8.0,
    w_inh: float = 8.0,
    ablate_lpi: bool = False,
    seed: int = 0,
    params: LIFParams | None = None,
) -> float:
    """Drive the opponency circuit with a flow field and return the LPLC2
    output firing rate in Hz.

    Each arm's T4T5/LPi relay neuron is forced to spike (optogenetic-style,
    via `force_spike`) with instantaneous probability `drive * base_rate_hz
    * dt`, i.e. it fires as a Poisson process whose rate is proportional to
    the arm's excitatory/inhibitory tuning strength computed by
    `arm_responses`.

    `ablate_lpi=True` deletes the inhibitory LPi->LPLC2 connections
    (w_inh forced to 0) -- the "delete a cell type, see what breaks"
    ablation from the project's task 5. Without opponency, LPLC2 should lose
    its ability to discriminate looming from translation/noise and fire at
    (almost) everything with coherent local motion.
    """
    p = params or LIFParams()
    if ablate_lpi:
        w_inh = 0.0
    W, idx = build_opponency_network(w_exc, w_inh)
    net = LIFNetwork(W, p, seed=seed)
    drives = arm_responses(flow_deg, positions_deg)
    exc = np.array([e for e, _ in drives])
    inh = np.array([i for _, i in drives])
    rng = np.random.default_rng(seed)

    def stim_fn(_step):
        r_exc = rng.random(4) < np.clip(exc * base_rate_hz * p.dt, 0, 1)
        r_inh = rng.random(4) < np.clip(inh * base_rate_hz * p.dt, 0, 1)
        if r_exc.any():
            net.force_spike(idx.t4t5[r_exc])
        if r_inh.any():
            net.force_spike(idx.lpi[r_inh])
        return None

    spikes = net.run(n_steps, stim_fn=stim_fn, record=np.array([idx.lplc2]))
    return float(rates(spikes, p.dt)[0])


def main():
    positions = visual_field_positions()

    exc_loom = arm_responses(expanding_flow(positions), positions)
    print("looming arm (exc, inh):", [(round(e, 3), round(i, 3)) for e, i in exc_loom])
    print("looming net opponency drive:", round(net_opponency_drive(exc_loom), 4))

    for d in (0.0, 45.0, 90.0, 180.0):
        drives = arm_responses(translational_flow(positions, d), positions)
        print(f"translation({d:>5.0f} deg) arm (exc, inh):",
              [(round(e, 3), round(i, 3)) for e, i in drives],
              " net:", round(net_opponency_drive(drives), 4))

    noise_drives = arm_responses(noise_flow(positions, seed=0), positions)
    print("noise arm (exc, inh):", [(round(e, 3), round(i, 3)) for e, i in noise_drives],
          " net:", round(net_opponency_drive(noise_drives), 4))

    print("\nLIF circuit firing rates (LPLC2 output), intact opponency:")
    r_loom = simulate_condition(expanding_flow(positions), positions, seed=1)
    print(f"  looming:     {r_loom:.1f} Hz")
    for d in (0.0, 90.0, 180.0):
        r_trans = simulate_condition(translational_flow(positions, d), positions, seed=1)
        print(f"  translation({d:>3.0f} deg): {r_trans:.1f} Hz")
    r_noise = simulate_condition(noise_flow(positions, seed=0), positions, seed=1)
    print(f"  noise:       {r_noise:.1f} Hz")

    print("\nLIF circuit firing rates, LPi ABLATED (opponency removed):")
    r_loom_abl = simulate_condition(expanding_flow(positions), positions, seed=1, ablate_lpi=True)
    print(f"  looming:     {r_loom_abl:.1f} Hz")
    for d in (0.0, 90.0, 180.0):
        r_trans_abl = simulate_condition(translational_flow(positions, d), positions, seed=1, ablate_lpi=True)
        print(f"  translation({d:>3.0f} deg): {r_trans_abl:.1f} Hz")
    r_noise_abl = simulate_condition(noise_flow(positions, seed=0), positions, seed=1, ablate_lpi=True)
    print(f"  noise:       {r_noise_abl:.1f} Hz")


if __name__ == "__main__":
    main()
