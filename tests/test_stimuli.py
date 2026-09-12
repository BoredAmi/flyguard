import numpy as np
import pytest

from flyguard.stimuli import (
    SUBTYPE_DIRECTIONS,
    SUBTYPE_DIRS_DEG,
    SUBTYPE_NAMES,
    arm_responses,
    expanding_flow,
    net_opponency_drive,
    noise_flow,
    simulate_condition,
    translational_flow,
    visual_field_positions,
)


def test_expanding_flow_drives_all_arms_with_no_inhibition():
    """Looming: every arm sees pure outward motion, so inhibition is zero
    and all four arms agree (equal, positive excitation)."""
    positions = visual_field_positions()
    drives = arm_responses(expanding_flow(positions), positions)
    exc = [e for e, _ in drives]
    inh = [i for _, i in drives]
    assert all(i == pytest.approx(0.0, abs=1e-9) for i in inh)
    assert all(e > 0.5 for e in exc)
    assert max(exc) - min(exc) < 1e-9, "all four arms should agree exactly"


@pytest.mark.parametrize("direction_deg", [0.0, 30.0, 45.0, 90.0, 137.0, 200.0, 315.0])
def test_translation_sums_to_exactly_zero_opponency_drive(direction_deg):
    """Uniform translation, at ANY heading, drives the four 90-degree-spaced
    arms into exact conflict: sum(exc - inh) == 0 by symmetry, since the
    four rectified-cosine channels are evenly spaced around the circle.
    This is the core discrimination claim from CLAUDE.md, checked
    analytically (no spiking involved)."""
    positions = visual_field_positions()
    drives = arm_responses(translational_flow(positions, direction_deg), positions)
    assert net_opponency_drive(drives) == pytest.approx(0.0, abs=1e-9)


def test_looming_opponency_drive_is_positive_and_larger_than_any_translation():
    positions = visual_field_positions()
    loom_drive = net_opponency_drive(arm_responses(expanding_flow(positions), positions))
    assert loom_drive > 0
    for d in np.linspace(0, 360, 12, endpoint=False):
        trans_drive = net_opponency_drive(arm_responses(translational_flow(positions, d), positions))
        assert loom_drive > trans_drive


def test_subtype_channels_are_four_distinct_and_labelled():
    """Structural property the circuit actually needs: four mutually
    distinct channels. Direction labels are now resolved (CLAUDE.md
    RESOLVED section) and should match a/b/c/d -> front-to-back /
    back-to-front / upward / downward."""
    assert len(SUBTYPE_DIRS_DEG) == 4
    assert len(set(SUBTYPE_DIRS_DEG)) == 4
    assert SUBTYPE_NAMES == ("a", "b", "c", "d")
    assert SUBTYPE_DIRECTIONS == {
        "a": "front_to_back",
        "b": "back_to_front",
        "c": "upward",
        "d": "downward",
    }


def test_noise_gives_near_zero_opponency_drive():
    """Spatially incoherent motion noise should, like translation, fail to
    coherently drive the four opponency arms -- net drive should sit near
    zero, far below looming's."""
    positions = visual_field_positions()
    loom_drive = net_opponency_drive(arm_responses(expanding_flow(positions), positions))
    noise_drive = net_opponency_drive(arm_responses(noise_flow(positions, seed=0), positions))
    assert abs(noise_drive) < 0.2
    assert noise_drive < loom_drive / 5


def test_lif_circuit_noise_stays_near_translation_floor():
    """Noise should not be confused with looming: its LPLC2 output rate
    should be well below looming and comparable to the translation floor."""
    positions = visual_field_positions()
    r_loom = simulate_condition(expanding_flow(positions), positions, n_steps=15000, seed=1)
    r_noise = simulate_condition(noise_flow(positions, seed=0), positions, n_steps=15000, seed=1)
    assert r_noise < r_loom / 3


def test_lpi_ablation_breaks_discrimination():
    """CLAUDE.md task 5: deleting a cell type (here, LPi) should make the
    detector fire at everything -- looming and translation/noise should
    converge once opponency is removed, and non-looming rates should rise
    sharply relative to the intact circuit."""
    positions = visual_field_positions()
    r_loom_intact = simulate_condition(expanding_flow(positions), positions, n_steps=15000, seed=1)
    r_trans_intact = simulate_condition(
        translational_flow(positions, 0.0), positions, n_steps=15000, seed=1
    )

    r_loom_ablated = simulate_condition(
        expanding_flow(positions), positions, n_steps=15000, seed=1, ablate_lpi=True
    )
    r_trans_ablated = simulate_condition(
        translational_flow(positions, 0.0), positions, n_steps=15000, seed=1, ablate_lpi=True
    )

    # Looming drive is unaffected -- LPi never fires for pure outward flow.
    assert r_loom_ablated == pytest.approx(r_loom_intact, rel=0.05)
    # Translation is no longer suppressed: it should rise sharply.
    assert r_trans_ablated > 3 * r_trans_intact
    # Discrimination margin collapses relative to the intact circuit.
    intact_margin = r_loom_intact - r_trans_intact
    ablated_margin = r_loom_ablated - r_trans_ablated
    assert ablated_margin < intact_margin * 0.8


def test_lif_circuit_discriminates_looming_from_translation():
    """End-to-end spiking check: the LPLC2 output neuron in the radial
    opponency circuit fires much faster for looming than for translation,
    at every translation heading."""
    positions = visual_field_positions()
    r_loom = simulate_condition(expanding_flow(positions), positions, n_steps=15000, seed=1)
    assert r_loom > 50, f"looming should drive LPLC2 to a clear spike rate, got {r_loom:.1f} Hz"

    for d in (0.0, 90.0, 180.0, 270.0):
        r_trans = simulate_condition(translational_flow(positions, d), positions, n_steps=15000, seed=1)
        assert r_trans < r_loom / 3, (
            f"translation({d} deg) = {r_trans:.1f} Hz should be well below "
            f"looming = {r_loom:.1f} Hz"
        )


def test_simulate_condition_is_deterministic():
    positions = visual_field_positions()
    flow = expanding_flow(positions)
    a = simulate_condition(flow, positions, n_steps=2000, seed=7)
    b = simulate_condition(flow, positions, n_steps=2000, seed=7)
    assert a == b
