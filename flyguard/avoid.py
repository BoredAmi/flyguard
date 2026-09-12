"""Closed-loop obstacle avoidance driven by the real connectome.

The agent drives down a corridor of obstacles (`flyguard.arena`); every tick
it renders its own eye view, estimates optical flow, feeds each half of the
visual field into that hemisphere's real T4/T5 columns, simulates the real
530-neuron core circuit, and steers on what comes out. Nothing is trained.

## The honest caveat, stated once and loudly

**Steering is not in the connectome.** This repo already established, from
the extracted subnetwork rather than from assumption, that LPLC2/LC4 project
onto DNp01/DNp02/DNp04/DNp11 -- the Giant Fiber escape pathway -- with
*zero* direct edges onto DNa01, DNa02 or MDN, the descending neurons that
steer and reverse. The measured circuit can say "stop", not "turn left".

So a closed-loop avoider needs a steering signal the anatomy does not
supply, and `bilateral_turn` below is it: compare the two hemispheres'
LPLC2 population rates and turn away from the louder one. The *ingredients*
are real -- two separately-reconstructed optic lobes, each with its own
retinotopic T4/T5 population and its own LPLC2 pool, wired to its own
DNp01 -- and the fly really does bias its escape direction by which eye
sees the loom. But the **comparison operator is an engineering addition**,
on the same footing as the direction convention: an assumption this project
makes explicit rather than a pathway it read off the wiring diagram. Do not
report closed-loop avoidance as "the connectome steers the robot".

## What the baselines are for

Three controllers run on identical arenas:

* `straight` -- ignores vision entirely. Measures how hard the arena is. A
  controller that cannot beat this has done nothing.
* `flow` -- same camera, same optical flow, same hemifield split, same
  steering law, but the turn signal comes straight from the pooled flow
  drive with no spiking circuit in between. This is the baseline that
  matters, because it isolates the only question worth asking here: what
  does simulating 530 real neurons add over the signal you fed them?
* `connectome` -- the real circuit.

Lee's tau is deliberately absent. It estimates time-to-contact for a single
expanding object from its apparent size; generalising it to "which of
several obstacles in a cluttered corridor, and on which side" is a separate
piece of work, not a baseline that can be dropped in honestly.

Usage:
    MUJOCO_GL=egl python -m flyguard.avoid --columns ~/flywire/column_assignment.csv \\
        --n-arenas 8 --json-out data/avoid_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from flyguard.paths import default_columns_path
from flyguard.arena import (
    AgentState,
    ArenaConfig,
    ArenaRenderer,
    clearance,
    collides,
    sample_obstacles,
    step_agent,
)
from flyguard.runtime import (
    BilateralEncoder,
    Command,
    CoreCircuit,
    SaccadicSteering,
    as_sides,
    bilateral_turn,
    normalize_drive,
)
from flyguard.runtime.circuit import BASE_CORE_TYPES

#: Kept as a module-level name because scripts and notebooks import it.
CORE_TYPES = list(BASE_CORE_TYPES)


# ---------------------------------------------------------------------------
# Perception (`BilateralEncoder`) and the steering law (`bilateral_turn`,
# `Command`) now live in `flyguard.runtime`, so this benchmark, the ROS2
# nodes and the recorder all run one implementation instead of three
# hand-synchronised copies. They are imported at the top of this module and
# re-exported here, because scripts and tests import them from `flyguard.avoid`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


class StraightController:
    """Vision-free control. Establishes how hard the arena is."""

    name = "straight"
    needs_vision = False

    def __init__(self, v_cruise: float = 1.0, **_):
        self.v_cruise = v_cruise
        self.turn_offset = 0.0
        self.estop_threshold = float("inf")

    def reset(self):
        pass

    def __call__(self, percept: dict | None, t: float) -> Command:
        return Command(v=self.v_cruise, omega=0.0)


class _BilateralController:
    """Shared machinery for the two vision controllers.

    Both are given **identical** gains, identical normalisation and identical
    calibration, so the only difference between them is whether 530 real
    neurons sit between the flow estimate and the turn command. Anything else
    would make the comparison meaningless.

    Three calibrated quantities, all measured by `calibrate_controller` on
    scripted straight runs rather than chosen by hand:

    * `turn_offset` -- the resting left/right imbalance, subtracted before
      steering. FAFB's two optic lobes are not reconstructed equally (this
      circuit has 108 vs 102 LPLC2 and 85 vs 129 LPi), so the bilateral
      difference has a non-zero zero-point. Nulling a differential sensor's
      offset is ordinary practice; leaving it in would just mean the agent
      drives in a circle. The measured value is reported, not hidden.
    * `saccade_threshold` -- how much turn evidence is needed to commit to a
      turn, set from the signal's own noise floor in an empty corridor.
    * `estop_threshold` -- the level the escape statistic must exceed. An
      escape that fires on almost every tick is not a detection, and
      `validate_arena` measures why it would: the absolute drive level sits
      barely above the self-motion floor, far below the contrast available in
      the left/right difference.

    The gains that are *not* calibrated -- cruise speed, saccade rate and
    duration, refractory period -- are ordinary robot-controller constants,
    shared identically by both controllers. They are not claims about the
    fly, and the connectome path itself still contains no fitted parameter.
    """

    needs_vision = True

    def __init__(self, drive_center, drive_scale: float, v_cruise: float = 1.0,
                 v_escape: float = 0.3, saccade_rate: float = 1.6,
                 saccade_duration_s: float = 0.3, refractory_s: float = 0.3,
                 estop_cooldown_s: float = 0.4, **_):
        self.drive_center = as_sides(drive_center)
        self.drive_scale = float(drive_scale)
        self.steering = SaccadicSteering(
            v_cruise=v_cruise, v_escape=v_escape, saccade_rate=saccade_rate,
            saccade_duration_s=saccade_duration_s, refractory_s=refractory_s,
            estop_cooldown_s=estop_cooldown_s)

    # The three calibrated quantities live on the steering object now;
    # `calibrate_controller` and `main` still assign them here by name.
    def _threshold_property(name):
        return property(lambda self: getattr(self.steering, name),
                        lambda self, v: setattr(self.steering, name, v))

    turn_offset = _threshold_property("turn_offset")
    saccade_threshold = _threshold_property("saccade_threshold")
    estop_threshold = _threshold_property("estop_threshold")
    v_cruise = _threshold_property("v_cruise")
    v_escape = _threshold_property("v_escape")
    del _threshold_property

    def _norm(self, side: str, d: float) -> float:
        return normalize_drive(d, self.drive_center[side], self.drive_scale)

    def _command(self, turn_raw: float, estop_stat: float, t: float,
                 telemetry: dict) -> Command:
        return self.steering.step(turn_raw, estop_stat, t, telemetry)

    def _reset_state(self):
        self.steering.reset()




#: Re-exported for the tests that import it from here.
_as_sides = as_sides


class FlowController(_BilateralController):
    """Same eyes, same hemifield split, same steering law and same gains as
    the connectome controller -- but the turn signal is the pooled T4/T5
    drive itself, with no spiking circuit in between. This is the baseline
    that isolates what simulating 530 real neurons actually adds."""

    name = "flow"

    def reset(self):
        self._reset_state()

    def __call__(self, percept: dict | None, t: float) -> Command:
        if percept is None:
            return Command(v=self.v_cruise, omega=0.0)
        dl = self._norm("left", percept["drive_left"])
        dr = self._norm("right", percept["drive_right"])
        # Mean, not max: the normalised drive clips at 1.0, and a max-based
        # statistic therefore pins at the ceiling whenever either side clips.
        return self._command(bilateral_turn(dl, dr), 0.5 * (dl + dr), t,
                             {"norm_left": dl, "norm_right": dr})


class ConnectomeController(_BilateralController):
    """The real 530-neuron core circuit (LPLC2 + LC4 + all 11 LPi subtypes +
    DNp01), both hemispheres, in the loop.

    Each hemisphere's LPLC2 pool is driven by its own hemifield's T4/T5
    drive and its own LPi pool by the complement `(1 - drive)`, the same
    dual-channel scheme `flyguard.stimuli.simulate_condition` and the ROS2
    `looming_node` already use -- LPi should be most active exactly when the
    view looks least like a loom. Steering reads the two LPLC2 population
    rates; the e-stop reads real DNp01 spikes, as in the ROS2 node.
    """

    name = "connectome"

    def __init__(self, drive_center, drive_scale, npz_path: Path | None = None,
                 base_rate_hz: float = 250.0, peak_weight: float = 30.0,
                 tick_hz: float = 10.0, seed: int = 1, **kwargs):
        super().__init__(drive_center, drive_scale, **kwargs)
        self.circuit = CoreCircuit(
            npz_path if npz_path is not None else Path("data/looming.npz"),
            tick_hz=tick_hz, base_rate_hz=base_rate_hz, peak_weight=peak_weight,
            seed=seed, record=("LPLC2", "DNp01"), bilateral=True)
        self.reset()

    @property
    def n_neurons(self) -> int:
        return self.circuit.n_neurons

    def reset(self):
        self.circuit.reset()
        self._reset_state()

    def __call__(self, percept: dict | None, t: float) -> Command:
        if percept is None:
            return Command(v=self.v_cruise, omega=0.0)
        dl = self._norm("left", percept["drive_left"])
        dr = self._norm("right", percept["drive_right"])

        response = self.circuit.step({"left": dl, "right": dr})
        rate_l = response.rate_hz("LPLC2", "left")
        rate_r = response.rate_hz("LPLC2", "right")
        # The escape statistic is DNp01's spike *count* this tick, not "did it
        # spike at all". A single Giant Fiber spike in a 100 ms window is not a
        # commitment to escape, and with both hemispheres driven hard the
        # any-spike test fires essentially every tick (measured: 26/30).
        dnp01_spikes = response.spike_count("DNp01")

        return self._command(bilateral_turn(rate_l, rate_r), float(dnp01_spikes), t,
                             {"norm_left": dl, "norm_right": dr,
                              "lplc2_left_hz": rate_l, "lplc2_right_hz": rate_r,
                              "dnp01_spikes": dnp01_spikes})



CONTROLLERS = {c.name: c for c in (StraightController, FlowController, ConnectomeController)}


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    controller: str
    arena_seed: int
    collided: bool
    reached_goal: bool
    ticks: int
    progress_x: float
    path_length: float
    min_clearance: float
    n_estop_ticks: int
    telemetry: list = field(default_factory=list)


def run_trial(controller, cfg: ArenaConfig, encoder: BilateralEncoder | None,
              resolution: int = 128, tick_hz: float = 10.0, max_ticks: int = 320,
              keep_telemetry: bool = False, force_straight: bool = False,
              frame_sink=None) -> TrialResult:
    """One traversal. Ends on collision, on reaching the goal, or on timeout.

    The agent's pose is integrated in closed form and collisions are checked
    geometrically, so the trial is fully reproducible: the only stochastic
    element is the LIF network's Poisson drive, which is seeded.

    `frame_sink`, if given, is called as `frame_sink(tick, frame, state)` for
    every rendered frame. It exists so a recorder can capture the agent's eye
    view without reimplementing this loop -- if the recording and the
    benchmark ever ran different code, the recording would stop being an
    honest picture of what was measured.

    `force_straight` still runs the controller and still records what it
    *would* have commanded, but drives straight regardless. That is how
    `calibrate_controller` observes a controller's resting output on a fixed,
    open-loop trajectory -- measuring the zero-point in closed loop would let
    the offset being measured steer the very trajectory used to measure it.
    """
    obstacles = sample_obstacles(cfg)
    controller.reset()
    dt = 1.0 / tick_hz
    state = AgentState(0.0, 0.0, 0.0)
    prev_frame = None
    path_length = 0.0
    min_clear = float("inf")
    n_estop = 0
    telemetry = []

    renderer = ArenaRenderer(obstacles, cfg, resolution) if controller.needs_vision else None
    try:
        for tick in range(max_ticks):
            t = tick * dt
            percept = None
            if renderer is not None:
                frame = renderer.render(state)
                if frame_sink is not None:
                    frame_sink(tick, frame, state)
                if prev_frame is not None:
                    percept = encoder(prev_frame, frame)
                prev_frame = frame

            cmd = controller(percept, t)
            if cmd.estop:
                n_estop += 1
            if keep_telemetry:
                telemetry.append({"t": round(t, 3), "x": round(state.x, 3), "y": round(state.y, 3),
                                  "heading": round(state.heading, 4), "v": cmd.v,
                                  "omega": round(cmd.omega, 4), "estop": cmd.estop,
                                  **{k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in cmd.telemetry.items()}})

            if force_straight:
                state = step_agent(state, controller.v_cruise, 0.0, dt)
                path_length += controller.v_cruise * dt
            else:
                state = step_agent_clamped(state, cmd.v, cmd.omega, dt, cfg)
                path_length += cmd.v * dt
            min_clear = min(min_clear, clearance(state, obstacles, cfg))

            if collides(state, obstacles, cfg):
                return TrialResult(controller.name, cfg.seed, True, False, tick + 1,
                                   state.x, path_length, min_clear, n_estop, telemetry)
            if state.x >= cfg.goal_x:
                return TrialResult(controller.name, cfg.seed, False, True, tick + 1,
                                   state.x, path_length, min_clear, n_estop, telemetry)
    finally:
        if renderer is not None:
            renderer.close()

    return TrialResult(controller.name, cfg.seed, False, False, max_ticks,
                       state.x, path_length, min_clear, n_estop, telemetry)


def step_agent_clamped(state: AgentState, v: float, omega: float, dt: float,
                        cfg: ArenaConfig) -> AgentState:
    """`arena.step_agent` with the heading held within +-80 degrees of down
    -corridor.

    Without this an agent that decides to turn can simply spin on the spot or
    drive back the way it came and time out, which scores as "did not
    collide" and flatters a controller that has in fact given up. Clamping
    makes a trial end in one of two honest ways: through the corridor, or
    into something.
    """
    nxt = step_agent(state, v, omega, dt)
    limit = math.radians(80.0)
    return AgentState(nxt.x, nxt.y, float(np.clip(nxt.heading, -limit, limit)))


# ---------------------------------------------------------------------------
# Drive calibration
# ---------------------------------------------------------------------------


def calibrate_drive(encoder: BilateralEncoder, cfg_template: ArenaConfig, seeds,
                    resolution: int = 128, tick_hz: float = 10.0,
                    v_cruise: float = 0.8, max_ticks: int = 120) -> dict:
    """Measure the operating range of the pooled T4/T5 drive by driving
    straight down a few arenas and taking percentiles of what comes back.

    The normalisation the controllers use has to be anchored to *something*;
    `record_live_demo.py` and the ROS2 `vision_node` anchor theirs to hand
    -measured means from a different scene, which does not transfer here.
    Two scripted passes give both anchors a plain meaning:

    * **centre**, per hemisphere, from an **empty** corridor -- "what an
      unobstructed view looks like". Cruising then sits at drive ~ 0.5, which
      keeps both the LPLC2 channel and LPi's `(1 - drive)` channel in range.
      Centring on the obstacle arenas instead put ordinary cruising near 0,
      where both LPLC2 pools fall nearly silent and the steering ratio
      becomes two small noisy numbers divided by each other -- measured as a
      heavy-tailed turn signal whose p99 noise was 0.232 against a median of
      0.037.
    * **scale**, shared, from the **obstacle** arenas -- the range real
      obstacles actually produce.

    The centre is per-hemisphere because the two optic lobes are not
    reconstructed identically and their pooled drives sit at measurably
    different levels. The scale is deliberately shared: see `_norm`.
    """
    def _scan(cfg_d) -> dict:
        cfg = ArenaConfig(**cfg_d)
        obstacles = sample_obstacles(cfg)
        state = AgentState(0.0, 0.0, 0.0)
        prev = None
        out = {"left": [], "right": []}
        with ArenaRenderer(obstacles, cfg, resolution) as renderer:
            for tick in range(max_ticks):
                frame = renderer.render(state)
                if prev is not None:
                    p = encoder(prev, frame)
                    out["left"].append(p["drive_left"])
                    out["right"].append(p["drive_right"])
                prev = frame
                state = step_agent(state, v_cruise, 0.0, 1.0 / tick_hz)
                if state.x >= cfg.goal_x or collides(state, obstacles, cfg):
                    break
        return out

    empty_cfg = {**cfg_template.__dict__, "n_obstacles": 0,
                 "corridor_length": cfg_template.goal_x}
    empty = {"left": [], "right": []}
    samples = {"left": [], "right": []}
    for seed in seeds:
        for target, cfg_d in ((empty, {**empty_cfg, "seed": seed}),
                              (samples, {**cfg_template.__dict__, "seed": seed})):
            got = _scan(cfg_d)
            for side in ("left", "right"):
                target[side].extend(got[side])

    centers, deviations = {}, []
    for side in ("left", "right"):
        centers[side] = float(np.median(empty[side]))
        deviations.append(np.abs(np.array(samples[side]) - centers[side]))
    # One shared half-range, set so ~10% of calibration samples clip. A
    # narrower band spends most of its time pinned at 0 or 1 and throws the
    # left-right difference away (measured: turn median 0.41 vs mean 0.05);
    # a much wider one leaves LPi's (1 - drive) channel with no contrast.
    scale = float(np.percentile(np.concatenate(deviations), 90))
    return {"n_samples": len(samples["left"]) + len(samples["right"]),
            "n_empty_samples": len(empty["left"]) + len(empty["right"]),
            "drive_center": centers, "drive_scale": scale,
            "side_offset": float(centers["right"] - centers["left"])}


def calibrate_controller(controller, encoder: BilateralEncoder, cfg_template: ArenaConfig,
                          seeds, resolution: int = 128, tick_hz: float = 10.0,
                          max_ticks: int = 400, estop_percentile: float = 99.0,
                          saccade_percentile: float = 99.0) -> dict:
    """Measure a controller's resting zero-point and escape threshold on
    scripted straight runs, then install both on the controller.

    Calibration runs use an **empty** corridor of the same length. That is
    the whole point: with no obstacles present, every turn signal the
    controller produces is by construction noise or bias, so the statistics
    have unambiguous meanings --

    * `turn_offset`, the median turn command, is the resting left/right
      imbalance and nothing else;
    * `saccade_threshold`, a high percentile of the centred turn magnitude,
      is the noise floor, so a saccade fires only on evidence an empty
      corridor does not produce;
    An earlier version calibrated on the obstacle arenas themselves and so
    measured genuine obstacle responses as though they were noise: it put the
    saccade threshold at 0.30-0.43 when the true empty-corridor noise floor
    is about 0.02, and the agent consequently almost never turned.

    The **escape threshold is the exception** and is calibrated on the
    obstacle arenas instead. Escape means "closer than almost any moment of
    an ordinary traversal", not "something is visible at all": referenced to
    an empty corridor it fired on 87% of ticks in a corridor with six
    obstacles, which left the agent permanently braking and is not a
    detection. This split is itself a finding -- the absolute drive level
    carries far less information than the left/right difference (1.18x vs
    0.73, `validate_arena`), so the escape channel needs the looser
    reference while steering can use the tight one.
    """
    def _collect(cfg_dicts):
        turns_, stats_ = [], []
        for cfg_d in cfg_dicts:
            r = run_trial(controller, ArenaConfig(**cfg_d), encoder, resolution, tick_hz,
                          max_ticks, keep_telemetry=True, force_straight=True)
            for row in r.telemetry:
                if "turn_raw" in row:
                    turns_.append(row["turn_raw"])
                    stats_.append(row["estop_stat"])
        return turns_, stats_

    empty = {**cfg_template.__dict__, "n_obstacles": 0,
             "corridor_length": cfg_template.goal_x}
    turns, _ = _collect([{**empty, "seed": s} for s in seeds])
    _, stats = _collect([{**cfg_template.__dict__, "seed": s} for s in seeds])
    if not turns or not stats:
        return {"turn_offset": 0.0, "estop_threshold": float("inf"),
                "saccade_threshold": float("inf"), "n_samples": 0}
    turn_offset = float(np.median(turns))
    threshold = float(np.percentile(stats, estop_percentile))
    # Escape must mean "beyond anything seen while cruising". If the statistic
    # saturates (the normalised drive clips at 1.0), a percentile can land
    # exactly on the ceiling and would then fire on every clipped tick, so
    # require the threshold to sit strictly above the calibration median.
    threshold = max(threshold, float(np.median(stats)) + 1e-6)

    # Saccade trigger: a high percentile of the *centred* turn magnitude seen
    # while cruising straight down a corridor. Most of that time the corridor
    # ahead is clear, so the bulk of the distribution is the signal's own
    # noise floor and only the tail corresponds to an obstacle worth turning
    # for. Calibrating it per controller also keeps the comparison fair --
    # the connectome's turn signal (a ratio of LPLC2 population rates) and
    # the flow baseline's (a ratio of pooled drives) need not share a scale.
    centred = np.abs(np.array(turns) - turn_offset)
    saccade_threshold = float(np.percentile(centred, saccade_percentile))
    controller.turn_offset = turn_offset
    controller.estop_threshold = threshold
    controller.saccade_threshold = saccade_threshold
    return {"turn_offset": turn_offset, "estop_threshold": threshold,
            "saccade_threshold": saccade_threshold,
            "n_samples": len(turns), "n_estop_samples": len(stats),
            "turn_raw_mean": float(np.mean(turns)),
            "turn_centred_median": float(np.median(centred)),
            "estop_stat_median": float(np.median(stats)),
            "estop_stat_max": float(np.max(stats))}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def main(columns_csv: Path, n_arenas: int, resolution: int, tick_hz: float,
         mirror: str, magnitude_weighted: bool, npz_path: Path,
         json_out: Path | None, seed0: int, max_ticks: int, n_obstacles: int,
         controllers: list[str] | None = None, disable_escape: bool = False,
         row_band: tuple[float, float] = (0.0, 0.6)):
    cfg_template = ArenaConfig(n_obstacles=n_obstacles)
    encoder = BilateralEncoder(columns_csv, mirror=mirror,
                               magnitude_weighted=magnitude_weighted,
                               row_band=row_band)
    print(f"T4/T5 columns: left {encoder.n_columns('left')}, right {encoder.n_columns('right')}  "
          f"(mirror={mirror}, magnitude_weighted={magnitude_weighted})")

    cal_seeds = [900, 901, 902]
    print("calibrating drive range on straight traversals...")
    cal = calibrate_drive(encoder, cfg_template, seeds=cal_seeds,
                          resolution=resolution, tick_hz=tick_hz,
                          v_cruise=1.0, max_ticks=max_ticks)
    print(f"  centres: left={cal['drive_center']['left']:.4f} "
          f"right={cal['drive_center']['right']:.4f} "
          f"(hemisphere offset {cal['side_offset']:+.4f}), shared scale {cal['drive_scale']:.4f}")

    common = dict(drive_center=cal["drive_center"], drive_scale=cal["drive_scale"],
                  tick_hz=tick_hz, npz_path=npz_path)
    results = {}
    controller_cal = {}
    wanted = {k: v for k, v in CONTROLLERS.items()
              if controllers is None or k in controllers}
    for name, cls in wanted.items():
        controller = cls(**common)
        if controller.needs_vision:
            cc = calibrate_controller(controller, encoder, cfg_template, seeds=cal_seeds,
                                      resolution=resolution, tick_hz=tick_hz,
                                      max_ticks=max_ticks)
            controller_cal[name] = cc
            if disable_escape:
                # Ablate the escape channel outright and re-measure, the same
                # way `validate_ablation.py` deletes a cell type: DNp01 was
                # measured to sit at 84-87 spikes per tick against a
                # refractory ceiling of ~90 across the whole drive range, so
                # as a *graded* statistic it carries nothing. This quantifies
                # what steering alone is worth without it.
                controller.estop_threshold = float("inf")
                cc["estop_disabled"] = True
            print(f"  {name}: turn_offset={cc['turn_offset']:+.3f} "
                  f"saccade_threshold={cc['saccade_threshold']:.3f} "
                  f"(centred median {cc['turn_centred_median']:.3f})  "
                  + (f"estop DISABLED (would have been {cc['estop_threshold']:.2f})"
                     if disable_escape else
                     f"estop_threshold={cc['estop_threshold']:.2f} "
                     f"(median stat {cc['estop_stat_median']:.2f})"))
        rows = []
        t0 = time.time()
        for k in range(n_arenas):
            cfg = ArenaConfig(**{**cfg_template.__dict__, "seed": seed0 + k})
            r = run_trial(controller, cfg, encoder, resolution, tick_hz, max_ticks)
            rows.append(r)
            print(f"  {name:11s} arena {cfg.seed:3d}  "
                  f"{'COLLIDED' if r.collided else ('goal' if r.reached_goal else 'timeout'):8s}  "
                  f"x={r.progress_x:5.1f}/{cfg.goal_x:.0f}  min_clear={r.min_clearance:5.2f}  "
                  f"estop_ticks={r.n_estop_ticks}")
        elapsed = time.time() - t0
        n = len(rows)
        results[name] = {
            "n_arenas": n,
            "collision_rate": sum(r.collided for r in rows) / n,
            "goal_rate": sum(r.reached_goal for r in rows) / n,
            "mean_progress_x": float(np.mean([r.progress_x for r in rows])),
            "mean_min_clearance": float(np.mean([r.min_clearance for r in rows])),
            "mean_estop_ticks": float(np.mean([r.n_estop_ticks for r in rows])),
            "seconds": round(elapsed, 1),
            "per_arena": [r.__dict__ for r in rows],
        }
        for row in results[name]["per_arena"]:
            row.pop("telemetry", None)

    print(f"\n{'controller':12s} {'collision':>10s} {'goal':>8s} {'mean x':>8s} {'min clear':>10s}")
    for name, r in results.items():
        print(f"{name:12s} {r['collision_rate']:10.2f} {r['goal_rate']:8.2f} "
              f"{r['mean_progress_x']:8.1f} {r['mean_min_clearance']:10.2f}")

    bundle = {"config": {"n_arenas": n_arenas, "resolution": resolution, "tick_hz": tick_hz,
                          "disable_escape": disable_escape,
                          "mirror": mirror, "magnitude_weighted": magnitude_weighted,
                          "n_obstacles": n_obstacles, "goal_x": cfg_template.goal_x,
                          "row_band": list(row_band),
                          "seed0": seed0, "max_ticks": max_ticks},
              "calibration": cal, "controller_calibration": controller_cal,
              "results": results}
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(bundle, f, indent=2)
        print(f"\nwrote {json_out}")
    return bundle


def _default_columns_path() -> str:
    """See `flyguard.paths` -- searches $FLYGUARD_COLUMNS, the working
    directory, ~/flywire/ and the packaged data dir, in that order."""
    return default_columns_path()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns", type=Path, default=Path(_default_columns_path()))
    ap.add_argument("--npz", type=Path, default=None)
    ap.add_argument("--n-arenas", type=int, default=8)
    ap.add_argument("--n-obstacles", type=int, default=7)
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--tick-hz", type=float, default=10.0)
    ap.add_argument("--max-ticks", type=int, default=320)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--mirror", default="anatomical", choices=["none", "anatomical"])
    ap.add_argument("--no-magnitude", action="store_true",
                     help="use the unit-normalised rectified-cosine drive (the repo default "
                          "for synthetic flow) instead of the magnitude-weighted one")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--row-band", type=float, nargs=2, default=(0.0, 0.6),
                     metavar=("TOP", "BOTTOM"),
                     help="fraction of image rows the encoder reads (default drops the "
                          "ground); pass 0 1 for the whole frame")
    ap.add_argument("--disable-escape", action="store_true",
                     help="ablate the DNp01/level escape channel, leaving steering only")
    ap.add_argument("--controllers", nargs="*", default=None,
                     choices=sorted(CONTROLLERS),
                     help="subset to run (default: all three)")
    a = ap.parse_args()
    main(a.columns, a.n_arenas, a.resolution, a.tick_hz, a.mirror,
         not a.no_magnitude, a.npz, a.json_out, a.seed0, a.max_ticks, a.n_obstacles,
         a.controllers, a.disable_escape, tuple(a.row_band))
