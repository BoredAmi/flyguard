"""Turning the circuit's two population rates into a motion command.

Everything in this file is an **engineering addition**, and that is the
single most important thing to know about it.

`flyguard.extract` established from the extracted subnetwork -- not from
assumption -- that LPLC2 and LC4 project onto DNp01/DNp02/DNp04/DNp11, the
Giant Fiber escape pathway, with *zero* direct edges onto DNa01, DNa02 or
MDN, the descending neurons that steer and reverse. The measured circuit can
say "stop". It cannot say "turn left".

So `estop` is anatomical and `omega` is not. The ingredients of the turn
signal are real -- two separately reconstructed optic lobes, each with its
own retinotopic T4/T5 population and its own DNp01, and flies really do bias
escape direction by which eye sees the loom -- but the comparison operator
below is ours. Do not describe a robot running this as "steered by the
connectome".

(This is, incidentally, the same seam every embodied-connectome project hits.
Eon Systems' embodied fly drives NeuroMechFly from DNa01/DNa02/oDN1 firing
rates through hand-written mappings, and says plainly that they "can be
somewhat arbitrarily chosen by hand". The connectome gives you a command
neuron and nothing downstream you can read off as behaviour.)
"""

from __future__ import annotations

import math

import numpy as np

from flyguard.runtime.types import SIDES, Command


def bilateral_turn(signal_left: float, signal_right: float, eps: float = 1e-6) -> float:
    """The engineering addition, isolated in one function so it is easy to
    find and easy to criticise.

    Returns a dimensionless turn command in [-1, 1]: **positive means turn
    left** (toward world +y, matching `flyguard.arena.step_agent`'s omega
    sign), which is what you want when the *right* hemisphere is louder.
    Contrast-normalised, so it depends on which side is louder rather than on
    the absolute drive level, which drifts with texture and speed.
    """
    total = signal_left + signal_right
    if total <= eps:
        return 0.0
    return float((signal_right - signal_left) / (total + eps))


def ema_trace(values, alpha: float) -> list[float]:
    """Exponential moving average, one trial's worth at a time, starting from
    zero. Pure and stateless so calibration can measure the same statistic
    `SaccadicSteering` accumulates online, without re-deriving it by hand and
    risking the two drifting apart.
    """
    ema = 0.0
    out = []
    for x in values:
        ema = alpha * float(x) + (1.0 - alpha) * ema
        out.append(ema)
    return out


def as_sides(value) -> dict:
    """Accept either one number or a {'left':..., 'right':...} mapping."""
    if hasattr(value, "keys"):
        return {s: float(value[s]) for s in SIDES}
    return {s: float(value) for s in SIDES}


def normalize_drive(drive: float, center: float, scale: float) -> float:
    """Map pooled drive onto the [0, 1] firing-rate fraction the circuit takes.

    Callers supply a **per-side centre and a shared gain**. The centre is
    per-side because the two optic lobes sit at measurably different resting
    drive levels -- FAFB reconstructs them unevenly. The gain is deliberately
    shared: a per-side span would rescale the hemispheres differently and so
    distort the very left/right difference the steering law reads.

    Centring at 0.5 also keeps LPi's `(1 - drive)` channel in range in both
    directions, which is what stops the circuit latching.
    """
    z = (drive - center) / max(scale, 1e-9)
    return float(np.clip(0.5 + 0.5 * z, 0.0, 1.0))


class SaccadicSteering:
    """Cruise straight; turn in discrete bursts; brake on escape.

    A continuously-steering version of this does not work, and the reason is
    not a gain that needs tuning. Rotating the camera generates large
    full-field optic flow that carries no distance information and swamps the
    encoder: traced live, both hemifields pinned at the top of their range for
    long stretches (turn signal 0.01, i.e. no information), then swung to
    +-0.99, and the agent locked at its heading limit and crawled into a wall
    on every arena.

    Real flies do not steer continuously either. They fly in straight segments
    punctuated by rapid body saccades, which confines rotational flow to brief
    intervals and leaves the straight segments' translational flow -- the part
    that actually carries depth -- clean. That is what this implements.

    **Saccadic suppression has to cover the settling period, not just the
    turn.** The flow estimate is computed between consecutive frames, so the
    tick *after* a saccade is still contaminated by it. Suppressing only
    during the turn left enough contamination to fire the escape, which fired
    the next saccade, and the agent oscillated between -37 and +46 degrees
    while braking.

    Four of the constants are *calibrated* from measurements rather than
    chosen (`turn_offset`, `saccade_threshold`, `turn_ema_threshold`,
    `estop_threshold` -- see `flyguard.runtime.calibration`). The rest --
    cruise speed, saccade rate and duration, refractory period, the EMA's
    own smoothing factor -- are ordinary robot-controller constants. They
    are not claims about the fly, and the connectome path still contains no
    fitted parameter.

    **`turn_ema_threshold` exists because of a real, diagnosed failure, not
    speculatively.** `diagnose_trial.py --arena 9 --controller connectome`
    showed a trial that collided with a corridor wall while the turn signal
    stayed the right sign (turning away from the wall) for the entire
    8-second approach, but only ever reached 0.01-0.10 against a calibrated
    `saccade_threshold` of 0.096 -- two ticks out of 82 crossed it, both too
    late. The escape channel could not help either: DNp01 sat pinned at
    exactly 87.0 spikes/tick the whole trial, one below its 88.0 threshold
    (see `runtime.circuit` and the DNp01-saturation finding elsewhere in
    this project) -- consistent, not a coincidence, since that channel was
    already known to carry no graded distance information. So a *sustained*
    weak bias produced no correction at all, from either channel, until
    contact.

    `bilateral_turn`'s instantaneous threshold is deliberately strict --
    it's a noise floor, calibrated so a single noisy tick can't fire a
    saccade. But seven straight seconds of the same sign is not noise; a
    symmetric noise process would not stay positive that long. The EMA
    tracks exactly that: it accumulates evidence the instantaneous check
    discards, and fires once the *smoothed* signal -- not any single tick --
    clears its own, separately calibrated noise floor. It is a second,
    independent trigger alongside the instantaneous one, not a replacement:
    an isolated strong tick (a pillar suddenly filling one hemifield) still
    fires immediately through `saccade_threshold`, and a weak-but-persistent
    bias (a wall approached at a shallow angle) now also fires, later than
    an instant response would but before it otherwise never would at all.
    """

    def __init__(
        self,
        *,
        v_cruise: float = 1.0,
        v_escape: float = 0.3,
        saccade_rate: float = 1.6,
        saccade_duration_s: float = 0.3,
        refractory_s: float = 0.3,
        estop_cooldown_s: float = 0.4,
        turn_offset: float = 0.0,
        saccade_threshold: float = float("inf"),
        estop_threshold: float = float("inf"),
        turn_ema_alpha: float = 0.2,
        turn_ema_threshold: float = float("inf"),
    ):
        self.v_cruise = float(v_cruise)
        self.v_escape = float(v_escape)
        self.saccade_rate = float(saccade_rate)
        self.saccade_duration_s = float(saccade_duration_s)
        self.refractory_s = float(refractory_s)
        self.estop_cooldown_s = float(estop_cooldown_s)
        self.turn_offset = float(turn_offset)
        self.saccade_threshold = float(saccade_threshold)
        self.estop_threshold = float(estop_threshold)
        self.turn_ema_alpha = float(turn_ema_alpha)
        self.turn_ema_threshold = float(turn_ema_threshold)
        self.reset()

    def reset(self) -> None:
        self._estop_until = -1.0
        self._saccade_until = -1.0
        self._refractory_until = -1.0
        self._saccade_omega = 0.0
        self._turn_ema = 0.0

    def cruise(self) -> Command:
        """The command issued before any percept exists (first frame, or a
        dropped camera message)."""
        return Command(v=self.v_cruise, omega=0.0)

    def step(self, turn_raw: float, estop_stat: float, t: float,
             telemetry: dict | None = None) -> Command:
        telemetry = {} if telemetry is None else dict(telemetry)
        turn = float(np.clip(turn_raw - self.turn_offset, -1.0, 1.0))
        in_saccade = t < self._saccade_until

        vision_valid = not in_saccade and t >= self._refractory_until
        fired_escape = vision_valid and estop_stat >= self.estop_threshold
        if fired_escape:
            self._estop_until = t + self.estop_cooldown_s
        estop = t < self._estop_until

        # Accumulate evidence only while vision is trustworthy -- the same
        # gate the instantaneous check uses -- so saccadic and post-saccade
        # rotational flow never enters the average, only untainted straight-
        # segment signal does.
        if vision_valid:
            self._turn_ema = self.turn_ema_alpha * turn + (1.0 - self.turn_ema_alpha) * self._turn_ema

        turn_ema_for_telemetry = self._turn_ema

        if in_saccade:
            omega = self._saccade_omega
        else:
            omega = 0.0
            # Three ways to start a saccade, because they cover different
            # failure shapes. The left/right difference handles obstacles off
            # to one side and is the strong, fast cue. It is blind to an
            # obstacle dead ahead, which by symmetry drives both hemispheres
            # equally and leaves the difference at ~0 -- measured directly: an
            # arena whose first obstacle sits near the corridor axis was
            # struck head-on at exactly the same place as by the vision-free
            # baseline. It is also blind to a *sustained but weak* bias, which
            # never spikes past the noise-floor threshold on any single tick
            # even though it never changes sign either -- see the class
            # docstring for the trial that diagnosed this. The EMA covers
            # that second case: it fires only once several seconds of
            # same-signed evidence clears its own noise floor, so a single
            # noisy tick still cannot trigger it.
            #
            # The frontal case is what the escape channel is for, and it is
            # the one this circuit's anatomy genuinely licenses: LPLC2 ->
            # DNp01 is a frontal-loom escape reflex. So an escape also commits
            # a turn, taking whatever weak side evidence exists to choose a
            # direction. Flies' escape turns are likewise directionally biased
            # rather than undirected.
            evidence = vision_valid and abs(turn) >= self.saccade_threshold
            ema_evidence = vision_valid and abs(self._turn_ema) >= self.turn_ema_threshold
            if evidence or ema_evidence or fired_escape:
                if evidence:
                    direction = turn if turn != 0.0 else 1.0
                elif ema_evidence:
                    direction = self._turn_ema
                else:
                    direction = turn if turn != 0.0 else 1.0
                self._saccade_omega = math.copysign(self.saccade_rate, direction)
                self._saccade_until = t + self.saccade_duration_s
                self._refractory_until = self._saccade_until + self.refractory_s
                omega = self._saccade_omega
                # The saccade is the corrective action for whatever evidence
                # just fired it -- start accumulating fresh rather than
                # letting stale evidence combine with the next straight
                # segment's own signal.
                self._turn_ema = 0.0

        # Escape *brakes*; it does not seize the steering. That matches what
        # DNp01 licenses, and an earlier version that overrode omega with a
        # full-rate turn drove the agent into the nearest wall.
        telemetry.update({"turn_raw": turn_raw, "turn": turn, "estop_stat": estop_stat,
                          "turn_ema": turn_ema_for_telemetry,
                          "saccade": bool(in_saccade or omega != 0.0)})
        return Command(v=self.v_escape if estop else self.v_cruise, omega=omega,
                       estop=estop, telemetry=telemetry)
