"""
Detector failure: corrupt what the controller can SEE, never the traffic itself.

WHY THE SEPARATION MATTERS
--------------------------
This wrapper sits between the environment and the controller and rewrites the
observation.  The simulation is untouched - the vehicles are still queueing,
they still cross the junction, and every metric is still measured from the real
simulation.  The controller simply cannot see some of them.

That is what makes this a *sensing* failure rather than a change in demand.  If
we deleted vehicles instead, both the controller's information and the problem
being solved would change at once, and the resulting number would answer
neither question.

WHAT THE OBSERVATION LOOKS LIKE
-------------------------------
sumo-rl's default observation per junction is 21 values:

    [0:4]    one-hot of the current green phase   <- internal signal state
    [4]      "minimum green has elapsed" flag     <- internal signal state
    [5:13]   density on each of the 8 lanes       <- DETECTOR-DERIVED
    [13:21]  queue on each of the 8 lanes         <- DETECTOR-DERIVED

Only the last sixteen are corrupted.  A signal always knows which phase it is
showing and how long it has been showing it - that is its own state, not a
measurement - so corrupting it would be modelling a different fault entirely.

A failed detector takes out BOTH the density and the queue reading for its
lane, because those come from the same physical sensor.  Failing them
independently would model a fault that does not occur.

THE TWO FAILURE MODES ARE NOT EQUIVALENT
----------------------------------------
    stuck-at-zero   The failed lane reports an empty road.  This is the
                    dangerous one: an approach that looks empty may never be
                    served, so the failure actively misleads rather than merely
                    withholding.
    freeze          The failed lane holds its last good reading.  Milder, and
                    closer to a communications dropout than to a dead loop.

They are reported separately because a controller can be robust to one and not
the other.

PAIRING - THE FAIRNESS RULE THIS INHERITS
-----------------------------------------
The single-intersection study's central fairness guarantee is that every
controller drives the identical hour of traffic.  Its corridor equivalent is
that every controller must also face the identical set of dead detectors.  The
failed set is therefore drawn from an RNG seeded by
(scenario, seed, rate, mode) and **not** by anything identifying the
controller, so the comparison stays paired per episode.
"""

from __future__ import annotations

import random
import zlib

import numpy as np

# Indices into the 21-value observation.
PHASE_SLICE = slice(0, 5)        # one-hot + min-green flag: never corrupted
DENSITY_SLICE = slice(5, 13)
QUEUE_SLICE = slice(13, 21)
N_LANES = 8

MODES = ("zero", "freeze")


class DetectorFailure:
    """
    Fails a fixed fraction of lane detectors for a whole episode.

    Which lanes fail is decided once, at episode start, and held for the hour:
    a detector that has died does not recover halfway through, and re-drawing
    every step would average the fault away into something much easier to
    tolerate than a real failure.
    """

    def __init__(self, rate: float, mode: str = "zero", scenario: str = "",
                 seed: int = 0):
        if not 0.0 <= rate <= 1.0:
            raise ValueError("failure rate must be in [0, 1], got %r" % rate)
        if mode not in MODES:
            raise ValueError("unknown failure mode %r; choose from %s"
                             % (mode, ", ".join(MODES)))
        self.rate = float(rate)
        self.mode = mode
        self.scenario = scenario
        self.seed = seed
        self.failed = {}          # {ts_id: sorted list of dead lane indices}
        self._last_good = {}      # {ts_id: last uncorrupted observation}

    # -- setup -----------------------------------------------------------
    def _rng(self) -> random.Random:
        """
        Seeded from the episode, never from the controller.

        This is the whole pairing guarantee: fixed-time, actuated, the
        heuristic and all three learned controllers must lose the SAME
        detectors on the same hour, or the comparison stops being paired.
        """
        key = "detfail_%s_%d_%.4f_%s" % (self.scenario, self.seed,
                                         self.rate, self.mode)
        return random.Random(zlib.crc32(key.encode("utf-8")))

    def on_episode_start(self, env) -> None:
        rng = self._rng()
        self.failed = {}
        self._last_good = {}
        n_failed = int(round(self.rate * N_LANES))
        for ts_id in sorted(env.ts_ids):
            # Drawn per junction so that a 25% failure rate means a quarter of
            # each junction's detectors, not a quarter of the corridor's
            # concentrated on one approach.
            self.failed[ts_id] = sorted(rng.sample(range(N_LANES), n_failed))

    # -- the corruption --------------------------------------------------
    def corrupt(self, observations: dict) -> dict:
        if self.rate <= 0.0:
            return observations

        out = {}
        for ts_id, obs in observations.items():
            dead = self.failed.get(ts_id, ())
            if not dead:
                out[ts_id] = obs
                continue

            corrupted = np.array(obs, dtype=np.float32, copy=True)
            previous = self._last_good.get(ts_id)
            for lane in dead:
                density_at = DENSITY_SLICE.start + lane
                queue_at = QUEUE_SLICE.start + lane
                if self.mode == "zero":
                    corrupted[density_at] = 0.0
                    corrupted[queue_at] = 0.0
                else:  # freeze
                    if previous is not None:
                        corrupted[density_at] = previous[density_at]
                        corrupted[queue_at] = previous[queue_at]
                    # With no previous reading there is nothing to hold, so the
                    # first step of an episode passes the true value through.
            # Remember the TRUE observation, so "freeze" holds the last real
            # reading rather than compounding its own corruption.
            self._last_good[ts_id] = np.array(obs, dtype=np.float32, copy=True)
            out[ts_id] = corrupted
        return out

    def describe(self) -> str:
        if self.rate <= 0.0:
            return "no detector failure"
        return "%.0f%% of lane detectors failed (%s)" % (100 * self.rate, self.mode)


class NoPerturbation:
    """The clean-conditions control, so the same code path runs unperturbed."""

    rate = 0.0
    mode = "none"

    def on_episode_start(self, env) -> None:
        pass

    def corrupt(self, observations: dict) -> dict:
        return observations

    def describe(self) -> str:
        return "no detector failure"


def make_perturbation(rate: float, mode: str, scenario: str, seed: int):
    """Build the right perturbation, including the null one."""
    if rate <= 0.0:
        return NoPerturbation()
    return DetectorFailure(rate=rate, mode=mode, scenario=scenario, seed=seed)
