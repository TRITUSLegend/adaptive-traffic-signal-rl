"""
The corridor environment and the single episode runner every controller uses.

This is the corridor's equivalent of `eval/simulation.py:run_episode`, and it
exists for the same reason: fixed-time, actuated, the heuristic and all three
learned controllers must be measured **by the same code**, not merely with the
same metrics.  A separate runner per controller is how a comparison quietly
stops being a comparison.

WHAT IS IMPORTED RATHER THAN REDEFINED
--------------------------------------
The reward function, the decision interval, the amber time and the minimum
green are imported from `eval.simulation` instead of being copied.  Copying
them would let the two studies drift apart silently, and the project's rule is
that every controller pays an identical safety cost.  Importing makes that
structural: there is exactly one definition of `YELLOW_TIME` in the repository.

`negative_queue_reward` divides by 8 lanes, which is 4 approaches x 2 lanes -
the same count at a corridor junction as at the single intersection, so the
reward has an identical scale and meaning here.

MULTI-AGENT SEMANTICS, AND THE TRAP IN THEM
-------------------------------------------
With `single_agent=False`, sumo-rl returns dicts keyed by traffic-signal id and
`step()` returns a **4-tuple** `(obs, rewards, dones, info)`, not the 5-tuple
gym API.  More importantly it returns entries only for the signals whose
`time_to_act` is true right now.

Signals therefore act **asynchronously**: a junction that has just switched is
serving its amber and is not asking for a decision, while its neighbours are.
Any code that assumes all three agents appear on every step will silently train
on misaligned transitions - pairing an observation from one junction with a
reward earned by another.  Everything here is written to the dict that actually
arrives, never to an assumed agent list.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sumo_setup  # noqa: E402,F401  (must run before any SUMO import)
import traci  # noqa: E402

from corridor import paths  # noqa: E402
from eval.simulation import (  # noqa: E402
    DELTA_TIME,
    EPISODE_SECONDS,
    MIN_GREEN,
    YELLOW_TIME,
    negative_queue_reward,
    parse_tripinfo,
    quiet_stdout,
)

DEFAULT_JUNCTIONS = 3


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class CorridorEpisodeResult:
    """Metrics for one simulated hour on the corridor."""

    controller: str
    scenario: str
    seed: int
    n_junctions: int = DEFAULT_JUNCTIONS
    # Perturbation context, so a robustness row is self-describing in the CSV.
    failure_rate: float = 0.0
    failure_mode: str = "none"
    network: str = "corridor3"

    total_reward: float = 0.0
    mean_queue_length_veh: float = 0.0
    max_queue_length_veh: float = 0.0
    phase_switches: int = 0
    vehicles_not_inserted: int = 0

    # Filled from the tripinfo report by parse_tripinfo().
    #
    # These names must match the keys that function actually returns. An
    # earlier version declared `mean_travel_time_s`, which parse_tripinfo does
    # not produce, so nothing ever set it and it would have been published as a
    # column of zeros - the same shape as defect D6, where an honesty metric
    # failed silently to zero and a broken measurement was indistinguishable
    # from a clean result. `test_corridor_metrics.py` now pins the agreement.
    mean_waiting_time_s: float = 0.0
    mean_time_loss_s: float = 0.0
    mean_trip_duration_s: float = 0.0
    mean_speed_mps: float = 0.0
    throughput_veh: int = 0
    vehicles_entered: int = 0

    trace: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "controller": self.controller,
            "scenario": self.scenario,
            "seed": self.seed,
            "n_junctions": self.n_junctions,
            "network": self.network,
            "failure_rate": self.failure_rate,
            "failure_mode": self.failure_mode,
            "mean_waiting_time_s": round(self.mean_waiting_time_s, 3),
            "mean_time_loss_s": round(self.mean_time_loss_s, 3),
            "mean_trip_duration_s": round(self.mean_trip_duration_s, 3),
            "mean_speed_mps": round(self.mean_speed_mps, 3),
            "mean_queue_length_veh": round(self.mean_queue_length_veh, 3),
            "max_queue_length_veh": round(self.max_queue_length_veh, 3),
            "throughput_veh": self.throughput_veh,
            "vehicles_entered": self.vehicles_entered,
            "vehicles_not_inserted": self.vehicles_not_inserted,
            "phase_switches": self.phase_switches,
            "total_reward": round(self.total_reward, 2),
        }


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------
def make_corridor_env(scenario: str, seed: int, fixed_ts: bool,
                      n_junctions: int = DEFAULT_JUNCTIONS,
                      actuated: bool = False,
                      tripinfo: Path | None = None,
                      net_override: Path | None = None,
                      route_override: Path | None = None,
                      begin_time: int = 0):
    """
    Build a multi-agent sumo-rl environment for one corridor episode.

    `fixed_ts=True` hands the signals to SUMO's own programs - which is how
    BOTH baselines work: the static plan when `actuated=False`, and the
    gap-based actuated plan when `actuated=True`.  The two differ only in which
    network file is loaded, so the geometry and the demand are identical.

    `net_override` / `route_override` exist for the transfer experiment, where
    the network is `cologne3` rather than one of ours.
    """
    from sumo_rl import SumoEnvironment

    net = net_override or paths.net_file(n_junctions, actuated=actuated)
    if not net.exists():
        raise FileNotFoundError(
            "%s missing - run:  python corridor/build_corridor.py --junctions %d"
            % (net, n_junctions))

    route = route_override or paths.route_file(n_junctions, scenario, seed)
    if not route.exists():
        raise FileNotFoundError(
            "%s missing - run:  python corridor/generate_corridor_routes.py"
            % route)

    extra = "--no-step-log true --duration-log.disable true"
    if tripinfo is not None:
        tripinfo.parent.mkdir(parents=True, exist_ok=True)
        # sumo-rl splits this string on spaces before handing it to SUMO, so
        # the path must contain none. The repository lives under a directory
        # with spaces in its name, so we pass a path relative to the repo root
        # and make that the working directory (see the os.chdir below). This
        # is the same care eval/simulation.py takes, for the same reason.
        rel = tripinfo.relative_to(paths.ROOT).as_posix()
        extra += (" --tripinfo-output %s --tripinfo-output.write-unfinished true"
                  % rel)

    return SumoEnvironment(
        net_file=str(net),
        route_file=str(route),
        use_gui=False,
        # `num_seconds` is an absolute end time in sumo-rl, not a duration, so
        # a non-zero start has to be added to keep the episode one hour long.
        # cologne3's demand only exists from 06:31, so its transfer episode
        # starts at its busiest hour rather than at t=0 where the road is empty.
        begin_time=begin_time,
        num_seconds=begin_time + EPISODE_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        single_agent=False,          # dict API: one entry per junction
        reward_fn=negative_queue_reward,
        fixed_ts=fixed_ts,
        sumo_seed=seed,
        sumo_warnings=False,
        additional_sumo_cmd=extra,
        add_system_info=True,
        add_per_agent_info=True,
    )


def signal_states(env) -> dict:
    """The light string each junction is currently displaying."""
    states = {}
    for ts_id in env.ts_ids:
        try:
            states[ts_id] = env.sumo.trafficlight.getRedYellowGreenState(ts_id)
        except traci.TraCIException:
            states[ts_id] = ""
    return states


# ---------------------------------------------------------------------------
# The episode runner
# ---------------------------------------------------------------------------
def run_corridor_episode(controller_factory, controller_name: str, scenario: str,
                         seed: int, n_junctions: int = DEFAULT_JUNCTIONS,
                         actuated: bool = False, keep_trace: bool = True,
                         quiet: bool = True, perturbation=None,
                         net_override: Path | None = None,
                         route_override: Path | None = None,
                         network_label: str | None = None,
                         begin_time: int = 0) -> CorridorEpisodeResult:
    """
    Simulate one hour on the corridor under one controller.

    `controller_factory(env)` must return an object exposing `.act(obs_dict)`
    -> action dict, `.uses_env_control`, and `.on_episode_start(env)`.  It is
    called once with `None` first, purely to read `uses_env_control`, because
    the environment has to know at construction time whether sumo-rl or SUMO
    drives the signals - and only the controller knows that.

    `perturbation`, when given, corrupts the observation dict before the
    controller sees it.  The simulation is untouched: the vehicles are still
    there, the controller simply cannot see them.  That separation is what
    makes it a *sensing* failure rather than a change in demand.
    """
    label = network_label or paths.corridor_name(n_junctions)
    tripinfo = paths.tripinfo_file(controller_name, scenario, seed, n_junctions)

    fixed_ts = not controller_factory(None).uses_env_control

    previous_cwd = os.getcwd()
    os.chdir(paths.ROOT)
    try:
        with quiet_stdout():
            env = make_corridor_env(
                scenario, seed, fixed_ts=fixed_ts, n_junctions=n_junctions,
                actuated=actuated, tripinfo=tripinfo,
                net_override=net_override, route_override=route_override,
                begin_time=begin_time)
        controller = controller_factory(env)

        result = CorridorEpisodeResult(
            controller=controller_name, scenario=scenario, seed=seed,
            n_junctions=n_junctions, network=label,
            failure_rate=getattr(perturbation, "rate", 0.0),
            failure_mode=getattr(perturbation, "mode", "none"))

        queue_samples = []
        last_green = {}

        with quiet_stdout():
            observations = env.reset()
        controller.on_episode_start(env)
        if perturbation is not None:
            perturbation.on_episode_start(env)

        done = False
        while not done:
            seen = (perturbation.corrupt(observations)
                    if perturbation is not None else observations)
            actions = controller.act(seen)
            observations, rewards, dones, info = env.step(actions)
            done = dones["__all__"]

            result.total_reward += float(sum(rewards.values()))

            stopped = float(info.get("system_total_stopped", 0.0))
            queue_samples.append(stopped)

            # Count green-to-green changes per junction, ignoring the amber in
            # between, so the number means "how often right of way was handed
            # over" - the same definition the single-intersection study uses,
            # summed across the corridor.
            states = signal_states(env)
            for ts_id, state in states.items():
                if state and "y" not in state:
                    if ts_id in last_green and state != last_green[ts_id]:
                        result.phase_switches += 1
                    last_green[ts_id] = state

            if keep_trace:
                row = {
                    "sim_time": float(info.get("step", env.sim_step)),
                    "queue_veh": stopped,
                    "system_mean_waiting_time_s":
                        float(info.get("system_mean_waiting_time", 0.0)),
                    "system_mean_speed_mps":
                        float(info.get("system_mean_speed", 0.0)),
                    "reward": float(sum(rewards.values())),
                }
                # The light string each junction is showing, one column per
                # junction. Recorded because it is the only way to answer the
                # question the single-intersection study's Addendum E had to
                # answer from its own traces: whether a policy quietly starves
                # a movement. An aggregate delay figure cannot show it - a
                # starved approach carries few enough vehicles that its misery
                # is diluted by the ones being served well - so the evidence
                # has to be the signal's own behaviour, sampled every step.
                for ts_id in sorted(states):
                    row["signal_%s" % ts_id] = states[ts_id]
                result.trace.append(row)

        # The insertion backlog must not fail silently to zero: zero is the
        # GOOD reading, the one used to argue no demand was hidden outside the
        # simulation, so a failed measurement that produced 0 would corroborate
        # a claim nothing had checked. Record -1 instead, which is visible.
        try:
            result.vehicles_not_inserted = len(
                env.sumo.simulation.getPendingVehicles())
        except traci.TraCIException as exc:
            print("  warning: could not read the insertion backlog for "
                  "%s/%s seed %d (%s); recording -1"
                  % (controller_name, scenario, seed, exc))
            result.vehicles_not_inserted = -1
        env.close()  # flushes the tripinfo file to disk
    finally:
        os.chdir(previous_cwd)

    result.mean_queue_length_veh = (
        sum(queue_samples) / len(queue_samples) if queue_samples else 0.0)
    result.max_queue_length_veh = max(queue_samples) if queue_samples else 0.0

    for key, value in parse_tripinfo(tripinfo).items():
        setattr(result, key, value)

    if not quiet:
        print("  %-22s %-11s seed %-3d  wait %6.1fs  queue %6.1f  through %5d"
              % (controller_name, scenario, seed, result.mean_waiting_time_s,
                 result.mean_queue_length_veh, result.throughput_veh))
    return result
