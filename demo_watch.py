"""
Watch a controller drive the traffic signals live, in SUMO's GUI.

This is a DEMONSTRATION script. It runs no experiment, trains nothing, and
writes nothing. Its only job is to open a window in which you can see real
vehicles queueing and the lights changing because a trained policy decided
they should, with the queue length printed to the terminal as it happens so
it is visibly a live decision loop rather than a recorded animation.

WHY IT IS A SEPARATE SCRIPT
---------------------------
`eval/simulation.py` and `corridor/env.py` both construct their environment
with `use_gui=False` hard-coded, and both write a tripinfo XML per episode.
Neither is wrong - a measurement harness should be headless and should record
what it measured - but both are exactly what a demo must not do. Rather than
add a flag to files that published numbers depend on (and which
`tests/test_corridor_isolation.py` hashes on purpose), this script builds its
own environment with `use_gui=True` and no tripinfo output, and then drives it
with the project's *existing* controllers and model loaders.

So the only thing duplicated here is the argument list handed to
`SumoEnvironment`. Every value in it - episode length, decision interval,
amber, minimum green, the reward function, the network and route paths - is
imported from the modules that define them, so there is still one definition
of each in the repository. The controllers and the model loading are reused
outright: `eval.evaluate.build_factory` and
`corridor.evaluate_corridor.build_factory` are the same functions the real
evaluation sweeps call.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
No results CSV, no trace, no tripinfo, no model is written or overwritten.
Nothing printed here should ever be quoted as a measurement: the GUI adds a
frame delay, and one unseeded-by-you run of one scenario is an anecdote. The
numbers in REPORT.md and CORRIDOR_REPORT.md come from `eval/evaluate.py` and
`corridor/evaluate_corridor.py`, over five seeds, headless.

USAGE
-----
    python demo_watch.py                          # DQN on the intersection
    python demo_watch.py --controller fixed-time  # the untuned baseline
    python demo_watch.py --corridor               # Independent DQN, 3 junctions
    python demo_watch.py --list                   # what you can pass

Close the SUMO window at any time to stop early; the script exits cleanly.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import sumo_setup  # noqa: E402,F401  (configures SUMO paths before any SUMO import)
import traci  # noqa: E402  (importable only after sumo_setup has run)

from eval.simulation import (  # noqa: E402
    DELTA_TIME,
    EPISODE_SECONDS,
    MIN_GREEN,
    NET_FILE,
    REWARD_FN,
    YELLOW_TIME,
    quiet_stdout,
    route_file,
)

# ---------------------------------------------------------------------------
# What you can ask for
# ---------------------------------------------------------------------------
# Baseline controller names, per study, spelled the way the rest of the
# project spells them. They are passed straight through to the existing
# `build_factory` of the matching evaluation script, so they build the same
# objects the published comparison used.
SINGLE_BASELINES = ["fixed-time", "fixed-time-tuned", "greedy-queue"]
CORRIDOR_BASELINES = ["fixed-time", "actuated", "greedy-queue"]

# `greedy` is what a person types at a demo; `greedy-queue` is the controller's
# name everywhere in the project. Accept the short form, use the project's.
ALIASES = {"greedy": "greedy-queue"}

# The model loaded when you do not name one.
DEFAULT_MODEL_SINGLE = "dqn"        # agent/models/dqn.zip - the report's headline
DEFAULT_MODEL_CORRIDOR = "idqn"     # corridor/models/idqn.zip

DEFAULT_SCENARIO_SINGLE = "ns_heavy"    # the scenario the DQN trained on
DEFAULT_SCENARIO_CORRIDOR = "calibrated"

# A demo seed, drawn from the held-out evaluation range so you are watching
# traffic the agent was never trained on.
DEFAULT_SEED = 100

# Milliseconds SUMO pauses between simulation steps. Without it the GUI runs
# as fast as the CPU allows and an hour of traffic flashes past unreadably.
# 60 ms puts a 3600-step hour at roughly three and a half minutes.
DEFAULT_DELAY_MS = 60

# How often the terminal line is printed, in decisions. The agent decides
# every DELTA_TIME seconds, so 6 decisions is one line per 30 simulated
# seconds - frequent enough to watch move, sparse enough to read.
DEFAULT_PRINT_EVERY = 6


def known_models(corridor: bool) -> list:
    """The trained policies actually present on disk, for --list and errors."""
    if corridor:
        from corridor import paths as corridor_paths
        directory = corridor_paths.MODELS_DIR
        return sorted(p.stem for p in directory.glob("*")
                      if p.suffix in (".zip", ".pt"))
    return sorted(p.stem for p in (ROOT / "agent" / "models").glob("*.zip"))


def resolve_controller(controller: str | None, model: str | None,
                       corridor: bool) -> str:
    """
    Turn what the user typed into the name `build_factory` expects.

    Returns a single string because that is the whole interface of both
    evaluation scripts' `build_factory`: baselines and learned policies are
    named in the same namespace there, which is what lets this script hand the
    demo straight to the code the real sweep uses.
    """
    baselines = CORRIDOR_BASELINES if corridor else SINGLE_BASELINES
    default_model = DEFAULT_MODEL_CORRIDOR if corridor else DEFAULT_MODEL_SINGLE

    # "model" means "whatever --model says", so the two flags compose.
    if controller in (None, "model"):
        return model or default_model

    name = ALIASES.get(controller, controller)

    # Checked before the membership test so this says why rather than "unknown".
    if name == "actuated" and not corridor:
        raise SystemExit(
            "there is no actuated controller for the single intersection: SUMO "
            "drives an actuated signal from its own program, and only the "
            "corridor networks were built with one. Run it there instead:\n"
            "    python demo_watch.py --corridor --controller actuated")

    if name in baselines:
        if model:
            raise SystemExit(
                "--model names a trained policy, so it cannot be combined with "
                "--controller %s. Drop one of the two." % controller)
        return name

    # Anything else is taken to be a model name, so tagged variants
    # (dqn_seed1, ppo_long, ma2c, ...) work without being enumerated here.
    available = known_models(corridor)
    if name not in available:
        raise SystemExit(
            "unknown controller %r.\n  baselines: %s\n  models on disk: %s"
            % (controller, ", ".join(baselines), ", ".join(available) or "none"))
    return name


# ---------------------------------------------------------------------------
# The environment, built for looking at rather than for measuring
# ---------------------------------------------------------------------------
def sumo_extra_args(delay_ms: int) -> str:
    """
    The raw SUMO options this demo adds.

    Note what is NOT here: `--tripinfo-output`. The measurement harness writes
    one per episode and parses it for the published metrics; a demo has nothing
    to publish, so it writes no file at all. `tests/test_demo_watch.py` asserts
    that, because "the demo quietly overwrote a result file" is precisely the
    kind of thing that would be noticed late.

    sumo-rl splits this string on spaces before handing it to SUMO, so no value
    in it may contain one.
    """
    return "--no-step-log true --duration-log.disable true --delay %d" % int(delay_ms)


def gui_env_kwargs(scenario: str, seed: int, fixed_ts: bool, seconds: int,
                   delay_ms: int, corridor: bool, n_junctions: int,
                   actuated: bool) -> dict:
    """
    Assemble the `SumoEnvironment` arguments for a visible run.

    Split out from `make_gui_env` so the interesting properties - the GUI is
    on, nothing is written - can be asserted in a test without starting SUMO,
    which is not practical headlessly.
    """
    if corridor:
        from corridor import paths as corridor_paths
        net = corridor_paths.net_file(n_junctions, actuated=actuated)
        route = corridor_paths.route_file(n_junctions, scenario, seed)
        if not net.exists():
            raise SystemExit(
                "%s missing - build it with:\n"
                "    python corridor/build_corridor.py --junctions %d"
                % (net, n_junctions))
        if not route.exists():
            raise SystemExit(
                "%s missing - generate it with:\n"
                "    python corridor/generate_corridor_routes.py" % route)
    else:
        net = NET_FILE
        if not net.exists():
            raise SystemExit(
                "%s missing - build it with:\n"
                "    python networks/build_network.py" % net)
        route = route_file(scenario, seed)  # raises with its own instructions

    return {
        "net_file": str(net),
        "route_file": str(route),
        # The one line this whole script exists for.
        "use_gui": True,
        "num_seconds": seconds,
        "delta_time": DELTA_TIME,
        "yellow_time": YELLOW_TIME,
        "min_green": MIN_GREEN,
        # The corridor is a multi-agent environment: one observation and one
        # action per junction, keyed by traffic-signal id.
        "single_agent": not corridor,
        "reward_fn": REWARD_FN,
        "fixed_ts": fixed_ts,
        "sumo_seed": seed,
        "sumo_warnings": False,
        "additional_sumo_cmd": sumo_extra_args(delay_ms),
        "add_system_info": True,
        "add_per_agent_info": True,
    }


def make_gui_env(**kwargs):
    from sumo_rl import SumoEnvironment
    return SumoEnvironment(**gui_env_kwargs(**kwargs))


# ---------------------------------------------------------------------------
# The watch loop
# ---------------------------------------------------------------------------
def build_factory(name: str, scenario: str, corridor: bool):
    """
    Hand the name to the evaluation script that already knows how to build it.

    This is the reuse that makes the demo trustworthy: the lights you watch are
    driven by the same controller objects, loaded from the same model files, as
    the numbers in the report.
    """
    if corridor:
        from corridor.evaluate_corridor import build_factory as corridor_factory
        return corridor_factory(name)

    from eval.evaluate import build_factory as single_factory
    factory = single_factory(name, scenario)
    if factory is None:
        raise SystemExit(
            "%s has not been trained yet - agent/models/%s.zip is missing"
            % (name, name))
    return factory


def watch(name: str, scenario: str, seed: int, corridor: bool, n_junctions: int,
          seconds: int, delay_ms: int, print_every: int) -> int:
    """
    Run one episode with the window open. Returns the number of decisions made.

    Closing the SUMO window kills the process TraCI is talking to, which
    surfaces as a TraCI error rather than a clean end-of-episode. That is a
    normal way to end a demo, so it is caught and reported as one.
    """
    factory = build_factory(name, scenario, corridor)

    # The environment has to know at construction time whether sumo-rl or SUMO
    # drives the signals, and only the controller knows that - so build one
    # against a placeholder purely to read the flag, then discard it. Same
    # move `run_episode` makes, for the same reason.
    fixed_ts = not factory(None).uses_env_control
    actuated = name == "actuated"

    label = "corridor (%d junctions)" % n_junctions if corridor else "intersection"
    print("Opening SUMO's window - %s, %s, scenario %r, seed %d."
          % (name, label, scenario, seed))
    print("SUMO drives this signal itself; the terminal shows what that costs."
          if fixed_ts else
          "Every %d simulated seconds the controller picks the next green phase."
          % DELTA_TIME)
    print("Close the window to stop early. Nothing is written to disk.\n")

    with quiet_stdout():
        env = make_gui_env(scenario=scenario, seed=seed, fixed_ts=fixed_ts,
                           seconds=seconds, delay_ms=delay_ms, corridor=corridor,
                           n_junctions=n_junctions, actuated=actuated)
    controller = factory(env)

    decisions = 0
    queues = []
    try:
        with quiet_stdout():
            reset = env.reset()
        # sumo-rl returns `(obs, info)` for a single agent and a bare dict for
        # the multi-agent corridor.
        observation = reset[0] if not corridor else reset
        controller.on_episode_start(env)

        done = False
        while not done:
            action = controller.act(observation)
            if corridor:
                observation, _reward, dones, info = env.step(action)
                done = dones["__all__"]
            else:
                observation, _reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            decisions += 1
            queue = float(info.get("system_total_stopped", 0.0))
            queues.append(queue)

            if decisions % max(1, print_every) == 0:
                print("  t=%5.0fs   queued now %5.1f veh   mean so far %5.1f   "
                      "lights %s"
                      % (float(info.get("step", env.sim_step)), queue,
                         sum(queues) / len(queues), _lights(env, corridor)))
    except (traci.FatalTraCIError, traci.TraCIException):
        print("\nSUMO closed - stopping here.")
    except KeyboardInterrupt:
        print("\nInterrupted - stopping here.")
    finally:
        # Closing an already-dead connection raises, which is not a failure:
        # it is what the user closing the window looks like from in here.
        with quiet_stdout(), contextlib.suppress(Exception):
            env.close()

    if queues:
        print("\n%s: %d decisions, mean queue %.1f veh, worst %.1f veh."
              % (name, decisions, sum(queues) / len(queues), max(queues)))
    print("Nothing was written. For the measured comparison see REPORT.md.")
    return decisions


def _lights(env, corridor: bool) -> str:
    """
    The light string(s) on display, which is what the window is showing.

    Read the same way `eval/simulation.py:_signal_state` reads it - from
    `getRedYellowGreenState` rather than from a phase index - because the
    index only moves under one of the two control regimes and the light string
    means the same thing under both.
    """
    with contextlib.suppress(Exception):
        states = [env.sumo.trafficlight.getRedYellowGreenState(ts)
                  for ts in env.ts_ids]
        return " ".join(states) if corridor else states[0]
    return "?"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watch a trained controller run the signals in SUMO's GUI. "
                    "Demonstration only - writes nothing, measures nothing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --list to see the controllers and models available.")
    parser.add_argument(
        "--controller", default=None,
        help="'model' (the default) to run a trained policy, or a baseline: "
             "%s for the intersection, %s with --corridor. 'greedy' is "
             "accepted for 'greedy-queue'."
             % (", ".join(SINGLE_BASELINES), ", ".join(CORRIDOR_BASELINES)))
    parser.add_argument(
        "--model", default=None,
        help="which trained policy to load (default: %s, or %s with --corridor)"
             % (DEFAULT_MODEL_SINGLE, DEFAULT_MODEL_CORRIDOR))
    parser.add_argument(
        "--corridor", action="store_true",
        help="watch the 3-junction arterial instead of the single intersection")
    parser.add_argument(
        "--junctions", type=int, default=3,
        help="corridor size, if a network of that size was built (default: 3)")
    parser.add_argument(
        "--scenario", default=None,
        help="demand pattern (default: %s, or %s with --corridor)"
             % (DEFAULT_SCENARIO_SINGLE, DEFAULT_SCENARIO_CORRIDOR))
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="which hour of traffic to run (default: %d, a held-out one)"
             % DEFAULT_SEED)
    parser.add_argument(
        "--seconds", type=int, default=EPISODE_SECONDS,
        help="simulated seconds to run (default: %d, one hour)" % EPISODE_SECONDS)
    parser.add_argument(
        "--delay", type=int, default=DEFAULT_DELAY_MS,
        help="milliseconds between simulation steps; raise it to slow the "
             "window down, 0 for as fast as possible (default: %d)"
             % DEFAULT_DELAY_MS)
    parser.add_argument(
        "--print-every", type=int, default=DEFAULT_PRINT_EVERY,
        help="print a queue line every N decisions (default: %d)"
             % DEFAULT_PRINT_EVERY)
    parser.add_argument(
        "--list", action="store_true",
        help="print the controllers and trained models available, then exit")
    return parser


def print_available() -> None:
    print("Single intersection   (python demo_watch.py ...)")
    print("  baselines : %s" % ", ".join(SINGLE_BASELINES))
    print("  models    : %s" % (", ".join(known_models(False)) or "none trained"))
    print("Corridor              (python demo_watch.py --corridor ...)")
    print("  baselines : %s" % ", ".join(CORRIDOR_BASELINES))
    print("  models    : %s" % (", ".join(known_models(True)) or "none trained"))


def main(argv: list | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.list:
        print_available()
        return

    name = resolve_controller(args.controller, args.model, args.corridor)
    scenario = args.scenario or (DEFAULT_SCENARIO_CORRIDOR if args.corridor
                                 else DEFAULT_SCENARIO_SINGLE)

    watch(name, scenario, args.seed, args.corridor, args.junctions,
          args.seconds, args.delay, args.print_every)


if __name__ == "__main__":
    main()
