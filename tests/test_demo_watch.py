"""
Tests for the GUI demonstration script (demo_watch.py).

WHY THESE TESTS EXIST
---------------------
The part of `demo_watch.py` that matters most cannot be tested here: whether a
window opens and shows cars. That needs a display and a human looking at it.
What *can* be tested is everything that happens before SUMO starts, and that
turns out to be where the script's two real risks live.

The first is the argument-to-controller mapping. The demo is the one place
where a person types a controller name freehand, so a typo, a stale model name
or `--controller actuated` against a network that has no actuated program must
fail with a sentence that says what to do instead - not with a TraCI error
thirty seconds into a simulation, or worse, by silently demonstrating a
different controller than the one that was asked for.

The second is the promise that the demo writes nothing. `eval/simulation.py`
and `corridor/env.py` both write a tripinfo XML per episode and both are
parsed into published metrics; a demo that inherited that behaviour would be
writing into the measurement harness's own output directory every time someone
showed the project to somebody. That promise is a property of the SUMO options
the script builds, so it is asserted directly rather than trusted.

HOW THE ENVIRONMENT IS CHECKED WITHOUT STARTING SUMO
----------------------------------------------------
`gui_env_kwargs` exists as a separate function from `make_gui_env` precisely so
the argument dictionary can be inspected. These tests read it and check the
three things a demo must get right - the GUI is on, nothing is written, and
every simulation constant still comes from the module that defines it rather
than from a copy that can drift.

The model-loading path is exercised for real: the committed `dqn.zip` is opened
and asked for an action. That is the step most likely to break silently if a
name or a path changes, and it is fast enough to run in the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import demo_watch  # noqa: E402
from eval import simulation  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse(*argv):
    return demo_watch.build_parser().parse_args(list(argv))


def test_the_defaults_are_a_runnable_demo():
    """Typing `python demo_watch.py` with no arguments must be enough."""
    args = parse()
    assert args.controller is None      # means "the model"
    assert args.model is None           # means "the default model"
    assert args.corridor is False
    assert args.list is False
    assert args.seed == demo_watch.DEFAULT_SEED
    assert args.delay == demo_watch.DEFAULT_DELAY_MS


def test_the_episode_length_default_is_the_projects_episode():
    """Not a number of its own: the demo shows the same hour the study ran."""
    assert parse().seconds == simulation.EPISODE_SECONDS


def test_the_demo_seed_is_one_the_agent_never_trained_on():
    """
    Otherwise the demo shows the agent replaying an hour it has memorised,
    which is the one thing a demonstration of a learned policy must not do.
    """
    from networks.generate_routes import EVALUATION_SEEDS

    assert demo_watch.DEFAULT_SEED in EVALUATION_SEEDS


def test_the_flags_parse():
    args = parse("--corridor", "--controller", "greedy", "--scenario", "peak",
                 "--seed", "103", "--seconds", "600", "--delay", "200",
                 "--print-every", "3", "--junctions", "5")
    assert args.corridor is True
    assert args.controller == "greedy"
    assert args.scenario == "peak"
    assert (args.seed, args.seconds, args.delay) == (103, 600, 200)
    assert (args.print_every, args.junctions) == (3, 5)


# ---------------------------------------------------------------------------
# Which controller a name resolves to
# ---------------------------------------------------------------------------
def test_the_default_is_the_reports_headline_model():
    assert demo_watch.resolve_controller(None, None, corridor=False) == "dqn"


def test_the_corridor_default_is_the_corridors_own_model():
    assert demo_watch.resolve_controller(None, None, corridor=True) == "idqn"


def test_model_flag_picks_a_different_policy():
    assert demo_watch.resolve_controller("model", "dqn_mixed", False) == "dqn_mixed"
    assert demo_watch.resolve_controller(None, "ppo_long", False) == "ppo_long"
    assert demo_watch.resolve_controller(None, "ippo", True) == "ippo"


@pytest.mark.parametrize("corridor", [False, True])
def test_greedy_is_accepted_for_the_controllers_real_name(corridor):
    """
    A demo is typed at a prompt, so the short name has to work - but it must
    resolve to the name the rest of the project uses, or `build_factory` would
    not recognise it.
    """
    assert demo_watch.resolve_controller("greedy", None, corridor) == "greedy-queue"


@pytest.mark.parametrize("name", demo_watch.SINGLE_BASELINES)
def test_single_baselines_pass_straight_through(name):
    assert demo_watch.resolve_controller(name, None, corridor=False) == name


@pytest.mark.parametrize("name", demo_watch.CORRIDOR_BASELINES)
def test_corridor_baselines_pass_straight_through(name):
    assert demo_watch.resolve_controller(name, None, corridor=True) == name


def test_actuated_on_the_single_intersection_explains_itself():
    """
    The single-intersection network has no actuated program - only the corridor
    networks were built with one. Asking for it is a reasonable thing to try,
    so the answer must name the flag that makes it work rather than saying the
    controller is unknown.
    """
    with pytest.raises(SystemExit) as raised:
        demo_watch.resolve_controller("actuated", None, corridor=False)
    assert "--corridor" in str(raised.value)


def test_an_unknown_name_lists_what_is_available():
    with pytest.raises(SystemExit) as raised:
        demo_watch.resolve_controller("dqn_v2", None, corridor=False)
    message = str(raised.value)
    assert "dqn_v2" in message
    assert "fixed-time" in message      # the baselines
    assert "dqn" in message             # the models actually on disk


def test_a_baseline_and_a_model_cannot_be_asked_for_at_once():
    """Silently ignoring one of the two would demo the wrong controller."""
    with pytest.raises(SystemExit):
        demo_watch.resolve_controller("fixed-time", "dqn", corridor=False)


def test_the_models_it_offers_are_the_ones_on_disk():
    assert "dqn" in demo_watch.known_models(corridor=False)
    assert "idqn" in demo_watch.known_models(corridor=True)
    # ma2c is stored as .pt, not .zip, so listing only .zip would hide it.
    assert "ma2c" in demo_watch.known_models(corridor=True)


# ---------------------------------------------------------------------------
# The environment arguments: GUI on, nothing written, no constant copied
# ---------------------------------------------------------------------------
def kwargs(**overrides):
    base = dict(scenario="ns_heavy", seed=100, fixed_ts=False,
                seconds=simulation.EPISODE_SECONDS, delay_ms=60,
                corridor=False, n_junctions=3, actuated=False)
    base.update(overrides)
    return demo_watch.gui_env_kwargs(**base)


def test_the_gui_is_actually_turned_on():
    """The entire reason this script exists."""
    assert kwargs()["use_gui"] is True


def test_nothing_is_written_anywhere():
    """
    The demo must not produce a tripinfo file. The evaluation harness writes
    one per episode into `results/tripinfo/` and parses it for the published
    metrics, so a demo that also wrote there would be dropping files into the
    measurement directory every time the project was shown to someone.
    """
    for options in (kwargs(), kwargs(corridor=True, scenario="calibrated")):
        assert "--tripinfo-output" not in options["additional_sumo_cmd"]
        assert "tripinfo" not in options["additional_sumo_cmd"]


def test_the_sumo_options_survive_being_split_on_spaces():
    """
    sumo-rl splits `additional_sumo_cmd` on whitespace before handing it to
    SUMO, so a value containing a space silently becomes two options. This is
    the same constraint that forced the relative tripinfo path in
    `eval/simulation.py`, recorded there as a design decision.
    """
    parts = demo_watch.sumo_extra_args(250).split()
    assert parts[parts.index("--delay") + 1] == "250"
    assert all(" " not in part for part in parts)


def test_the_timing_constants_are_the_projects_own():
    """
    Copied constants drift. Every timing value here must be the one defined in
    `eval/simulation.py`, so that a change there reaches the demo too - and so
    that what is demonstrated is the signal that was measured.
    """
    options = kwargs()
    assert options["delta_time"] == simulation.DELTA_TIME
    assert options["yellow_time"] == simulation.YELLOW_TIME
    assert options["min_green"] == simulation.MIN_GREEN
    assert options["reward_fn"] is simulation.REWARD_FN


def test_the_single_intersection_run_is_single_agent():
    options = kwargs()
    assert options["single_agent"] is True
    assert options["net_file"].endswith("single_intersection.net.xml")
    assert "ns_heavy_seed100" in options["route_file"]


def test_the_corridor_run_is_multi_agent():
    options = kwargs(corridor=True, scenario="calibrated")
    assert options["single_agent"] is False
    assert "corridor3" in options["net_file"]


def test_the_actuated_demo_loads_the_actuated_network():
    """
    SUMO drives an actuated signal from its own program, which lives in a
    separate network file. Loading the static one would quietly demonstrate
    fixed-time while calling it actuated.
    """
    options = kwargs(corridor=True, scenario="calibrated", actuated=True,
                     fixed_ts=True)
    assert "actuated" in options["net_file"]
    assert "actuated" not in kwargs(corridor=True,
                                    scenario="calibrated")["net_file"]


def test_a_missing_route_file_says_how_to_make_one():
    with pytest.raises((SystemExit, FileNotFoundError)) as raised:
        kwargs(seed=987654)
    assert "generate_routes" in str(raised.value)


# ---------------------------------------------------------------------------
# Model loading - the real zip, opened and asked for an action
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dqn_factory():
    return demo_watch.build_factory("dqn", "ns_heavy", corridor=False)


def test_the_committed_model_loads_and_decides(dqn_factory):
    """
    The step most likely to break silently if a path or a name changes. An
    empty intersection is a legitimate observation, so the policy must return
    one of the four green phases for it.
    """
    controller = dqn_factory(None)
    assert controller.uses_env_control is True
    assert controller.name == "dqn"

    observation = np.zeros(21, dtype=np.float32)   # 4 phase + 1 flag + 8 + 8
    action = controller.act(observation)
    assert isinstance(action, int)
    assert 0 <= action < 4


def test_a_baseline_needs_no_model_and_hands_control_to_sumo(dqn_factory):
    """
    `uses_env_control` decides how the environment is constructed, so getting
    it wrong would have sumo-rl and SUMO both trying to drive the same signal.
    """
    fixed = demo_watch.build_factory("fixed-time", "ns_heavy", corridor=False)
    assert fixed(None).uses_env_control is False
    assert dqn_factory(None).uses_env_control is True


def test_the_corridor_model_loads_as_one_controller_for_every_junction():
    controller = demo_watch.build_factory("idqn", "calibrated", corridor=True)(None)
    assert controller.uses_env_control is True
    assert controller.name == "idqn"


def test_the_corridor_actuated_baseline_hands_control_to_sumo():
    controller = demo_watch.build_factory("actuated", "calibrated",
                                          corridor=True)(None)
    assert controller.uses_env_control is False
