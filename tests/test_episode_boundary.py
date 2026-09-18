"""
The end of a corridor episode is a TIME LIMIT, not a terminal state.

WHY THIS TEST EXISTS
--------------------
A corridor episode stops after 3600 simulated seconds. The traffic does not
finish - the cars are still queueing when we stop watching. That is truncation,
and reinforcement learning treats truncation and termination differently:

    terminal  the return really ends here, so the value of the next state is 0
    truncated the return continues, we simply stopped observing, so the value
              of the next state must still be bootstrapped

Getting this wrong does not crash anything. It teaches the agent that nothing
after 3600 s matters, which biases every value estimate near the episode
boundary. With gamma=0.95 that is roughly the last 20 of 720 steps.

Both learners had it wrong. `CorridorVecEnv` omitted `TimeLimit.truncated`,
which SB3's own `DummyVecEnv` sets and which the replay buffer and PPO both
consult; MA2C's loop independently used `bootstrap = 0.0` on done. These tests
pin both, because the failure is silent in each case.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor import train_ma2c  # noqa: E402
from corridor.vec_env import episode_end_info  # noqa: E402


# ---------------------------------------------------------------------------
# The VecEnv contract with SB3
# ---------------------------------------------------------------------------
def test_the_episode_end_is_flagged_as_a_time_limit():
    """
    The one key whose absence is invisible. SB3 reads it as
    `info.get("TimeLimit.truncated", False)`, so omitting it silently selects
    the wrong behaviour rather than raising.
    """
    info = episode_end_info(np.zeros(21), -1000.0, 720, 15.0)
    assert info["TimeLimit.truncated"] is True


def test_the_terminal_observation_is_carried_for_the_auto_reset():
    """
    The VecEnv resets itself on done and returns the fresh observation, so the
    pre-reset one has to travel in `info` or the boundary transition is stored
    against the wrong next state.
    """
    terminal = np.arange(21, dtype=np.float32)
    info = episode_end_info(terminal, -1000.0, 720, 15.0)
    assert np.array_equal(info["terminal_observation"], terminal)


def test_the_episode_record_carries_the_curve_data():
    info = episode_end_info(np.zeros(21), -1234.5, 720, 16.25)
    assert info["episode"]["r"] == -1234.5
    assert info["episode"]["l"] == 720
    assert info["episode"]["queue"] == 16.25


def test_the_info_dict_has_no_other_keys():
    """
    Keeps the contract explicit: anything else SB3 needs must be added here
    deliberately rather than appearing by accident.
    """
    info = episode_end_info(np.zeros(21), 0.0, 0, 0.0)
    assert set(info) == {"terminal_observation", "TimeLimit.truncated",
                         "episode"}


def test_sb3_reads_the_flag_the_way_we_set_it():
    """
    Pins the spelling against the library rather than against our memory of it.
    `TimeLimit.truncated` is a string key; a typo would be undetectable.
    """
    from stable_baselines3.common import buffers

    source = inspect.getsource(buffers)
    assert 'info.get("TimeLimit.truncated", False)' in source
    assert "TimeLimit.truncated" in episode_end_info(np.zeros(1), 0.0, 0, 0.0)


# ---------------------------------------------------------------------------
# MA2C's own loop, which does not go through SB3 at all
# ---------------------------------------------------------------------------
def test_ma2c_does_not_branch_its_bootstrap_on_the_episode_ending():
    """
    MA2C has a hand-written loop, so SB3's flag cannot help it. The mistake it
    had was structural: `if done: bootstrap = 0 else: <compute it>`. The value
    of the final state must be computed the same way whether or not the hour
    happened to end on this step.
    """
    source = inspect.getsource(train_ma2c.train)
    assert "if done:\n" not in source.split("bootstrap")[0][-200:], (
        "the bootstrap is still selected by whether the episode ended")
    assert "bootstrap" in source


def test_ma2c_only_zeroes_the_bootstrap_for_an_absent_agent():
    """
    There is one legitimate zero: an agent missing from the final observation
    has no state to bootstrap from. That is a different case from the episode
    ending, and the two must not be conflated.
    """
    source = inspect.getsource(train_ma2c.train)
    # The zero must be guarded by the agent being absent, not by `done`.
    assert "if ts not in observations:" in source
    lines = [line.strip() for line in source.splitlines()]
    for i, line in enumerate(lines):
        if line == 'batch[ts]["bootstrap"] = 0.0':
            assert lines[i - 1] == "if ts not in observations:", (
                "the bootstrap is zeroed somewhere other than the "
                "absent-agent case")
