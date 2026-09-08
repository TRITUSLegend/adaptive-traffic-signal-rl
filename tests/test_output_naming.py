"""
Tests for the names the evaluation writes and the other scripts read back.

WHY THESE TESTS EXIST
---------------------
The per-episode traces are written by `eval/evaluate.py` and read back by two
other scripts. Each of them built the filename itself, and they disagreed the
moment the `+safe` suffix arrived: the writer produced

    trace_greedy_queue+safe_ns_heavy.csv

while `eval/followups.py` looked for

    trace_greedy_queue_safe_ns_heavy.csv

Nothing raised, because a missing trace is a legitimate state - most
controllers have no trace for most scenarios - so `trace_bounds` returns None
and the row is simply omitted. Addendum E would have been published with the
constrained controllers missing from the one table that exists to show them.

That is the failure mode these tests are aimed at: not a crash, but a silently
short table. They check the contract from both ends - that every script uses
the one function, and that a file written by the writer is found by the reader.

The second group covers the other half of the same problem. `evaluation_raw.csv`
now carries controllers the report's tables do not describe (extra training
seeds, the mixed-demand agent, the `+safe` wrappers), and a generated table
that iterated over all of them would either grow rows it should not have or
raise KeyError on a label it was never given.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.evaluate as evaluate  # noqa: E402
import eval.followups as followups  # noqa: E402
import eval.make_plots as make_plots  # noqa: E402
import eval.paths as paths  # noqa: E402
from eval.make_tables import LABELS, insertion_backlog_table  # noqa: E402


# ---------------------------------------------------------------------------
# One naming function, used by everybody
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module", [evaluate, followups, make_plots])
def test_every_script_names_traces_with_the_same_function(module):
    """
    THE INVARIANT. Two copies of this rule is what caused the bug; identity,
    not equality, is what stops a copy being reintroduced.
    """
    assert module.trace_path is paths.trace_path


@pytest.mark.parametrize("controller,scenario,expected", [
    ("dqn", "ns_heavy", "trace_dqn_ns_heavy.csv"),
    ("greedy-queue", "peak", "trace_greedy_queue_peak.csv"),
    ("fixed-time-tuned", "surge", "trace_fixed_time_tuned_surge.csv"),
    ("dqn+safe", "balanced", "trace_dqn+safe_balanced.csv"),
    ("greedy-queue+safe", "ns_heavy", "trace_greedy_queue+safe_ns_heavy.csv"),
    ("dqn_seed1", "light", "trace_dqn_seed1_light.csv"),
])
def test_trace_filenames_are_what_the_committed_files_are_called(
        controller, scenario, expected):
    """Pinned to the literal names, because files already on disk use them."""
    assert paths.trace_filename(controller, scenario) == expected


def test_a_wrapped_controller_is_not_confused_with_a_tagged_one():
    """
    `dqn+safe` is the DQN under the safety bounds; `dqn_safe` would be a model
    trained with `--tag safe`. Flattening `+` to `_` would give them the same
    trace file, and the second run would overwrite the first.
    """
    assert (paths.trace_filename("dqn+safe", "peak")
            != paths.trace_filename("dqn_safe", "peak"))


def test_no_two_controllers_in_the_full_sweep_share_a_trace_file():
    controllers = evaluate.CONTROLLER_ORDER + [
        "dqn_seed1", "dqn_seed2", "dqn_mixed", "ppo_long",
        "dqn+safe", "ppo+safe", "greedy-queue+safe", "fixed-time-tuned+safe",
    ]
    names = [paths.trace_filename(c, "ns_heavy") for c in controllers]
    assert len(set(names)) == len(names)


# ---------------------------------------------------------------------------
# The writer and the reader, end to end
# ---------------------------------------------------------------------------
class FakeResult:
    """Just enough of an EpisodeResult for write_trace."""

    def __init__(self, rows):
        self.trace = rows


def synthetic_trace(hold_seconds: int = 40):
    """
    A trace that holds one phase for `hold_seconds`, then serves another.

    The exact numbers do not matter - what matters is that trace_bounds can
    read the file at all, which is what the naming bug broke.
    """
    rows = []
    time = 0.0
    for state in ("GGrrrrGGrrrr", "rrrGGrrrrGGr"):
        for _ in range(hold_seconds // 5):
            time += 5.0
            rows.append({
                "sim_time": time,
                "queue_veh": 4.0,
                "system_mean_waiting_time_s": 9.0,
                "system_mean_speed_mps": 7.5,
                "reward": -0.5,
                "signal_state": state,
            })
    return rows


@pytest.mark.parametrize("controller", ["dqn", "dqn+safe", "greedy-queue+safe"])
def test_a_trace_written_by_the_evaluation_is_found_by_the_addendum(
        controller, tmp_path, monkeypatch):
    """
    The end-to-end version of the bug: write it with evaluate, read it with
    followups, and require the second to find the first.
    """
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)

    evaluate.write_trace(FakeResult(synthetic_trace()),
                         paths.trace_path(controller, "ns_heavy"))
    bounds = followups.trace_bounds(controller, "ns_heavy")

    assert bounds is not None, "the addendum could not find the trace"
    longest_green, worst_gap = bounds
    assert longest_green == pytest.approx(40.0)
    assert worst_gap > 0


def test_a_missing_trace_is_still_reported_as_missing(tmp_path, monkeypatch):
    """The None path has to survive: most pairs genuinely have no trace."""
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    assert followups.trace_bounds("dqn+safe", "never_simulated") is None


# ---------------------------------------------------------------------------
# Generated tables ignore controllers the report does not describe
# ---------------------------------------------------------------------------
def backlog_row(controller, scenario, seed, not_inserted):
    return {"controller": controller, "scenario": scenario, "seed": seed,
            "vehicles_not_inserted": not_inserted}


def test_the_backlog_table_ignores_addendum_only_controllers():
    """
    The variants live in the same CSV as the five the report compares. A
    section 5 table that listed them would invite a reader to match its rows
    against a comparison table that has no such rows.
    """
    raw = [backlog_row("fixed-time", "surge", s, 220) for s in range(100, 105)]
    raw += [backlog_row("dqn_seed2", "surge", s, 180) for s in range(100, 105)]
    raw += [backlog_row("dqn+safe", "surge", s, 160) for s in range(100, 105)]

    table = insertion_backlog_table(raw)

    assert "220" in table
    assert "dqn_seed2" not in table and "180" not in table
    assert "safe" not in table and "160" not in table


def test_an_unlabelled_controller_does_not_crash_the_table():
    """
    Before the guard this raised KeyError on LABELS - at the very last step of
    the pipeline, after the training and evaluation runs had already gone by.
    """
    raw = [backlog_row("something-nobody-labelled", "surge", s, 99)
           for s in range(100, 105)]
    assert insertion_backlog_table(raw)  # renders a header, raises nothing


def test_every_controller_the_report_compares_has_a_label():
    """The other direction: the guard must not silently hide a real row."""
    for controller in evaluate.CONTROLLER_ORDER:
        assert controller in LABELS


def test_the_backlog_table_still_reports_the_five_it_should():
    """A guard that dropped everything would also pass the test above."""
    raw = [backlog_row(c, "surge", s, 200)
           for c in evaluate.CONTROLLER_ORDER for s in range(100, 105)]
    table = insertion_backlog_table(raw)
    for controller in evaluate.CONTROLLER_ORDER:
        assert LABELS[controller].replace("**", "") in table
