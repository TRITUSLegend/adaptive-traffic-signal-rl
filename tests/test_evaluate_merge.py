"""
Tests for merge_with_existing (eval/evaluate.py).

WHY THESE TESTS EXIST
---------------------
`eval/evaluate.py --append` lets a newly trained controller be folded into an
existing evaluation_raw.csv without re-running the whole 150-episode sweep.
That is a genuine convenience and also the one place in the project where
freshly measured numbers meet numbers read back off disk.

Two things have to hold. A re-run of an episode must REPLACE the old row, not
sit beside it - otherwise the summary would average a stale measurement
together with the one that was meant to supersede it, and the mean would drift
without anything looking wrong. And values read back from CSV arrive as
strings, so they must be converted before the summary arithmetic sees them;
`max("9", "100")` is "9".

The current evaluation_summary.csv still shows the fingerprint of this path -
PPO's integer columns print as 0 where every other controller's print as 0.0 -
which is harmless but is exactly the seam these tests cover.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.evaluate import merge_with_existing, summarise  # noqa: E402

FIELDS = ["controller", "scenario", "seed", "mean_waiting_time_s",
          "mean_queue_length_veh", "mean_time_loss_s", "throughput_veh",
          "vehicles_not_inserted", "phase_switches"]


def row(controller, scenario, seed, wait, throughput=1700, not_inserted=0):
    return {
        "controller": controller,
        "scenario": scenario,
        "seed": seed,
        "mean_waiting_time_s": wait,
        "mean_queue_length_veh": wait / 2.0,
        "mean_time_loss_s": wait + 8.0,
        "throughput_veh": throughput,
        "vehicles_not_inserted": not_inserted,
        "phase_switches": 138,
    }


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "evaluation_raw.csv"
    existing = [
        row("fixed-time", "ns_heavy", 100, 52.0),
        row("fixed-time", "ns_heavy", 101, 54.0),
        row("dqn", "ns_heavy", 100, 21.0),
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(existing)
    return path


def find(rows, controller, scenario, seed):
    hits = [r for r in rows
            if r["controller"] == controller and r["scenario"] == scenario
            and int(r["seed"]) == seed]
    assert len(hits) == 1, (controller, scenario, seed, len(hits))
    return hits[0]


def test_a_new_episode_is_added_and_the_old_ones_are_kept(raw_csv):
    merged = merge_with_existing([row("ppo", "ns_heavy", 100, 19.8)], raw_csv)
    assert len(merged) == 4
    assert find(merged, "ppo", "ns_heavy", 100)["mean_waiting_time_s"] == 19.8
    assert find(merged, "fixed-time", "ns_heavy", 100)["mean_waiting_time_s"] == 52.0


def test_re_running_an_episode_replaces_it_rather_than_duplicating_it(raw_csv):
    """
    THE INVARIANT THAT MATTERS. Re-running a controller is how a stale
    measurement gets replaced, so the new row must win and the old one must
    disappear entirely - not be averaged with its replacement.
    """
    merged = merge_with_existing([row("dqn", "ns_heavy", 100, 20.6)], raw_csv)
    assert len(merged) == 3
    assert find(merged, "dqn", "ns_heavy", 100)["mean_waiting_time_s"] == 20.6


def test_an_episode_is_identified_by_controller_scenario_and_seed(raw_csv):
    """The same seed under a different scenario is a different episode."""
    merged = merge_with_existing([row("dqn", "peak", 100, 39.4)], raw_csv)
    assert len(merged) == 4
    assert find(merged, "dqn", "ns_heavy", 100)["mean_waiting_time_s"] == 21.0
    assert find(merged, "dqn", "peak", 100)["mean_waiting_time_s"] == 39.4


def test_values_read_back_from_disk_are_numbers_not_strings(raw_csv):
    """
    Everything the summary arithmetic touches must come back typed. A string
    would not raise - it would sort and compare wrongly.
    """
    merged = merge_with_existing([row("ppo", "ns_heavy", 100, 19.8)], raw_csv)
    old = find(merged, "fixed-time", "ns_heavy", 101)
    assert isinstance(old["seed"], int)
    for field in ("mean_waiting_time_s", "mean_queue_length_veh",
                  "throughput_veh", "vehicles_not_inserted", "phase_switches"):
        assert isinstance(old[field], float), (field, type(old[field]))
    assert isinstance(old["controller"], str)


def test_merged_rows_summarise_without_type_errors(raw_csv):
    """
    The end-to-end point of the conversion: rows off disk and rows from a fresh
    run have to average together.
    """
    merged = merge_with_existing([row("fixed-time", "ns_heavy", 102, 56.0)],
                                 raw_csv)
    summary = summarise(merged)
    fixed = [e for e in summary if e["controller"] == "fixed-time"][0]
    assert fixed["n_runs"] == 3
    assert fixed["mean_waiting_time_s_mean"] == pytest.approx(54.0)
    assert fixed["mean_waiting_time_s_std"] > 0


def test_the_insertion_sentinel_survives_a_round_trip(raw_csv):
    """
    -1 means "could not be measured" and must not be rounded, clipped, or
    otherwise turned into the 0 that means "nothing was dropped".
    """
    merged = merge_with_existing(
        [row("dqn", "surge", 100, 37.6, not_inserted=-1)], raw_csv)
    assert find(merged, "dqn", "surge", 100)["vehicles_not_inserted"] == -1
    summary = summarise(merged)
    entry = [e for e in summary
             if e["controller"] == "dqn" and e["scenario"] == "surge"][0]
    assert entry["vehicles_not_inserted_max"] == -1


# ---------------------------------------------------------------------------
# Controller-name handling (tagged models and the +safe suffix)
# ---------------------------------------------------------------------------
from eval.evaluate import (  # noqa: E402
    CONTROLLER_ORDER,
    build_factory,
    is_learned,
    model_name,
)


@pytest.mark.parametrize("name,expected", [
    ("dqn", True), ("ppo", True),
    ("dqn_seed1", True), ("dqn_mixed", True), ("ppo_long", True),
    ("dqn+safe", True), ("dqn_seed1+safe", True),
    ("greedy-queue", False), ("fixed-time", False),
    ("fixed-time-tuned", False), ("greedy-queue+safe", False),
])
def test_learned_controllers_are_recognised(name, expected):
    assert is_learned(name) is expected


@pytest.mark.parametrize("name,expected", [
    ("dqn", "dqn"),
    ("dqn+safe", "dqn"),
    ("dqn_seed1+safe", "dqn_seed1"),
    ("greedy-queue+safe", "greedy-queue"),
])
def test_the_safe_suffix_is_stripped_to_find_the_model(name, expected):
    assert model_name(name) == expected


def test_the_default_sweep_is_unchanged_by_the_new_names():
    """
    Tagged and wrapped controllers must be opt-in. If any leaked into
    CONTROLLER_ORDER, the default `python eval/evaluate.py` would silently
    stop reproducing the published sweep.
    """
    assert CONTROLLER_ORDER == ["fixed-time", "fixed-time-tuned",
                                "greedy-queue", "dqn", "ppo"]
    assert not any("+safe" in c or "_" in c for c in CONTROLLER_ORDER)


def test_wrapping_a_non_learned_controller_needs_no_model():
    """A wrapped heuristic is buildable with no model file present."""
    factory = build_factory("greedy-queue+safe", "ns_heavy")
    assert factory is not None
    assert factory(None).uses_env_control is True


def test_wrapping_an_untrained_model_yields_no_factory():
    """
    Rather than raising, an absent model gives None so the sweep can skip it -
    the same behaviour the untagged names already had.
    """
    assert build_factory("dqn_doesnotexist+safe", "ns_heavy") is None


def test_a_wrapped_fixed_plan_still_lets_sumo_drive():
    """
    The fixed-time plans are run by SUMO, not by us. Wrapping must not take
    that over, or the baseline would silently become a different controller.
    """
    factory = build_factory("fixed-time+safe", "ns_heavy")
    assert factory(None).uses_env_control is False
