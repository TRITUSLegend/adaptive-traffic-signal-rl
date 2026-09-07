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
