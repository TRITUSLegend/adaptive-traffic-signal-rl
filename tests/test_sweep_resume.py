"""
A long sweep must survive being interrupted.

WHY THIS TEST EXISTS
--------------------
The detector-failure sweep accumulated its rows in memory and wrote the CSV
only after the loop finished. It reached 140 of 210 episodes - eighty minutes
of simulation - before the machine ran out of memory, SUMO failed to launch,
and every one of those episodes was lost.

The fix is to write after each episode and skip what is already recorded. That
is only trustworthy if the skip key identifies an episode exactly: too loose
and a re-run silently drops episodes that were never measured, too strict and
it repeats hours of work. Both failures are quiet, so the key is pinned here.

Fixtures are written to tmp_path, never to a real output path.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor.robustness import episode_key, load_done  # noqa: E402

FIELDS = ["controller", "scenario", "seed", "network", "failure_rate",
          "failure_mode", "mean_waiting_time_s"]


def row(controller="idqn", scenario="calibrated", seed=100,
        network="corridor3", rate="0.2500", mode="zero", wait=30.0):
    return {"controller": controller, "scenario": scenario, "seed": seed,
            "network": network, "failure_rate": rate, "failure_mode": mode,
            "mean_waiting_time_s": wait}


def write(tmp_path, rows):
    path = tmp_path / "robust.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# What identifies an episode
# ---------------------------------------------------------------------------
def test_the_key_covers_every_dimension_that_varies():
    """
    Controller, scenario, seed, network, failure rate and mode. Drop any one
    and two genuinely different episodes collide, so the second is skipped
    having never been run.
    """
    base = episode_key(row())
    assert base != episode_key(row(controller="ippo"))
    assert base != episode_key(row(scenario="peak"))
    assert base != episode_key(row(seed=101))
    assert base != episode_key(row(network="corridor5"))
    assert base != episode_key(row(rate="0.5000"))
    assert base != episode_key(row(mode="freeze"))


def test_the_key_ignores_the_measurement_itself():
    """Two runs of the same episode are the same episode, whatever came out."""
    assert episode_key(row(wait=30.0)) == episode_key(row(wait=99.9))


def test_failure_rate_is_compared_numerically_not_textually():
    """
    '0.25' and '0.2500' are the same rate. Comparing the raw strings would
    re-run every episode whose formatting differed, which is how a resume
    quietly becomes a restart.
    """
    assert episode_key(row(rate="0.25")) == episode_key(row(rate="0.2500"))


def test_seed_type_does_not_change_the_key():
    """A seed read from CSV is a string; one built in code is an int."""
    assert episode_key(row(seed=100)) == episode_key(row(seed="100"))


# ---------------------------------------------------------------------------
# Loading what is done
# ---------------------------------------------------------------------------
def test_a_missing_file_means_nothing_is_done(tmp_path):
    assert load_done(tmp_path / "absent.csv") == {}


def test_every_recorded_episode_is_loaded(tmp_path):
    rows = [row(seed=s) for s in (100, 101, 102)]
    done = load_done(write(tmp_path, rows))
    assert len(done) == 3
    for r in rows:
        assert episode_key(r) in done


def test_a_recorded_episode_would_be_skipped(tmp_path):
    """The whole point: a completed episode is not recomputed."""
    done = load_done(write(tmp_path, [row(seed=100)]))
    assert episode_key(row(seed=100)) in done
    assert episode_key(row(seed=101)) not in done


def test_the_clean_control_is_keyed_apart_from_a_failure_run(tmp_path):
    """
    Rate 0 is stored with mode 'none' and belongs to every mode's column. It
    must not collide with a real failure run at the same seed.
    """
    clean = row(rate="0.0000", mode="none")
    failed = row(rate="0.0000", mode="zero")
    assert episode_key(clean) != episode_key(failed)


def test_resuming_does_not_lose_the_rows_it_skips(tmp_path):
    """
    The rows already on disk are carried into the new run's output, or a
    resume would finish having thrown away what it skipped.
    """
    rows = [row(seed=s) for s in (100, 101)]
    done = load_done(write(tmp_path, rows))
    carried = list(done.values())
    assert len(carried) == 2
    assert {r["seed"] for r in carried} == {"100", "101"}
