"""
The corridor's result row must agree with what actually measures it.

WHY THIS TEST EXISTS
--------------------
`CorridorEpisodeResult` declared a field called `mean_travel_time_s`. That is
not a key `eval.simulation.parse_tripinfo` returns, so nothing ever assigned
it, and the dataclass default of 0.0 was written into every row of the
evaluation CSV and into the report's tables.

Nothing failed. The column existed, the numbers looked like numbers, and a
reader would have taken "mean travel time 0.0 s" as a measurement. That is
exactly defect D6 in the single-intersection study - the honesty metric that
failed silently to zero, where a broken reading was indistinguishable from a
clean result.

The fix is not to be more careful with field names. It is to make the agreement
between the producer and the consumer checkable, which is what this module
does.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor.env import CorridorEpisodeResult  # noqa: E402
from corridor.evaluate_corridor import METRICS  # noqa: E402
from eval.simulation import parse_tripinfo  # noqa: E402

# Written by hand from parse_tripinfo's return statement. Kept here rather than
# derived, so that a change to that function is noticed here rather than
# silently accommodated.
TRIPINFO_KEYS = {
    "vehicles_entered",
    "throughput_veh",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "mean_trip_duration_s",
    "mean_speed_mps",
}

TRIPINFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
  <tripinfo id="a" depart="0.00" arrival="60.00" duration="60.00"
            waitingTime="10.00" timeLoss="12.00" routeLength="600.00"/>
  <tripinfo id="b" depart="0.00" arrival="-1" duration="80.00"
            waitingTime="30.00" timeLoss="40.00" routeLength="400.00"/>
</tripinfos>
"""


@pytest.fixture
def report(tmp_path):
    path = tmp_path / "tripinfo.xml"
    path.write_text(TRIPINFO_XML, encoding="utf-8")
    return path


def test_parse_tripinfo_returns_exactly_the_keys_we_think_it_does(report):
    assert set(parse_tripinfo(report)) == TRIPINFO_KEYS


def test_every_parsed_metric_has_a_home_on_the_result():
    """
    The failure that hid: a key the parser produces but the result does not
    declare is dropped on the floor, and the metric is simply absent.
    """
    declared = {f.name for f in fields(CorridorEpisodeResult)}
    missing = TRIPINFO_KEYS - declared
    assert not missing, (
        "parse_tripinfo returns %s, which CorridorEpisodeResult does not "
        "declare - those measurements would be discarded" % sorted(missing))


def test_no_declared_tripinfo_field_is_left_unassigned():
    """
    The mirror image, and the one that actually happened: a field the result
    declares but the parser never produces stays at its default and is
    published as a real measurement of zero.
    """
    declared = {f.name for f in fields(CorridorEpisodeResult)}
    # Fields the episode loop sets itself rather than reading from tripinfo.
    set_by_the_loop = {
        "controller", "scenario", "seed", "n_junctions", "network",
        "failure_rate", "failure_mode", "total_reward",
        "mean_queue_length_veh", "max_queue_length_veh", "phase_switches",
        "vehicles_not_inserted", "trace",
    }
    unexplained = declared - TRIPINFO_KEYS - set_by_the_loop
    assert not unexplained, (
        "%s is declared on CorridorEpisodeResult but is neither set by the "
        "episode loop nor produced by parse_tripinfo, so it would be "
        "published as a zero" % sorted(unexplained))


def test_every_summary_metric_is_a_real_field():
    """`METRICS` drives the summary CSV; a name that is not a field aggregates
    nothing."""
    declared = {f.name for f in fields(CorridorEpisodeResult)}
    for metric in METRICS:
        assert metric in declared, "%s is summarised but is not a field" % metric


def test_the_summary_reports_both_delay_and_throughput():
    """The project's rule: the two are never separated."""
    assert "mean_waiting_time_s" in METRICS
    assert "throughput_veh" in METRICS


def test_a_parsed_report_populates_the_result_without_leaving_zeros(report):
    """End to end on a two-vehicle report, with hand-computed values."""
    result = CorridorEpisodeResult(controller="c", scenario="s", seed=0)
    for key, value in parse_tripinfo(report).items():
        setattr(result, key, value)

    assert result.vehicles_entered == 2
    assert result.throughput_veh == 1            # 'b' never arrived
    assert result.mean_waiting_time_s == pytest.approx(20.0)
    assert result.mean_time_loss_s == pytest.approx(26.0)
    assert result.mean_trip_duration_s == pytest.approx(70.0)
    assert result.mean_speed_mps == pytest.approx((600 / 60 + 400 / 80) / 2)


def test_the_summary_row_carries_every_metric_it_promises(report):
    result = CorridorEpisodeResult(controller="c", scenario="s", seed=0)
    for key, value in parse_tripinfo(report).items():
        setattr(result, key, value)
    summary = result.summary()
    for metric in METRICS:
        assert metric in summary, "%s missing from the summary row" % metric


def test_unfinished_vehicles_are_still_counted(report):
    """
    Survivorship bias: the worst-delayed vehicles are the ones still queued
    when the hour ends. Dropping them makes every controller look better and
    the congested ones most of all.
    """
    parsed = parse_tripinfo(report)
    assert parsed["vehicles_entered"] == 2
    assert parsed["throughput_veh"] == 1
    assert parsed["mean_waiting_time_s"] > 10.0    # 'b' is in the average
