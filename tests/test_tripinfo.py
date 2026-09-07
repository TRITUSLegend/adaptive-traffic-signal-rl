"""
Tests for parse_tripinfo (eval/simulation.py).

WHY THESE TESTS EXIST
---------------------
Every per-vehicle number the report quotes - mean waiting time above all -
comes out of this one function. It reads SUMO's trip report and averages it.

The subtle part is which vehicles are counted. Under heavy congestion the
worst-delayed vehicles are exactly the ones still stuck in a queue when the
hour ends, and SUMO marks those with arrival="-1". If they were dropped from
the averages, every controller would look better than it is and the congested
ones would improve most - which would flatter the fixed-time baseline in
precisely the comparison the report is making. The project passes
--tripinfo-output.write-unfinished so they appear in the file; this test
asserts the parser then actually includes them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.simulation import parse_tripinfo  # noqa: E402

HEADER = '<?xml version="1.0" encoding="UTF-8"?>\n<tripinfos>\n'
FOOTER = "</tripinfos>\n"


def trip(vid, depart, arrival, duration, waiting, loss, length):
    return (
        '  <tripinfo id="%s" depart="%.2f" arrival="%.2f" duration="%.2f" '
        'waitingTime="%.2f" timeLoss="%.2f" routeLength="%.2f"/>\n'
        % (vid, depart, arrival, duration, waiting, loss, length)
    )


@pytest.fixture
def report(tmp_path):
    """
    Three vehicles: two that finished, one still driving when the hour ended.

    The unfinished vehicle is given by far the largest waiting time, which is
    what makes its inclusion or exclusion visible in the mean.
    """
    path = tmp_path / "tripinfo.xml"
    path.write_text(
        HEADER
        + trip("veh0", 0.0, 100.0, 100.0, 10.0, 20.0, 500.0)
        + trip("veh1", 10.0, 130.0, 120.0, 20.0, 30.0, 600.0)
        + trip("veh2", 20.0, -1.0, 200.0, 120.0, 130.0, 400.0)   # unfinished
        + FOOTER,
        encoding="utf-8",
    )
    return path


def test_a_missing_file_reports_no_vehicles(tmp_path):
    """
    An episode that produced no trip report must not silently contribute zeros
    to an average - it reports that nothing entered, and nothing else.
    """
    result = parse_tripinfo(tmp_path / "absent.xml")
    assert result == {"vehicles_entered": 0}


def test_every_vehicle_that_entered_is_counted(report):
    assert parse_tripinfo(report)["vehicles_entered"] == 3


def test_throughput_counts_only_completed_trips(report):
    """
    Throughput is the vehicles that got through, so the unfinished one must not
    be in it - even though it IS in the waiting-time average. The two counts
    deliberately differ.
    """
    assert parse_tripinfo(report)["throughput_veh"] == 2


def test_unfinished_vehicles_are_included_in_the_waiting_time(report):
    """
    THE SURVIVORSHIP-BIAS GUARD. All three waiting times are averaged.

    If the parser filtered on arrival >= 0 the mean would be 15.0 instead of
    50.0 - a controller with a long queue at the hour's end would be rewarded
    for it.
    """
    result = parse_tripinfo(report)
    assert result["mean_waiting_time_s"] == pytest.approx((10 + 20 + 120) / 3)
    assert result["mean_waiting_time_s"] != pytest.approx((10 + 20) / 2)


def test_the_other_averages_cover_every_vehicle_too(report):
    result = parse_tripinfo(report)
    assert result["mean_time_loss_s"] == pytest.approx((20 + 30 + 130) / 3)
    assert result["mean_trip_duration_s"] == pytest.approx((100 + 120 + 200) / 3)


def test_speed_is_route_length_over_duration(report):
    result = parse_tripinfo(report)
    expected = (500 / 100 + 600 / 120 + 400 / 200) / 3
    assert result["mean_speed_mps"] == pytest.approx(expected)


def test_a_zero_duration_vehicle_does_not_divide_by_zero(tmp_path):
    """
    A vehicle inserted and removed within the same step has duration 0. It
    still counts as having entered, but contributes no speed sample.
    """
    path = tmp_path / "tripinfo.xml"
    path.write_text(
        HEADER
        + trip("veh0", 0.0, 100.0, 100.0, 10.0, 20.0, 500.0)
        + trip("veh1", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        + FOOTER,
        encoding="utf-8",
    )
    result = parse_tripinfo(path)
    assert result["vehicles_entered"] == 2
    assert result["mean_speed_mps"] == pytest.approx(500 / 100)


def test_an_empty_report_averages_to_zero_rather_than_crashing(tmp_path):
    path = tmp_path / "tripinfo.xml"
    path.write_text(HEADER + FOOTER, encoding="utf-8")
    result = parse_tripinfo(path)
    assert result["vehicles_entered"] == 0
    assert result["mean_waiting_time_s"] == 0.0
    assert result["throughput_veh"] == 0


def test_non_tripinfo_elements_are_ignored(tmp_path):
    """
    SUMO writes other element types into this file depending on the options in
    force. Only <tripinfo> elements are vehicles.
    """
    path = tmp_path / "tripinfo.xml"
    path.write_text(
        HEADER
        + trip("veh0", 0.0, 100.0, 100.0, 10.0, 20.0, 500.0)
        + '  <personinfo id="p0" depart="0.00"/>\n'
        + FOOTER,
        encoding="utf-8",
    )
    assert parse_tripinfo(path)["vehicles_entered"] == 1
