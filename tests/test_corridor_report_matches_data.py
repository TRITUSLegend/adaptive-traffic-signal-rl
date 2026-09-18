"""
Every number in CORRIDOR_REPORT.md must trace back to the CSVs.

WHY THIS TEST EXISTS
--------------------
Defect D4 in the single-intersection study: REPORT.md claimed its tables came
from the evaluation data, and they did - the first time someone typed them.
Nothing kept them there, and the claim of provenance was therefore an intention
rather than a guarantee. `tests/test_report_matches_data.py` closed that loop by
parsing the report back out of the markdown and checking it against the CSVs.

The corridor report is generated the same way and is checked the same way. This
matters more here, not less: the corridor has three result tables drawn from
three different CSVs, and a row silently attributed to the wrong source would
be invisible to a reader.

WHAT IS CHECKED
---------------
Structure and provenance, not prose: that every generated region exists and is
filled, that the controllers and scenarios named are the ones actually
evaluated, that waiting-time cells match the raw data to rounding, and that the
report's prominent single-seed caveat is present. The caveat is checked because
it is the one claim whose absence would make every number in the document
misleading.
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor import paths  # noqa: E402
from corridor.evaluate_corridor import CONTROLLER_ORDER, LABELS  # noqa: E402

REPORT = ROOT / "CORRIDOR_REPORT.md"

GENERATED_REGIONS = ["headline", "reduction", "throughput", "convergence",
                     "detector", "margin", "transfer"]

# Each region is built from a different source, so each has its own
# precondition. Without this the whole set would be gated on the clean sweep
# alone, and a region whose data genuinely had not been produced yet would fail
# pointing at the wrong missing file.
REGION_SOURCE = {
    "headline": lambda: paths.EVALUATION_RAW,
    "reduction": lambda: paths.EVALUATION_RAW,
    "throughput": lambda: paths.EVALUATION_RAW,
    "convergence": lambda: paths.training_curve("idqn"),
    "detector": lambda: paths.ROBUSTNESS_RAW,
    "margin": lambda: paths.ROBUSTNESS_RAW,
    "transfer": lambda: paths.TRANSFER_RAW,
}


def report_text() -> str:
    if not REPORT.exists():
        pytest.skip("CORRIDOR_REPORT.md not written yet")
    return REPORT.read_text(encoding="utf-8")


def raw_rows() -> list:
    if not paths.EVALUATION_RAW.exists():
        pytest.skip("corridor evaluation has not been run")
    with paths.EVALUATION_RAW.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def require_results() -> None:
    """
    Skip while the evaluation has not been run; fail once it has.

    The distinction matters and is the whole point of these checks. An empty
    generated region BEFORE the sweep is simply work not yet done. An empty
    region AFTER the data exists is exactly the defect the single-intersection
    study shipped - a report promising five addenda while containing one and a
    half, because the generator was never re-run once the data landed. Only the
    second is a failure, so only the second fails.
    """
    if not paths.EVALUATION_RAW.exists():
        pytest.skip("corridor evaluation has not been run yet")


def region(text: str, name: str) -> str:
    begin = "<!-- BEGIN GENERATED: %s -->" % name
    end = "<!-- END GENERATED: %s -->" % name
    assert begin in text and end in text, "no generated region %r" % name
    return text[text.index(begin) + len(begin):text.index(end)]


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GENERATED_REGIONS)
def test_every_generated_region_exists(name):
    text = report_text()
    assert "<!-- BEGIN GENERATED: %s -->" % name in text
    assert "<!-- END GENERATED: %s -->" % name in text


@pytest.mark.parametrize("name", GENERATED_REGIONS)
def test_no_generated_region_was_left_empty(name):
    """
    An empty region once the data exists is exactly the failure the
    single-intersection study shipped: the report promised five addenda and
    contained one and a half, because the generator had not been re-run after
    the data landed.
    """
    source = REGION_SOURCE[name]()
    if not source.exists():
        pytest.skip("%s has not been produced yet, so region %r cannot be "
                    "filled" % (source.name, name))
    body = region(report_text(), name).strip()
    assert body, ("generated region %r is empty although %s exists - run:  "
                  "python corridor/make_corridor_tables.py --write"
                  % (name, source.name))
    assert "|" in body, "region %r contains no table" % name


def test_the_report_states_the_single_seed_caveat_prominently():
    """
    One training seed, no significance test. Every margin in the document is a
    point estimate. If that sentence is ever dropped, the numbers start
    implying a confidence the experiment cannot support.
    """
    text = report_text().lower()
    assert "single seed" in text or "one seed" in text
    assert "significance" in text
    assert "point estimate" in text


def test_the_report_states_that_it_does_not_modify_the_original_study():
    text = report_text()
    assert "REPORT.md" in text


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def test_waiting_time_cells_match_the_raw_data():
    """
    Parse the headline table back out of the markdown and check each cell
    against the mean of its seeds. This is the check that makes the report's
    provenance a guarantee rather than a claim.
    """
    rows = raw_rows()
    body = region(report_text(), "headline")

    pattern = re.compile(
        r"^\|\s*(.+?)\s*\|\s*`(\w+)`\s*\|\s*([\d.]+)\s*±", re.MULTILINE)
    found = pattern.findall(body)
    assert found, "no parseable rows in the headline table"

    by_label = {LABELS[c]: c for c in CONTROLLER_ORDER}
    checked = 0
    for label, scenario, value in found:
        controller = by_label.get(label.strip())
        if controller is None:
            continue
        seeds = [float(r["mean_waiting_time_s"]) for r in rows
                 if r["controller"] == controller and r["scenario"] == scenario]
        if not seeds:
            continue
        assert abs(statistics.mean(seeds) - float(value)) < 0.05, (
            "%s/%s: report says %s, data gives %.3f"
            % (controller, scenario, value, statistics.mean(seeds)))
        checked += 1
    assert checked >= 4, "only %d cells were verified" % checked


def test_the_report_names_only_controllers_that_were_evaluated():
    rows = raw_rows()
    evaluated = {r["controller"] for r in rows}
    body = region(report_text(), "headline")
    for controller in CONTROLLER_ORDER:
        if LABELS[controller] in body:
            assert controller in evaluated, (
                "%s appears in the report but was never evaluated" % controller)


def test_throughput_is_reported_alongside_waiting_time():
    """
    The project's central measurement rule. A controller can fake excellent
    delay by refusing to admit vehicles, so the headline table must carry
    throughput in the same row.
    """
    require_results()
    body = region(report_text(), "headline")
    header = body.strip().splitlines()[0].lower()
    assert "wait" in header
    assert "throughput" in header


def test_every_evaluated_cell_used_the_expected_number_of_seeds():
    rows = raw_rows()
    counts = {}
    for row in rows:
        counts[(row["controller"], row["scenario"])] = counts.get(
            (row["controller"], row["scenario"]), 0) + 1
    assert counts, "no evaluation rows"
    assert set(counts.values()) == {5}, (
        "expected 5 seeds per cell, found %s" % sorted(set(counts.values())))


def test_no_evaluated_episode_silently_lost_its_insertion_backlog():
    """
    -1 is the sentinel for a failed backlog reading. Zero is the good value and
    the one the report cites, so a broken measurement must never be
    indistinguishable from a clean result.
    """
    rows = raw_rows()
    broken = [r for r in rows if int(float(r["vehicles_not_inserted"])) < 0]
    assert not broken, (
        "%d episodes failed to read the insertion backlog" % len(broken))
