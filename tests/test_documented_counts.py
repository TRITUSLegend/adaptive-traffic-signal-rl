"""
The test counts quoted in the documentation must match the suite.

WHY THESE TESTS EXIST
---------------------
VERIFICATION.md is the evidence document, and "171 tests, all passing" is one
of its claims. It was still saying 171 when the suite had grown to 211, and its
per-module table was stale in four rows. Nobody would call that a serious
defect, which is exactly why it survived: it is a number in a document, and
documents are checked by reading.

So it is checked by running instead - the same move the report's generated
tables make. If a module is added, renamed or grown, this fails and names the
file and the number to change.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import conftest  # noqa: E402  (pytest has already imported tests/conftest.py)

TEST_MODULES = sorted(p.name for p in (ROOT / "tests").glob("test_*.py"))

# Every place a document states the size of the whole suite. Each pattern must
# appear exactly once and must capture the total.
#
# PROJECT_CONTEXT.md was added to this list after an audit found its per-module
# table had drifted to listing 11 of 23 modules. It had drifted precisely
# because nothing ran against it: it is the file that claims to be the source of
# truth, and it was the only one of the three checked by reading. The same audit
# found two stale suite totals and a stale episode count in it.
TOTAL_CLAIMS = [
    ("README.md", r"tests/\s+(\d+) tests, run with"),
    ("VERIFICATION.md", r"\*\*Fixed\*\* — (\d+) tests"),
    ("VERIFICATION.md", r"\*\*(\d+) tests\*\* in `tests/`"),
    ("VERIFICATION.md", r"\| Test suite \| \*\*(\d+) passed\*\*"),
    ("VERIFICATION.md", r"python -m pytest\s+# (\d+) tests"),
    ("PROJECT_CONTEXT.md", r"`python -m pytest` → \*\*(\d+) passed\*\*"),
    ("PROJECT_CONTEXT.md", r"\| `python -m pytest` \| (\d+) tests"),
    # PROJECT_CONTEXT.md writes the interpreter as `./.venv/Scripts/python.exe`
    # because that is how this project must be invoked, so the pattern has to
    # allow the suffix. A regex that matched nothing made this check pass
    # vacuously until the "exactly one" assertion caught it.
    ("PROJECT_CONTEXT.md", r"python(?:\.exe)? -m pytest\s+# (\d+) tests"),
    ("PROJECT_CONTEXT.md", r"tests/\s+(\d+) tests; documented counts"),
]

MODULE_TABLE_ROW = re.compile(r"^\| `(test_\w+\.py)` \| (\d+) \|", re.MULTILINE)

# Claims in PROJECT_CONTEXT.md that name a quantity with a ground truth on
# disk. Each entry is (regex capturing the number, callable returning reality,
# description used in the failure message).
#
# Deliberately NOT included: anything whose truth is narrative rather than
# countable ("the queue heuristic bypassed its detectors"), and anything that
# changes on every commit (a commit count) - a check that fails for being
# correct teaches people to ignore the suite.
ON_DISK_CLAIMS = [
    (r"routes/\s+(\d+) route files", lambda: _count("networks/routes/*.rou.xml"),
     "committed single-intersection route files"),
    (r"All \*\*(\d+) trained models\*\* present", lambda: _count("agent/models/*.zip"),
     "trained single-intersection models"),
    (r"logs/     per-episode CSVs, summaries, (\d+) trace CSVs",
     lambda: _count("results/logs/trace_*.csv"), "single-intersection trace CSVs"),
    (r"plots/    (\d+) figures used in REPORT.md",
     lambda: _count("results/plots/*.png"), "figures in results/plots"),
    (r"`results/logs/evaluation_raw\.csv` → \*\*(\d+) rows\*\*",
     lambda: _rows("results/logs/evaluation_raw.csv"),
     "rows in the single-intersection evaluation"),
    (r"robustness studies\. (\d+) evaluation episodes",
     lambda: sum(_rows(p) for p in (
         "results/corridor/logs/corridor_evaluation_raw.csv",
         "results/corridor/logs/corridor_robustness_raw.csv",
         "results/corridor/logs/corridor_transfer_raw.csv")),
     "corridor evaluation episodes across all three studies"),
]


def _count(pattern: str) -> int:
    head, tail = pattern.rsplit("/", 1)
    return len(list((ROOT / head).glob(tail))) if (ROOT / head).is_dir() else -1


def _rows(relative: str) -> int:
    import csv as _csv
    path = ROOT / relative
    if not path.exists():
        return -1
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in _csv.DictReader(fh))


def read(name: str) -> str:
    path = ROOT / name
    if not path.exists():
        pytest.skip("%s not present" % name)
    return path.read_text(encoding="utf-8")


def whole_suite_only():
    """These checks only mean anything when everything was collected."""
    per_module = conftest.COLLECTED.get("per_module")
    if per_module is None or sorted(per_module) != TEST_MODULES:
        pytest.skip("run the whole suite for the documented-count checks")
    return per_module


@pytest.mark.parametrize("document,pattern", TOTAL_CLAIMS)
def test_the_documented_total_matches_the_suite(document, pattern):
    whole_suite_only()
    total = conftest.COLLECTED["tests"]
    found = re.findall(pattern, read(document))

    assert len(found) == 1, (
        "expected exactly one %r in %s, found %d - if the wording changed, "
        "update TOTAL_CLAIMS so the claim stays checked"
        % (pattern, document, len(found)))
    assert int(found[0]) == total, (
        "%s says %s tests; the suite collects %d" % (document, found[0], total))


def test_the_per_module_table_matches_the_suite():
    per_module = whole_suite_only()
    documented = dict(MODULE_TABLE_ROW.findall(read("VERIFICATION.md")))

    assert documented, "VERIFICATION.md has no per-module test table"
    for module, count in documented.items():
        assert module in per_module, "%s is listed but does not exist" % module
        assert int(count) == per_module[module], (
            "VERIFICATION.md says %s has %s tests; it has %d"
            % (module, count, per_module[module]))


def test_every_test_module_appears_in_the_table():
    """A new module must be described, not silently left out of the evidence."""
    per_module = whole_suite_only()
    documented = dict(MODULE_TABLE_ROW.findall(read("VERIFICATION.md")))

    missing = sorted(set(per_module) - set(documented))
    assert not missing, "not listed in VERIFICATION.md: %s" % ", ".join(missing)


def test_the_documented_rows_add_up_to_the_documented_total():
    """
    The table and the headline number are written separately, so they can
    disagree with each other even when both look plausible.
    """
    text = read("VERIFICATION.md")
    rows = MODULE_TABLE_ROW.findall(text)
    total = re.findall(r"\*\*(\d+) tests\*\* in `tests/`", text)

    assert len(total) == 1
    assert sum(int(n) for _, n in rows) == int(total[0]), (
        "the per-module rows sum to %d, the headline says %s"
        % (sum(int(n) for _, n in rows), total[0]))


# ---------------------------------------------------------------------------
# PROJECT_CONTEXT.md's countable claims, against what is actually on disk
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("pattern,truth,description", ON_DISK_CLAIMS)
def test_project_context_counts_match_the_repository(pattern, truth, description):
    """
    The file that calls itself the source of truth was the only one of the
    three whose numbers nobody ran against anything. An audit found three stale
    figures in it, including a suite total two releases old and a corridor
    episode count that missed a whole evaluation.

    Each claim here names something countable. If one fails, either the
    repository changed and the sentence needs updating, or something was
    deleted that should not have been - both worth knowing.
    """
    text = read("PROJECT_CONTEXT.md")
    found = re.findall(pattern, text)
    assert len(found) == 1, (
        "expected exactly one %r in PROJECT_CONTEXT.md, found %d - if the "
        "wording changed, update ON_DISK_CLAIMS so the claim stays checked"
        % (pattern, len(found)))

    actual = truth()
    assert actual >= 0, (
        "cannot verify %r: the thing it counts is missing from disk"
        % description)
    assert int(found[0]) == actual, (
        "PROJECT_CONTEXT.md says %s = %s; the repository has %d"
        % (description, found[0], actual))


def test_project_context_corridor_result_table_matches_the_data():
    """
    Section 12 quotes each corridor controller's delay against fixed-time.
    Those five numbers are the summary a reader is most likely to cite without
    opening CORRIDOR_REPORT.md, so they are recomputed from the raw episodes
    rather than trusted.
    """
    import csv as _csv
    import statistics

    raw = ROOT / "results" / "corridor" / "logs" / "corridor_evaluation_raw.csv"
    if not raw.exists():
        pytest.skip("corridor evaluation has not been run")
    with raw.open(encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))

    scenarios = ["calibrated", "balanced", "peak", "reverse"]

    def mean_wait(controller, scenario):
        values = [float(r["mean_waiting_time_s"]) for r in rows
                  if r["controller"] == controller and r["scenario"] == scenario]
        return statistics.mean(values) if values else None

    baseline = {s: mean_wait("fixed-time", s) for s in scenarios}
    if any(v is None for v in baseline.values()):
        pytest.skip("fixed-time rows missing")

    labels = {
        "Actuated (SUMO gap-based)": "actuated",
        "Greedy queue heuristic": "greedy-queue",
        "Independent DQN": "idqn",
        "Independent PPO": "ippo",
    }
    text = read("PROJECT_CONTEXT.md")
    checked = 0
    for label, controller in labels.items():
        # The table is written with a typographic minus (U+2212), not a hyphen.
        # Matching only ASCII silently found nothing and the test passed
        # vacuously until the "at least four rows" assertion caught it - which
        # is why that assertion is there.
        match = re.search(r"\| \*{0,2}%s\*{0,2} \| \*{0,2}([+\-−][\d.]+)%%"
                          % re.escape(label), text)
        if match is None:
            continue
        claimed_text = match.group(1).replace("−", "-")
        changes = [100.0 * (mean_wait(controller, s) - baseline[s]) / baseline[s]
                   for s in scenarios if mean_wait(controller, s) is not None]
        actual = statistics.mean(changes)
        claimed = float(claimed_text)
        assert abs(actual - claimed) < 0.1, (
            "PROJECT_CONTEXT.md section 12 says %s is %+.1f%% against "
            "fixed-time; the data gives %+.1f%%" % (label, claimed, actual))
        checked += 1
    assert checked >= 4, "only %d of the corridor result rows were verified" % checked


def test_no_commit_count_is_quoted_anywhere():
    """
    A commit count is stale the moment the next commit lands, and a check that
    fails for being correct teaches people to ignore the suite. It was removed
    from PROJECT_CONTEXT.md rather than tested; this keeps it out.
    """
    for name in ("PROJECT_CONTEXT.md", "README.md", "VERIFICATION.md"):
        text = read(name)
        assert not re.search(r"\b\d+ commits\b", text), (
            "%s quotes a commit count, which goes stale on the next commit - "
            "describe the branch instead" % name)
