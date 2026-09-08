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

# Every place either document states the size of the whole suite. Each pattern
# must appear exactly once and must capture the total.
TOTAL_CLAIMS = [
    ("README.md", r"tests/\s+(\d+) tests, run with"),
    ("VERIFICATION.md", r"\*\*Fixed\*\* — (\d+) tests"),
    ("VERIFICATION.md", r"\*\*(\d+) tests\*\* in `tests/`"),
    ("VERIFICATION.md", r"\| Test suite \| \*\*(\d+) passed\*\*"),
    ("VERIFICATION.md", r"python -m pytest\s+# (\d+) tests"),
]

MODULE_TABLE_ROW = re.compile(r"^\| `(test_\w+\.py)` \| (\d+) \|", re.MULTILINE)


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
