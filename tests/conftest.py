"""
Shared pytest configuration.

Records what the collection actually found, so `test_documented_counts.py` can
check the numbers README.md and VERIFICATION.md quote against the suite itself.

WHY THIS EXISTS
---------------
Those counts drifted twice. VERIFICATION.md claimed 171 tests while the suite
had grown to 211, and its per-module table was stale in four rows. That is the
same failure the report's generated tables were built to prevent - a document
confidently describing a state of the world that has moved - so it is closed
the same way, by making the claim checkable rather than by being careful.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

COLLECTED = {}


def pytest_collection_modifyitems(session, config, items):
    COLLECTED["tests"] = len(items)
    COLLECTED["per_module"] = Counter(Path(str(item.path)).name for item in items)
