"""
The corridor extension must not modify the finished single-intersection work.

WHY THIS TEST EXISTS
--------------------
The rule is stated in three places - CORRIDOR_PLAN.md section 7,
PROJECT_CONTEXT.md section 5, and the instructions this work was done under -
and a rule stated three times is still only a rule. This makes it checkable:
the single-intersection sources and REPORT.md are hashed, so an edit to any of
them fails the suite naming the file, rather than being discovered after it has
moved a published number.

It is the same move `tests/test_documented_counts.py` makes for documentation
and `tests/test_report_matches_data.py` makes for the report's numbers: close
the loop by running something, not by remembering to be careful.

IF ONE OF THESE FILES IS EVER CHANGED ON PURPOSE
------------------------------------------------
This test SHOULD fail - that is its job. It does not mean the change is wrong;
it means the change needs a deliberate decision rather than a silent one.
Recompute the hash and update the table below **in the same commit as the
change**, so review sees both together.

REPORT.md is included because the instruction is that it must not be touched
"not even a pointer line". It is hashed as a whole, and `make_tables.py --write`
is required to leave it byte-identical anyway, so this pins the same guarantee
from the other side.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# sha256 of each protected file, recorded 2026-09-19 at commit 03d5a40.
PROTECTED = {
    "eval/simulation.py":
        "a8a6f9a239209f13cb84feaae28012d3ba32a94fd2d1bc95183618a31b8effc6",
    "eval/paths.py":
        "9ee137247977bbe19ceffbc6fba7a7cbccbb3f43d5cef383177821fb7d099776",
    "eval/evaluate.py":
        "5268fb3d4a2f8a5724c63a93b7d8a50894cc9eeca41d686fb01e55c1f92617e3",
    "eval/followups.py":
        "280587036f6dd9a71b8693a513eb80324802ff952b4eccf33ee2b866860417a4",
    "eval/significance.py":
        "8f0a9b34ecf0b33f2f280efeeb79ed9bc4b6ec8e2a6264f63705f3e9b4434a21",
    "eval/make_tables.py":
        "b2b4d3a4bbe82b336f844d6a27dfad94e8717bbf8b9cb4cd6c8e4b0e0a8af7d6",
    "eval/make_plots.py":
        "2b5d3fdad5c5fd5bdcae9ad29d51d97732b29b5772f888389717d099610a92f6",
    "networks/build_network.py":
        "dbd00a08dcc248b98747b952b5c804b610bbd5125fdfe8a78b563dab546b6661",
    "networks/generate_routes.py":
        "dd01585777f13908c66814b16c3257e4e97b12574b239398c8f4d9d93b4a8b5b",
    "agent/training.py":
        "757ba53efb982d140982de855d1b1b1ce7d712e60edc16c40ca3342e2d1e1843",
    "agent/train_dqn.py":
        "d24aed5080a27fb546648fe9f7c5e0b11e9959706578e741b4a6ae377c6a3c2c",
    "agent/train_ppo.py":
        "d78297d3c7fd747be7e361abe12d71ebf1a68a06819edbf1d63843868cb6d3e0",
    "agent/train_mixed.py":
        "061c6b57cb0cf7073529fca17c2b92c4fb1212524375c781ab483084b8f68c10",
    "baseline/fixed_time.py":
        "65a74fa688eb510f53f51d981731324167bc1f4ad1e2ab044b8ac23a20303158",
    "baseline/run_baseline.py":
        "3a8fe9919e8888379b54d5d5f213d6fbc224597e4a6aaf9326d3111b249b1bd9",
    "REPORT.md":
        "969a2b2b32b73d4ddceabc00b8e73ec605b524c9ab8fe9f2d1ce72ec0ed59d5c",
}

# Trained policies that produced published numbers and must never be rewritten.
PROTECTED_MODELS = [
    "agent/models/dqn.zip",
    "agent/models/ppo.zip",
    "agent/models/dqn_seed1.zip",
    "agent/models/dqn_seed2.zip",
    "agent/models/dqn_mixed.zip",
    "agent/models/ppo_long.zip",
]


@pytest.mark.parametrize("relative_path", sorted(PROTECTED))
def test_single_intersection_file_is_unchanged(relative_path):
    path = ROOT / relative_path
    assert path.exists(), "%s has been deleted" % relative_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == PROTECTED[relative_path], (
        "%s has been modified by the corridor work.\n"
        "The corridor must not change the single-intersection study. If this "
        "change was deliberate, update the hash in tests/test_corridor_"
        "isolation.py in the SAME commit.\n  expected %s\n  actual   %s"
        % (relative_path, PROTECTED[relative_path], actual))


@pytest.mark.parametrize("relative_path", PROTECTED_MODELS)
def test_published_model_still_exists(relative_path):
    """
    These produced the report's numbers and cannot be regenerated identically.
    Existence is checked rather than content, because a zip's bytes depend on
    its timestamps - but a missing model is unrecoverable and must be loud.
    """
    assert (ROOT / relative_path).exists(), (
        "%s is missing - it produced published numbers and cannot be "
        "regenerated" % relative_path)


def test_the_committed_route_files_are_all_present():
    """
    The irreplaceable demand: these predate the seeding fix, so the current
    generator cannot reproduce them. 60 files, per PROJECT_CONTEXT.md.
    """
    routes = sorted((ROOT / "networks" / "routes").glob("*.rou.xml"))
    assert len(routes) == 60, (
        "expected 60 committed route files, found %d" % len(routes))


def test_corridor_writes_nowhere_near_the_single_intersection_outputs():
    """
    The corridor's own path table must not point at any single-intersection
    directory. Structural separation beats remembering to be careful.
    """
    from corridor import paths as cpaths

    forbidden = [
        ROOT / "networks" / "routes",
        ROOT / "agent" / "models",
        ROOT / "results" / "logs",
        ROOT / "results" / "plots",
    ]
    corridor_outputs = [
        cpaths.NETWORKS_DIR, cpaths.ROUTES_DIR, cpaths.MODELS_DIR,
        cpaths.LOGS_DIR, cpaths.PLOTS_DIR, cpaths.TRIPINFO_DIR,
    ]
    for out in corridor_outputs:
        for bad in forbidden:
            assert bad not in out.parents and out != bad, (
                "%s would write inside %s" % (out, bad))


def test_corridor_reuses_the_single_intersection_safety_constants():
    """
    Every controller in both studies must pay an identical safety cost. The
    corridor imports these rather than copying them, so there is exactly one
    definition in the repository and they cannot drift apart.
    """
    from corridor.build_corridor import MAX_DUR, MIN_DUR, YELLOW
    from corridor.controllers import SafetyConstrainedCorridorController
    from eval.simulation import (
        MIN_GREEN,
        YELLOW_TIME,
        SafetyConstrainedController,
    )

    assert YELLOW == YELLOW_TIME
    assert MIN_DUR == MIN_GREEN
    assert MAX_DUR == SafetyConstrainedController.MAX_GREEN
    assert (SafetyConstrainedCorridorController.MAX_GREEN
            == SafetyConstrainedController.MAX_GREEN)
    assert (SafetyConstrainedCorridorController.MAX_STARVATION
            == SafetyConstrainedController.MAX_STARVATION)
