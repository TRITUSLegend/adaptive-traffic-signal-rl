"""
The one place corridor output filenames are decided.

WHY THIS EXISTS
---------------
Defect F2 in the single-intersection work: the trace filename was built in
three places, two of them disagreed, and because "no trace file" is a
legitimate state the mismatch did not raise - it silently dropped rows from a
table whose whole purpose was to show those rows. `eval/paths.py` fixed that by
making one function the single naming authority.

The corridor has strictly more names to get wrong (controllers x scenarios x
perturbation levels x corridor sizes), so it starts with the same discipline
instead of acquiring it after being bitten.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORRIDOR_DIR = ROOT / "corridor"
NETWORKS_DIR = CORRIDOR_DIR / "networks"
ROUTES_DIR = CORRIDOR_DIR / "routes"
MODELS_DIR = CORRIDOR_DIR / "models"

RESULTS_DIR = ROOT / "results" / "corridor"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
TRIPINFO_DIR = RESULTS_DIR / "tripinfo"


def _safe(name: str) -> str:
    """
    Make a controller or scenario name safe for a filename.

    `+` is kept deliberately: `dqn+safe` is how the single-intersection study
    names a wrapped controller, and `eval/paths.py` keeps it too. Consistency
    with the existing convention matters more than shell-friendliness here,
    and the F2 lesson is that the danger is two *different* spellings, not an
    awkward one.
    """
    return name.replace("-", "_").replace(" ", "_").replace("/", "_")


def corridor_name(n_junctions: int) -> str:
    """Canonical name of an N-junction corridor, e.g. 3 -> 'corridor3'."""
    if n_junctions < 1:
        raise ValueError("a corridor needs at least one junction, got %r" % n_junctions)
    return "corridor%d" % n_junctions


def network_dir(n_junctions: int) -> Path:
    return NETWORKS_DIR / corridor_name(n_junctions)


def net_file(n_junctions: int, actuated: bool = False) -> Path:
    """
    The compiled network.

    Two files are built from identical geometry: one carrying the static
    fixed-time programs and one carrying SUMO's gap-based actuated programs.
    Two files rather than two programs in one file because sumo-rl splits
    `additional_sumo_cmd` on spaces, which makes program-switching via an
    additional-file path fragile on a repo whose own path contains spaces.
    """
    stem = corridor_name(n_junctions) + ("_actuated" if actuated else "")
    return network_dir(n_junctions) / (stem + ".net.xml")


def route_file(n_junctions: int, scenario: str, seed: int) -> Path:
    return ROUTES_DIR / corridor_name(n_junctions) / (
        "%s_seed%d.rou.xml" % (_safe(scenario), seed))


def model_file(algorithm: str, tag: str = "") -> Path:
    stem = _safe(algorithm) + (("_" + _safe(tag)) if tag else "")
    suffix = ".pt" if algorithm.startswith("ma2c") else ".zip"
    return MODELS_DIR / (stem + suffix)


def training_curve(algorithm: str, tag: str = "") -> Path:
    stem = _safe(algorithm) + (("_" + _safe(tag)) if tag else "")
    return LOGS_DIR / ("training_curve_%s.csv" % stem)


def trace_file(controller: str, scenario: str, n_junctions: int = 3) -> Path:
    return LOGS_DIR / ("trace_%s_%s_%s.csv" % (
        _safe(controller), _safe(scenario), corridor_name(n_junctions)))


def tripinfo_file(controller: str, scenario: str, seed: int,
                  n_junctions: int = 3) -> Path:
    return TRIPINFO_DIR / ("%s_%s_seed%d_%s.xml" % (
        _safe(controller), _safe(scenario), seed, corridor_name(n_junctions)))


EVALUATION_RAW = LOGS_DIR / "corridor_evaluation_raw.csv"
EVALUATION_SUMMARY = LOGS_DIR / "corridor_evaluation_summary.csv"
ROBUSTNESS_RAW = LOGS_DIR / "corridor_robustness_raw.csv"
TRANSFER_RAW = LOGS_DIR / "corridor_transfer_raw.csv"
