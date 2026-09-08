"""
Verify that the whole toolchain actually works before spending an hour on training.

Run:
    python smoke_test.py          # normal
    python smoke_test.py -v       # show full tracebacks on failure

It checks, in order:
    1. SUMO can be found and reports its version.
    2. Every Python package we depend on imports.
    3. The network and route files have been built.
    4. A short headless SUMO simulation runs, moves vehicles, and the traffic
       light changes phase.
    5. sumo-rl can wrap that simulation as a reinforcement-learning environment
       with the observation and action spaces we expect.

If this prints "ALL CHECKS PASSED" the rest of the project will run.  If it
fails it says which step broke and what to do about it, which is much easier to
debug than a crash 20 minutes into training.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PASS = "  [ok]   "
FAIL = "  [FAIL] "


# ---------------------------------------------------------------------------
# 1. SUMO installation
# ---------------------------------------------------------------------------
def check_sumo() -> str:
    import subprocess

    import sumo_setup

    lines = [sumo_setup.describe()]
    version = subprocess.run(
        [sumo_setup.binary("sumo"), "--version"], capture_output=True, text=True
    )
    lines += [ln for ln in version.stdout.splitlines() if ln.strip()][:1]
    for tool in ("netconvert", "duarouter"):
        lines.append("%-12s -> %s" % (tool, Path(sumo_setup.binary(tool)).name))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------------------
def check_packages() -> str:
    import sumo_setup  # noqa: F401  ensures SUMO's tools are importable

    out = []
    for module in ("numpy", "pandas", "matplotlib", "gymnasium", "torch",
                   "stable_baselines3", "sumo_rl", "traci", "sumolib"):
        mod = __import__(module)
        out.append("%-20s %s" % (module, getattr(mod, "__version__", "")))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3. Generated input files
# ---------------------------------------------------------------------------
def check_files() -> str:
    net = ROOT / "networks" / "single_intersection.net.xml"
    if not net.exists():
        raise FileNotFoundError(
            "%s missing - run:  python networks/build_network.py" % net.name)
    routes = sorted((ROOT / "networks" / "routes").glob("*.rou.xml"))
    if not routes:
        raise FileNotFoundError(
            "no route files - run:  python networks/generate_routes.py")
    return "network      %s (%.0f KB)\nroute files  %d, e.g. %s" % (
        net.name, net.stat().st_size / 1024, len(routes), routes[0].name)


# ---------------------------------------------------------------------------
# 4. A real (short) simulation
# ---------------------------------------------------------------------------
def check_simulation() -> str:
    import sumo_setup
    import traci

    net = ROOT / "networks" / "single_intersection.net.xml"
    route = ROOT / "networks" / "routes" / "ns_heavy_seed0.rou.xml"
    traci.start([
        sumo_setup.binary("sumo"),
        "-n", str(net),
        "-r", str(route),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--duration-log.disable", "true",
    ])
    try:
        tl_id = traci.trafficlight.getIDList()[0]
        states_seen = set()
        peak_vehicles = 0
        arrived = 0
        for _ in range(300):
            traci.simulationStep()
            states_seen.add(traci.trafficlight.getRedYellowGreenState(tl_id))
            peak_vehicles = max(peak_vehicles, traci.vehicle.getIDCount())
            arrived += traci.simulation.getArrivedNumber()
        n_links = len(traci.trafficlight.getControlledLinks(tl_id))
    finally:
        traci.close()

    if peak_vehicles == 0:
        raise RuntimeError("no vehicles were inserted - check the route file")
    if len(states_seen) < 4:
        raise RuntimeError("traffic light did not cycle through its phases")

    return ("traffic light        %s\ncontrolled movements %d\n"
            "signal states seen   %d (expect 8: 4 green + 4 amber)\n"
            "peak vehicles        %d\ncompleted trips      %d"
            % (tl_id, n_links, len(states_seen), peak_vehicles, arrived))


# ---------------------------------------------------------------------------
# 5. The RL environment wrapper
# ---------------------------------------------------------------------------
def check_rl_env() -> str:
    import sumo_setup  # noqa: F401
    from sumo_rl import SumoEnvironment

    env = SumoEnvironment(
        net_file=str(ROOT / "networks" / "single_intersection.net.xml"),
        route_file=str(ROOT / "networks" / "routes" / "ns_heavy_seed0.rou.xml"),
        use_gui=False,
        num_seconds=300,
        delta_time=5,
        yellow_time=4,
        min_green=10,
        single_agent=True,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log true --duration-log.disable true",
    )
    try:
        observation, _ = env.reset()
        total = 0.0
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(
                env.action_space.sample())
            total += float(reward)
            if terminated or truncated:
                break
        signal = env.traffic_signals[env.ts_ids[0]]
        return ("observation space  %s\naction space       %s\n"
                "green phases       %d\n10 random steps ran, cumulative reward %.1f"
                % (env.observation_space, env.action_space,
                   signal.num_green_phases, total))
    finally:
        env.close()


CHECKS = [
    ("1. SUMO installation", check_sumo),
    ("2. Python packages", check_packages),
    ("3. Network and demand files", check_files),
    ("4. Headless SUMO simulation (300 simulated seconds)", check_simulation),
    ("5. sumo-rl reinforcement-learning environment", check_rl_env),
]


def main() -> int:
    verbose = "-v" in sys.argv
    print("=" * 70)
    print("Adaptive Traffic Signal Control - environment smoke test")
    print("=" * 70)

    failures = []
    for label, fn in CHECKS:
        print("\n" + label)
        try:
            for line in (fn() or "").splitlines():
                print(PASS + line)
        except Exception as exc:  # noqa: BLE001 - report anything that breaks
            print(FAIL + str(exc))
            failures.append((label, exc))
            if verbose:
                traceback.print_exc()

    print("\n" + "=" * 70)
    if failures:
        print("%d CHECK(S) FAILED" % len(failures))
        for label, exc in failures:
            print("  - %s: %s" % (label, exc))
        if not verbose:
            print("\nRe-run with -v for full tracebacks.")
        print("=" * 70)
        return 1
    print("ALL CHECKS PASSED - the toolchain is working.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
