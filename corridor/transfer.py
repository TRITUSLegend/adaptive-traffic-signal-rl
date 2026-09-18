"""
Zero-shot transfer of a corridor policy to a network it never trained on.

TWO KINDS OF TRANSFER, REPORTED SEPARATELY
------------------------------------------
They answer different questions and must not be pooled:

  1. **Clean transfer** - corridor2, corridor5, and a tighter-spaced corridor3.
     These are built by the same generator, so every junction has the same 4
     phases and 8 lanes the policy trained on. The ONLY thing that differs is
     the topology: how many junctions, who neighbours whom, how far apart.
     Nothing is adapted, so the measured degradation is topology and nothing
     else. This is the clean measurement.

  2. **Real-topology transfer** - `cologne3`, a real Cologne corridor from the
     RESCO benchmark. Much stronger evidence, because we did not draw it. But
     its junctions are structurally incompatible with ours, measured:

         360082      3 green phases,  5 controlled lanes  -> 14-value observation
         360086      4 green phases,  6 controlled lanes  -> 17-value observation
         GS_cluster  4 green phases,  8 controlled lanes  -> 21-value observation

     Our policy expects 4 phases and 8 lanes, i.e. 21 values. Only one of the
     three junctions fits. So a projection is unavoidable, and it is defined
     explicitly below rather than left implicit.

WHY THE PROJECTION IS REPORTED AS A CONFOUND, NOT A SOLUTION
------------------------------------------------------------
CORRIDOR_PLAN.md section 5c listed three options: evaluate only on matching
junctions, define an explicit projection, or report that real-topology transfer
was not achievable. Option (i) leaves one junction of three, which is not a
corridor. Option (iii) discards the brief's strongest robustness test. So
option (ii) is taken - and its cost is stated wherever the number appears:

    a cologne3 result mixes the difficulty of an unseen topology with the cost
    of the adaptation itself, and the two cannot be separated from that number
    alone. The clean transfers exist precisely so there is an unconfounded
    measurement to read it against.

WHAT THE BASELINES DO NOT NEED
------------------------------
Fixed-time and actuated are run by SUMO from the network's own program, and the
greedy heuristic reads the live signal whatever shape it is. **None of them
needs adapting.** Only the learned controllers do. That asymmetry is itself
worth reporting: it is a real deployment cost of a learned policy that a
conventional controller does not pay.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corridor import paths  # noqa: E402
from corridor.build_corridor import (  # noqa: E402
    DETECTOR_GAP,
    MAX_DUR,
    MAX_GAP,
    MIN_DUR,
)

# What the trained policy expects, and why: 4 green phases + 1 min-green flag
# + 8 lane densities + 8 lane queues.
TRAINED_PHASES = 4
TRAINED_LANES = 8
TRAINED_OBS = TRAINED_PHASES + 1 + 2 * TRAINED_LANES   # 21

COLOGNE3_DIR = (paths.ROOT / ".venv" / "Lib" / "site-packages" / "sumo_rl"
                / "nets" / "RESCO" / "cologne3")
COLOGNE3_NET = COLOGNE3_DIR / "cologne3.net.xml"
COLOGNE3_ROUTES = COLOGNE3_DIR / "cologne3.rou.xml"

# cologne3's demand spans 06:31-07:59; this is its busiest full hour, the one
# the corridor's own demand was calibrated against.
COLOGNE3_BEGIN = 25200


def adapt_observation(obs, n_phases: int, n_lanes: int) -> np.ndarray:
    """
    Project a junction's observation onto the 21 values the policy expects.

    Layout in, and out:  [phase one-hot][min-green flag][densities][queues]

    Shorter parts are zero-padded and longer parts are truncated. Zero is the
    neutral filler here: it reads as "no such phase" and as "an empty lane",
    which is what a junction with fewer phases or fewer lanes genuinely has.
    Truncation does lose information - a junction with more lanes than we
    trained on has approaches the policy simply cannot see - and that is part
    of the adaptation cost this function exists to make explicit.
    """
    obs = np.asarray(obs, dtype=np.float32)
    phases = obs[:n_phases]
    min_green = obs[n_phases:n_phases + 1]
    densities = obs[n_phases + 1:n_phases + 1 + n_lanes]
    queues = obs[n_phases + 1 + n_lanes:n_phases + 1 + 2 * n_lanes]

    def fit(part, width):
        out = np.zeros(width, dtype=np.float32)
        take = min(len(part), width)
        out[:take] = part[:take]
        return out

    return np.concatenate([
        fit(phases, TRAINED_PHASES),
        fit(min_green, 1),
        fit(densities, TRAINED_LANES),
        fit(queues, TRAINED_LANES),
    ])


def clamp_action(action: int, n_phases: int) -> int:
    """
    Keep an action inside the phases a junction actually has.

    A policy trained on four phases can ask for phase 3 at a junction with
    three. Clamping rather than wrapping keeps the choice adjacent to what was
    asked for; wrapping would turn "serve the last phase" into "serve the
    first", which is a different instruction rather than a nearby one.
    """
    return max(0, min(int(action), n_phases - 1))


class AdaptedPolicyController:
    """
    Drives an SB3 policy on a network whose junctions do not match training.

    On a corridor we generated this is a no-op in effect - every junction
    already has 4 phases and 8 lanes, so `adapt_observation` returns its input
    unchanged and `clamp_action` never binds. It only does real work on
    cologne3.
    """

    uses_env_control = True

    def __init__(self, policy, name: str, env=None):
        self.policy = policy
        self.name = name
        self.env = env
        self._shape = {}

    def on_episode_start(self, env) -> None:
        self.env = env
        self._shape = {}
        for ts_id in env.ts_ids:
            signal = env.traffic_signals[ts_id]
            self._shape[ts_id] = (int(signal.num_green_phases),
                                  len(signal.lanes))

    def act(self, observations: dict) -> dict:
        actions = {}
        for ts_id, obs in observations.items():
            n_phases, n_lanes = self._shape.get(
                ts_id, (TRAINED_PHASES, TRAINED_LANES))
            projected = adapt_observation(obs, n_phases, n_lanes)
            action, _ = self.policy.predict(projected, deterministic=True)
            actions[ts_id] = clamp_action(action, n_phases)
        return actions


class AdaptedMA2CController:
    """
    The same adaptation for MA2C, which additionally needs a neighbour graph.

    MA2C's agents are matched positionally onto the target's junctions, in
    sorted id order. On a corridor we built that is the true west-to-east
    order. On `cologne3` it is an assumption - sorted ids are not guaranteed to
    follow the road - and it is flagged wherever a cologne3 number is reported.
    """

    uses_env_control = True

    def __init__(self, model_path: Path, name: str = "ma2c", env=None):
        self.model_path = model_path
        self.name = name
        self.env = env
        self.model = None
        self._shape = {}

    def on_episode_start(self, env) -> None:
        from corridor.ma2c import MA2C

        self.env = env
        ts_ids = sorted(env.ts_ids)
        self.model = MA2C.load(self.model_path, ts_ids=ts_ids)
        self.model.reset_policies()
        self._shape = {}
        for ts_id in ts_ids:
            signal = env.traffic_signals[ts_id]
            self._shape[ts_id] = (int(signal.num_green_phases),
                                  len(signal.lanes))

    def act(self, observations: dict) -> dict:
        projected = {}
        for ts_id, obs in observations.items():
            n_phases, n_lanes = self._shape.get(
                ts_id, (TRAINED_PHASES, TRAINED_LANES))
            projected[ts_id] = adapt_observation(obs, n_phases, n_lanes)
        raw = self.model.act(projected, deterministic=True)
        return {ts: clamp_action(a, self._shape.get(ts, (TRAINED_PHASES, 0))[0])
                for ts, a in raw.items()}


def adapted_policy_factory(policy, name: str):
    return lambda env: AdaptedPolicyController(policy, name, env)


def adapted_ma2c_factory(model_path: Path, name: str = "ma2c"):
    return lambda env: AdaptedMA2CController(model_path, name, env)


# ---------------------------------------------------------------------------
# Preparing cologne3
# ---------------------------------------------------------------------------
def cologne3_actuated_net() -> Path:
    """
    Write an actuated copy of cologne3 into our own tree.

    The bundled network is never modified - it is a dependency's file and the
    real-data provenance depends on it being untouched. The copy keeps every
    phase exactly as the city defined it and changes only the timing rule, so
    the actuated baseline differs from cologne3's own fixed-time plan in
    responsiveness and nothing else.
    """
    out = paths.NETWORKS_DIR / "cologne3" / "cologne3_actuated.net.xml"
    out.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(COLOGNE3_NET)
    for logic in tree.getroot().iter("tlLogic"):
        logic.set("type", "actuated")
        logic.set("programID", "actuated")
        for key, value in (("max-gap", MAX_GAP), ("detector-gap", DETECTOR_GAP),
                           ("show-detectors", "false")):
            param = ET.SubElement(logic, "param")
            param.set("key", key)
            param.set("value", str(value))
        for phase in logic.iter("phase"):
            state = phase.get("state", "")
            if "y" in state:
                continue          # amber is fixed, never extended
            phase.set("minDur", str(MIN_DUR))
            phase.set("maxDur", str(MAX_DUR))
    tree.write(out, encoding="UTF-8", xml_declaration=True)
    return out


def cologne3_junction_shapes() -> dict:
    """
    Green-phase and controlled-lane counts per junction, read from the network.

    Used by the report to state exactly how far the target is from what the
    policy trained on, rather than asserting that it differs.
    """
    root = ET.parse(COLOGNE3_NET).getroot()
    lanes = {}
    for conn in root.iter("connection"):
        tl = conn.get("tl")
        if tl:
            lanes.setdefault(tl, set()).add(
                "%s_%s" % (conn.get("from"), conn.get("fromLane")))

    shapes = {}
    for logic in root.iter("tlLogic"):
        greens = [p.get("state") for p in logic.iter("phase")
                  if "y" not in p.get("state")
                  and ("G" in p.get("state") or "g" in p.get("state"))]
        shapes[logic.get("id")] = {
            "green_phases": len(greens),
            "lanes": len(lanes.get(logic.get("id"), ())),
        }
    return shapes


def describe_adaptation() -> str:
    """One line per junction, for the report's provenance."""
    lines = []
    for ts_id, shape in sorted(cologne3_junction_shapes().items()):
        obs = shape["green_phases"] + 1 + 2 * shape["lanes"]
        lines.append(
            "%s: %d green phases, %d lanes -> %d-value observation%s"
            % (ts_id, shape["green_phases"], shape["lanes"], obs,
               "" if obs == TRAINED_OBS else " (adapted to %d)" % TRAINED_OBS))
    return "\n".join(lines)
