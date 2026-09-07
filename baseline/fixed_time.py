"""
The fixed-time (non-adaptive) signal controllers we compare the RL agent against.

Two of them:

  fixed-time         The plan compiled into the network: equal green for
                     north-south and east-west. This is "a standard fixed-cycle
                     signal plan" as the brief asks for, and it is what a signal
                     looks like when nobody has retimed it for current demand.

  fixed-time-tuned   The same idea, but with the green split calculated from
                     the demand using Webster's method - the standard textbook
                     procedure a traffic engineer would actually apply.

WHY BOTH
--------
Beating an untuned signal is a weak result: the untuned plan gives the busy
arterial and the quiet side street identical green time, so almost anything
that looks at traffic will beat it. If we only reported that comparison, the
honest reading would be "we beat a strawman".

The tuned plan is the real bar. It gets the green split right for the average
demand, and the only thing it still cannot do is react to the moment-to-moment
randomness of arrivals. So the gap between `fixed-time-tuned` and the RL agent
isolates the thing adaptive control is actually supposed to buy you, rather
than measuring how badly the baseline was configured.

Note that Webster's method needs to be told the demand. That is a real
advantage we are handing the baseline - it gets information the RL agent never
receives - and it is deliberate: we want the baseline to be as strong as it
reasonably can be.

WEBSTER'S METHOD IN ONE PARAGRAPH
---------------------------------
For each phase, work out its "flow ratio" y = (vehicles wanting to go) /
(vehicles that could go if the light stayed green). Add these up over the
phases to get Y. Lost time L is the time each cycle that nobody can use - here
the four 4-second ambers, so 16 s. Webster's cycle length that minimises delay
is C = (1.5L + 5) / (1 - Y). The usable green in that cycle is C - L, and it
is shared out between the phases in proportion to their flow ratios.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.simulation import MIN_GREEN, YELLOW_TIME, Controller  # noqa: E402
from networks.build_network import (  # noqa: E402
    EAST_WEST,
    NORTH_SOUTH,
    PHASE_NAMES,
)
from networks.generate_routes import SCENARIOS, TURN_SPLIT  # noqa: E402

# Vehicles per hour one lane can discharge while its light stays green.
# 1800 veh/h (one vehicle every 2 s) is the standard saturation flow rate for
# an urban through lane.
SATURATION_FLOW = 1800.0

# Webster's cycle is unbounded as demand approaches capacity, so it has to be
# clamped. 40-120 s is the usual practical range for an urban intersection:
# below 40 s the lost time dominates, above 120 s waits become unacceptable.
MIN_CYCLE = 40
MAX_CYCLE = 120


def flow_ratios(scenario: str) -> list:
    """
    Flow ratio y for each of the four phases, in phase order.

    Phase order is [NS-through, NS-left, EW-through, EW-left], matching
    networks/build_network.py.

    Each approach has one kerb lane carrying straight + right traffic and one
    median lane carrying only left turns, so the demand on a phase is the
    demand of the movements it serves, and its capacity is one lane's
    saturation flow. Opposing approaches (north and south) run together in the
    same phase, so the phase is sized by the busier of the two - that is the
    "critical" movement.
    """
    flows = SCENARIOS[scenario]["flows"]
    through_share = TURN_SPLIT["straight"] + TURN_SPLIT["right"]
    left_share = TURN_SPLIT["left"]

    def critical(approaches: set, share: float) -> float:
        return max(flows[a] for a in approaches) * share / SATURATION_FLOW

    return [
        critical(NORTH_SOUTH, through_share),
        critical(NORTH_SOUTH, left_share),
        critical(EAST_WEST, through_share),
        critical(EAST_WEST, left_share),
    ]


# MIN_GREEN and YELLOW_TIME are imported from eval.simulation rather than
# redeclared here. They are the two constraints that make the comparison fair -
# every controller must pay the same amber cost and respect the same minimum
# green - so there must be exactly one definition of each in the project. A
# second copy that drifted out of step would hand one controller an advantage
# silently, with no error and nothing visibly wrong in the results.


def allocate_greens(cycle: float, ratios: list) -> list:
    """
    Split a cycle's usable green between the phases, respecting the minimum.

    Green is shared in proportion to flow ratio, but no phase may drop below
    MIN_GREEN. Phases that would fall below it are pinned there and the
    remaining green is re-shared among the others - repeatedly, because pinning
    one phase reduces what is left for the rest and can push a second phase
    under the floor too.

    Returns None if the cycle is too short to give every phase its minimum.
    """
    lost_time = len(ratios) * YELLOW_TIME
    usable = cycle - lost_time
    n = len(ratios)
    if usable < n * MIN_GREEN:
        return None

    greens = [0.0] * n
    pinned = [False] * n
    for _ in range(n + 1):  # each pass pins at least one phase, so this ends
        free = [i for i in range(n) if not pinned[i]]
        if not free:
            break
        remaining = usable - sum(greens[i] for i in range(n) if pinned[i])
        free_ratio = sum(ratios[i] for i in free)
        for i in free:
            greens[i] = (remaining * ratios[i] / free_ratio if free_ratio > 0
                         else remaining / len(free))
        below = [i for i in free if greens[i] < MIN_GREEN]
        if not below:
            break
        for i in below:
            greens[i] = MIN_GREEN
            pinned[i] = True
    return greens


def webster_delay(cycle: float, green: float, flow_vph: float) -> float:
    """
    Webster's estimate of average delay per vehicle, in seconds.

    Two terms:
      - uniform delay, from the fact that arrivals are spread through a cycle
        but can only leave during green;
      - random delay, from arrivals clumping by chance, which grows sharply as
        the approach nears saturation.

    Returns a large number when demand exceeds capacity, where average delay is
    not finite - that is what stops the search below from choosing such a plan.
    """
    if flow_vph <= 0:
        return 0.0
    green_ratio = green / cycle
    capacity = SATURATION_FLOW * green_ratio
    saturation = flow_vph / capacity
    if saturation >= 0.98:
        return 1e6 * saturation  # oversaturated: unusable, but still ordered

    uniform = cycle * (1 - green_ratio) ** 2 / (2 * (1 - green_ratio * saturation))
    flow_vps = flow_vph / 3600.0
    random_delay = saturation ** 2 / (2 * flow_vps * (1 - saturation))
    return uniform + random_delay


def phase_demands(scenario: str) -> list:
    """Total vehicles/hour served by each phase, in phase order."""
    flows = SCENARIOS[scenario]["flows"]
    through_share = TURN_SPLIT["straight"] + TURN_SPLIT["right"]
    left_share = TURN_SPLIT["left"]
    return [
        sum(flows[a] for a in NORTH_SOUTH) * through_share,
        sum(flows[a] for a in NORTH_SOUTH) * left_share,
        sum(flows[a] for a in EAST_WEST) * through_share,
        sum(flows[a] for a in EAST_WEST) * left_share,
    ]


def webster_green_times(scenario: str) -> list:
    """
    Green duration in seconds for each phase, chosen to minimise predicted delay.

    WHY NOT JUST USE WEBSTER'S CYCLE FORMULA
    ----------------------------------------
    The textbook formula C = (1.5L + 5)/(1 - Y) assumes green can be split
    freely in proportion to demand. Here it cannot: every phase needs at least
    MIN_GREEN, and this intersection has four phases, two of which serve light
    left-turn demand.

    Applying the formula naively produces a plan that is *worse than doing
    nothing*. On the `balanced` scenario it returns a 52 s cycle, but the two
    left-turn phases then have to be raised to their 10 s minimum, which pushes
    the real cycle out to 64 s while the through phases keep the 14 s they were
    allocated for a 52 s cycle. Their share of the cycle therefore falls, and
    the through movement's demand-to-capacity ratio goes from 0.58 under the
    untuned plan to 0.81 - measurably more delay, from an operation that was
    supposed to reduce it.

    So instead of trusting the closed-form cycle, we search: for every cycle
    length in the practical range, allocate green under the minimum-green
    constraint and predict the resulting flow-weighted delay with Webster's
    delay formula. Then keep the best. The formula's cycle is what this search
    converges to whenever the constraint is not binding, and it is strictly
    better whenever it is.
    """
    ratios = flow_ratios(scenario)
    demands = phase_demands(scenario)

    best_greens, best_delay = None, float("inf")
    for cycle in range(MIN_CYCLE, MAX_CYCLE + 1):
        greens = allocate_greens(float(cycle), ratios)
        if greens is None:
            continue
        # Flow-weighted average delay across the four phases: the number a
        # randomly chosen driver at this intersection would expect to wait.
        total_flow = sum(demands)
        delay = sum(
            demand * webster_delay(float(cycle), green, demand)
            for green, demand in zip(greens, demands)
        ) / total_flow
        if delay < best_delay:
            best_greens, best_delay = greens, delay

    if best_greens is None:
        # Every candidate cycle was oversaturated; fall back to the longest
        # cycle, which maximises capacity.
        best_greens = allocate_greens(float(MAX_CYCLE), ratios)

    return [max(float(MIN_GREEN), round(g)) for g in best_greens]


def describe_plan(scenario: str) -> str:
    """Human-readable summary of the tuned plan, for logs and the report."""
    greens = webster_green_times(scenario)
    ratios = flow_ratios(scenario)
    cycle = sum(greens) + len(greens) * YELLOW_TIME
    lines = ["Webster-tuned plan for '%s'  (cycle %ds)" % (scenario, cycle)]
    for name, green, ratio in zip(PHASE_NAMES, greens, ratios):
        lines.append("    %-11s green %3ds   flow ratio %.3f" % (name, green, ratio))
    return "\n".join(lines)


class TunedFixedTimeController(Controller):
    """
    A fixed-cycle signal whose green split is sized for the scenario's demand.

    Like the untuned baseline it never looks at traffic while running: it
    installs its timing plan once, before any vehicle arrives, and then runs
    that plan blind for the whole hour.
    """

    name = "fixed-time-tuned"
    uses_env_control = False  # SUMO runs the plan; our `act` is ignored

    def __init__(self, env=None, scenario: str = "ns_heavy"):
        super().__init__(env)
        self.scenario = scenario

    def on_episode_start(self, env) -> None:
        """
        Overwrite the signal's timing plan with the Webster split.

        This has to happen here rather than in the network file because the
        right split depends on the scenario being run, and we evaluate the same
        network under six different demand patterns.
        """
        self.env = env
        traci_conn = env.sumo
        ts_id = env.ts_ids[0]

        # Read the existing program so we inherit its phase STATES (which
        # lights are green in which phase) and only replace the DURATIONS.
        logic = traci_conn.trafficlight.getAllProgramLogics(ts_id)[0]
        greens = webster_green_times(self.scenario)

        green_index = 0
        for phase in logic.phases:
            if "y" in phase.state:
                phase.duration = float(YELLOW_TIME)
            else:
                phase.duration = float(greens[green_index % len(greens)])
                green_index += 1
            # SUMO also honours minDur/maxDur for actuated programs; pin them
            # to the same value so the plan really is fixed-time.
            phase.minDur = phase.duration
            phase.maxDur = phase.duration

        traci_conn.trafficlight.setProgramLogic(ts_id, logic)
        # setProgramLogic leaves the signal mid-way through the old plan, so
        # restart it cleanly at phase 0.
        traci_conn.trafficlight.setPhase(ts_id, 0)

    def act(self, observation) -> int:
        return 0  # ignored


def tuned_fixed_time_factory(scenario: str):
    return lambda env: TunedFixedTimeController(env, scenario)


if __name__ == "__main__":
    # `python baseline/fixed_time.py` prints the tuned plan for every scenario,
    # which is handy for checking the timings before running an experiment.
    for name in SCENARIOS:
        print(describe_plan(name))
        print()
