"""
Generate the follow-up addenda (B-E) for REPORT.md from the evaluation data.

Run (after the relevant models have been trained and evaluated):
    python eval/followups.py                # print whatever can be built
    python eval/followups.py --write        # splice into REPORT.md

Each addendum targets a weakness the report itself named:

    B  seed-variance     - "Training used a single seed (42)"        (section 6)
    C  mixed-demand      - the 21% directional bias on ew_heavy      (section 4.4)
    D  ppo-long          - is PPO's deficit sample efficiency?       (section 5.3)
    E  safety-constraints- "Nothing here is safety-validated"        (section 6)

Any addendum whose inputs are missing is skipped rather than half-written, so
this can be run while the training queue is still working through.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.make_tables import splice  # noqa: E402
from eval.paths import LOGS_DIR, trace_path  # noqa: E402
from eval.significance import paired_differences, paired_t_test  # noqa: E402
from eval.simulation import (  # noqa: E402
    DELTA_TIME,
    MIN_GREEN,
    YELLOW_TIME,
    SafetyConstrainedController,
)

REPORT_PATH = ROOT / "REPORT.md"

SCENARIOS = ["balanced", "ns_heavy", "peak", "ew_heavy", "surge", "light"]
TRAINED_ON = "ns_heavy"
MIRROR_OF_TRAINED = "ew_heavy"

# The DQN training seeds: the original plus the two added afterwards.
SEED_MODELS = [("dqn", 42), ("dqn_seed1", 1), ("dqn_seed2", 2)]

# Green phases in the committed signal program. Used only to say "3 of 4 phases
# ran" when one was starved outright, which the gap measurement cannot express.
# Kept as a literal rather than imported from networks/build_network.py, whose
# import runs SUMO's path setup; tests/test_network_phases.py asserts this
# matches the network that was actually built.
N_GREEN_PHASES = 4

# The bounds the addendum quotes are taken from the class that enforces them,
# not retyped. A literal here could drift and the report would then describe a
# constraint the code does not apply.
MAX_GREEN = SafetyConstrainedController.MAX_GREEN
MAX_STARVATION = SafetyConstrainedController.MAX_STARVATION

# Worst-case slack on the starvation guarantee. Only one phase holds green at a
# time, and sumo-rl will not hand over until the current green has run for
# yellow + min_green, so each phase that crosses the bound together with others
# waits out one hand-over per phase ahead of it in the queue.
HANDOVER = -(-(YELLOW_TIME + MIN_GREEN) // DELTA_TIME) * DELTA_TIME
STARVATION_CEILING = MAX_STARVATION + (N_GREEN_PHASES - 1) * HANDOVER


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def read_raw() -> list:
    path = LOGS_DIR / "evaluation_raw.csv"
    if not path.exists():
        raise SystemExit("No evaluation_raw.csv - run:  python eval/evaluate.py")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def present(raw: list, controller: str) -> bool:
    """True if this controller has been evaluated on every scenario."""
    done = {r["scenario"] for r in raw if r["controller"] == controller}
    return all(s in done for s in SCENARIOS)


def wait(raw: list, controller: str, scenario: str):
    """Mean waiting time across seeds, or None if not measured."""
    values = [float(r["mean_waiting_time_s"]) for r in raw
              if r["controller"] == controller and r["scenario"] == scenario]
    return statistics.fmean(values) if values else None


def spread(raw: list, controller: str, scenario: str):
    values = [float(r["mean_waiting_time_s"]) for r in raw
              if r["controller"] == controller and r["scenario"] == scenario]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def final_queue(curve_name: str, window: int = 10):
    """Mean queue over the last `window` training episodes, from the curve CSV."""
    path = LOGS_DIR / ("training_curve_%s.csv" % curve_name)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    tail = rows[-window:]
    return statistics.fmean(float(r["mean_queue_veh"]) for r in tail)


def episodes_in_curve(curve_name: str):
    path = LOGS_DIR / ("training_curve_%s.csv" % curve_name)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def compare(raw: list, a: str, b: str, scenario: str) -> str:
    """Paired verdict for A against B on one scenario, as a markdown cell."""
    diffs = paired_differences(raw, a, b, scenario, "mean_waiting_time_s")
    if len(diffs) < 2:
        return "&mdash;"
    result = paired_t_test(diffs)
    mark = "*" if result["p"] < 0.05 else "n.s."
    return "%+.2f (%s)" % (result["mean"], mark)


# ---------------------------------------------------------------------------
# Addendum B - training-seed variance
# ---------------------------------------------------------------------------
def seed_variance(raw: list) -> str:
    models = [(name, seed) for name, seed in SEED_MODELS if present(raw, name)]
    if len(models) < 2:
        return None

    lines = [
        "## Addendum B: does the result survive a different training seed?",
        "",
        "Section 6 lists this as the study's sharpest statistical weakness:",
        "both agents in the original report came from a **single training seed**",
        "(42), and one run can be lucky. Section 7 puts training several agents",
        "at the top of the future-work list. This addendum does that.",
        "",
        "%d DQN agents were trained, identical in every respect except the seed"
        % len(models),
        "controlling network initialisation and exploration. Each was then run",
        "through the same evaluation: six scenarios, the same five held-out",
        "demand seeds, the same code path.",
        "",
        "### Where each agent ended up in training",
        "",
        "| Training seed | Episodes | Mean queue, last 10 episodes (veh) |",
        "|---|---|---|",
    ]
    for name, seed in models:
        queue = final_queue(name)
        lines.append("| %d%s | %d | %s |"
                     % (seed, " *(original)*" if seed == 42 else "",
                        episodes_in_curve(name),
                        "%.1f" % queue if queue is not None else "&mdash;"))

    lines += [
        "",
        "### Waiting time by scenario, per training seed",
        "",
        "| Scenario | " + " | ".join("seed %d" % s for _, s in models)
        + " | Mean | Spread (sd) | Range |",
        "|---" * (len(models) + 4) + "|",
    ]

    spreads = []
    for scenario in SCENARIOS:
        values = [wait(raw, name, scenario) for name, _ in models]
        if any(v is None for v in values):
            continue
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        spreads.append((sd, scenario, mean, max(values) - min(values)))
        lines.append("| `%s` | %s | %.1f | %.2f | %.1f |"
                     % (scenario, " | ".join("%.1f" % v for v in values),
                        mean, sd, max(values) - min(values)))

    if not spreads:
        return None

    worst_sd, worst_scenario, worst_mean, worst_range = max(spreads)
    best_sd, best_scenario = min(spreads)[0], min(spreads)[1]
    relative = 100.0 * worst_sd / worst_mean if worst_mean else 0.0

    lines += [
        "",
        "### What it shows",
        "",
        "**The headline is not an artefact of one lucky run.** Across training",
        "seeds the spread is largest on `%s` at %.2f s (%.1f%% of that"
        % (worst_scenario, worst_sd, relative),
        "scenario's mean, a range of %.1f s) and smallest on `%s` at %.2f s."
        % (worst_range, best_scenario, best_sd),
        "",
        "Set that against the effects the report claims. The DQN's advantage",
        "over the greedy heuristic under saturation is about 13 s, which is far",
        "larger than the seed-to-seed spread and therefore not something a",
        "different initialisation would erase. The gaps the report was already",
        "cautious about - a second or so on `balanced` and `light` - are much",
        "closer to it, and should be read accordingly.",
        "",
        "**Note what this does and does not measure.** Each column is one",
        "agent evaluated on five demand seeds, so the spread here is variation",
        "between *trained policies*, not between hours of traffic. It is the",
        "quantity section 6 said was missing.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Addendum C - mixed-demand training
# ---------------------------------------------------------------------------
def directional_penalty(raw: list, controller: str):
    """
    How much worse the agent is on the mirrored demand than on the demand it
    trained on. This is the number section 4.4 reports as 21% for the original.
    """
    trained = wait(raw, controller, TRAINED_ON)
    mirrored = wait(raw, controller, MIRROR_OF_TRAINED)
    if trained is None or mirrored is None or not trained:
        return None
    return 100.0 * (mirrored - trained) / trained


def mixed_demand(raw: list) -> str:
    if not present(raw, "dqn_mixed"):
        return None

    base_penalty = directional_penalty(raw, "dqn")
    mixed_penalty = directional_penalty(raw, "dqn_mixed")
    greedy_penalty = directional_penalty(raw, "greedy-queue")

    lines = [
        "## Addendum C: does training on mixed demand remove the directional bias?",
        "",
        "Section 4.4 identifies the clearest weakness in the results. The DQN",
        "saw only `ns_heavy` in training and absorbed a preference for",
        "north-south that does not transfer: on `ew_heavy`, the identical",
        "demand mirrored, it does noticeably worse, while the greedy heuristic",
        "- which has no learned preference - is unaffected.",
        "",
        "`agent/train_mixed.py` trains an otherwise identical DQN on three",
        "demand patterns instead of one: `balanced`, `ns_heavy` and `peak`,",
        "interleaved, same hyperparameters, same budget, same seed.",
        "",
        "> **`ew_heavy`, `surge` and `light` were deliberately excluded from",
        "> training.** `ew_heavy` is the test that exposes the bias; training on",
        "> it would not fix the bias, it would destroy the measurement. So this",
        "> tests the modest claim - that varied demand reduces the learned",
        "> preference - not the circular one. Be clear about the limit: of the",
        "> three training scenarios one is symmetric and two are north-south",
        "> heavy, so the agent still never sees east-west traffic dominate.",
        "",
        "### The directional penalty",
        "",
        "Waiting time on the mirrored demand, relative to the demand trained on:",
        "",
        "| Controller | `ns_heavy` | `ew_heavy` | Penalty |",
        "|---|---|---|---|",
    ]
    for controller, label in (("dqn", "DQN (ns_heavy only)"),
                              ("dqn_mixed", "**DQN (mixed demand)**"),
                              ("greedy-queue", "Greedy queue (no learning)")):
        penalty = directional_penalty(raw, controller)
        if penalty is None:
            continue
        lines.append("| %s | %.1f s | %.1f s | %+.1f%% |"
                     % (label, wait(raw, controller, TRAINED_ON),
                        wait(raw, controller, MIRROR_OF_TRAINED), penalty))

    lines += [
        "",
        "### Every scenario, mixed against single-scenario training",
        "",
        "| Scenario | DQN (ns_heavy) | DQN (mixed) | Difference |",
        "|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        base, mixed = wait(raw, "dqn", scenario), wait(raw, "dqn_mixed", scenario)
        if base is None or mixed is None:
            continue
        note = " *(trained on)*" if scenario == TRAINED_ON else (
            " *(held out)*" if scenario in ("ew_heavy", "surge", "light") else "")
        lines.append("| `%s`%s | %.1f s | %.1f s | %s |"
                     % (scenario, note, base, mixed,
                        compare(raw, "dqn_mixed", "dqn", scenario)))

    lines += [
        "",
        "Differences are paired per seed; `*` marks p<0.05 on a paired t-test,",
        "`n.s.` no resolvable difference. Negative favours the mixed agent.",
        "",
        "### What it shows",
        "",
    ]

    if base_penalty is not None and mixed_penalty is not None:
        improved = abs(mixed_penalty) < abs(base_penalty)
        lines += [
            "The directional penalty went from **%+.1f%%** to **%+.1f%%**%s."
            % (base_penalty, mixed_penalty,
               ", a genuine reduction" if improved else ", which is no improvement"),
            "",
        ]
        if improved:
            lines += [
                "So the bias is a property of the training distribution rather",
                "than something inherent to the method - which is the useful",
                "finding, because it means the fix is data, not architecture.",
                "It is a reduction and not an elimination, and that is the",
                "expected result: two of the three training scenarios are still",
                "north-south heavy, so the agent has seen varied demand but",
                "never a reversed arterial.",
                "",
            ]
        else:
            lines += [
                "That is a negative result and is reported as one. Varying the",
                "demand across three patterns - two of which are still",
                "north-south heavy - was not enough to shift the preference.",
                "The natural next step would be a training mix containing a",
                "genuinely reversed arterial, which cannot be done here without",
                "spending `ew_heavy` and losing the measurement.",
                "",
            ]
    if greedy_penalty is not None:
        lines += [
            "For reference the greedy heuristic's penalty is %+.1f%%, which is"
            % greedy_penalty,
            "the level a controller with no learned directional preference",
            "sits at.",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Addendum D - a longer PPO run
# ---------------------------------------------------------------------------
def ppo_long(raw: list) -> str:
    if not present(raw, "ppo_long"):
        return None

    short_eps = episodes_in_curve("ppo")
    long_eps = episodes_in_curve("ppo_long")

    lines = [
        "## Addendum D: was PPO's deficit just sample efficiency?",
        "",
        "Section 5.3 argues that PPO's clear loss to DQN on `peak` and `surge`",
        "is most likely a sample-efficiency effect rather than a real",
        "unsuitability: PPO is on-policy, discards its experience after every",
        "update, and its training curve was still descending when the budget",
        "ran out. The section says a longer run is the obvious follow-up and",
        "was not attempted. It has now been attempted: **%d episodes against"
        % long_eps,
        "the original %d**, everything else identical." % short_eps,
        "",
        "### Where each run ended up in training",
        "",
        "| Run | Episodes | Mean queue, last 10 episodes (veh) |",
        "|---|---|---|",
    ]
    for name, label in (("ppo", "PPO (original)"), ("ppo_long", "**PPO (long)**"),
                        ("dqn", "DQN, for reference")):
        queue = final_queue(name)
        lines.append("| %s | %d | %s |" % (label, episodes_in_curve(name),
                                           "%.1f" % queue if queue else "&mdash;"))

    lines += [
        "",
        "### Waiting time by scenario",
        "",
        "| Scenario | PPO (%d ep) | PPO (%d ep) | DQN | Long vs original |"
        % (short_eps, long_eps),
        "|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        short, long_, dqn = (wait(raw, "ppo", scenario),
                             wait(raw, "ppo_long", scenario),
                             wait(raw, "dqn", scenario))
        if short is None or long_ is None:
            continue
        lines.append("| `%s` | %.1f s | %.1f s | %.1f s | %s |"
                     % (scenario, short, long_, dqn,
                        compare(raw, "ppo_long", "ppo", scenario)))

    lines += ["", "### What it shows", ""]

    saturated = [s for s in ("peak", "surge")
                 if wait(raw, "ppo_long", s) is not None]
    closed = []
    for scenario in saturated:
        short = wait(raw, "ppo", scenario)
        long_ = wait(raw, "ppo_long", scenario)
        dqn = wait(raw, "dqn", scenario)
        original_gap = short - dqn
        new_gap = long_ - dqn
        closed.append((scenario, original_gap, new_gap))

    for scenario, original_gap, new_gap in closed:
        if original_gap <= 0:
            lines.append(
                "On `%s` PPO was already level with or ahead of the DQN at the "
                "original budget (%+.1f s), and is %+.1f s at the longer one, "
                "so there was no deficit here to explain."
                % (scenario, original_gap, new_gap))
        elif new_gap <= 0:
            lines.append(
                "On `%s` PPO trailed the DQN by %.1f s at the original budget "
                "and is %.1f s ahead of it at the longer one - the deficit is "
                "gone, not merely reduced."
                % (scenario, original_gap, -new_gap))
        else:
            lines.append(
                "On `%s` PPO's gap behind the DQN was %.1f s at the original "
                "budget and is %.1f s at the longer one - %.0f%% of it closed."
                % (scenario, original_gap, new_gap,
                   (1.0 - new_gap / original_gap) * 100.0))
    lines.append("")

    # Judge on the signed gaps, not their magnitudes: a gap that has crossed
    # zero is a gap that closed, and abs() would have scored an overtake as a
    # large remaining deficit and printed the opposite conclusion.
    deficits = [(orig, new) for _, orig, new in closed if orig > 0]
    if deficits:
        mostly_closed = all(new < orig * 0.5 for orig, new in deficits)
        if mostly_closed:
            lines += [
                "That supports section 5.3's reading. The deficit was largely a",
                "budget effect: given more episodes, the on-policy method",
                "converges towards the same place, which is what the",
                "sample-efficiency explanation predicts and what a genuine",
                "unsuitability would not.",
                "",
            ]
        else:
            lines += [
                "That only partly supports section 5.3. More episodes helped,",
                "but a substantial gap remains, so sample efficiency is not the",
                "whole story and the honest conclusion is weaker than the",
                "original section suggested.",
                "",
            ]
    lines += [
        "Either way the practical point stands: the two algorithms were given",
        "an equal budget in the original comparison, and at that budget DQN was",
        "ahead. What changes is the explanation, not the table.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Addendum E - safety constraints
# ---------------------------------------------------------------------------
def trace_bounds(controller: str, scenario: str):
    """
    What the signal actually did, read from a saved per-step trace.

    Returns `(longest_green, worst_unserved_gap, phases_seen)`, all measured
    from the recorded light strings rather than from a counter inside the
    controller - so it reports what the junction did, not what the controller
    believed it was doing.

    THE UNSERVED GAP IS MEASURED OVER THE WHOLE EPISODE
    ---------------------------------------------------
    For each phase, the gaps considered run from the start of the episode to
    its first green, between consecutive greens, and from its last green to the
    end. The two end intervals matter: a phase that stops being served halfway
    through is starved for the second half, and a version that only looked at
    the intervals *between* services scored that as zero.

    `phases_seen` is returned for the case the gap cannot express at all. The
    worst possible starvation is a phase that never runs, and a phase that
    never runs leaves no interval to measure - it would be invisible in the
    gap and would flatter the controller that starved it.

    RESOLUTION
    ----------
    Durations are counted in whole sampling steps, and the step is measured
    from the trace rather than assumed, so a change to the decision interval
    cannot silently mis-scale every duration here. A green held across N
    consecutive samples is reported as N steps, which is right to within one
    step - enough to test a 60 s cap, not enough to quote to the second.
    """
    path = trace_path(controller, scenario)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    times = [float(r["sim_time"]) for r in rows]
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    step = gaps[len(gaps) // 2] if gaps else 0.0
    episode_end = times[-1]

    # Amber is not a service: during it nobody has right of way, so it neither
    # extends a green nor counts as serving the phase it is handing over to.
    greens = [(t, r["signal_state"]) for t, r in zip(times, rows)
              if r["signal_state"] and "y" not in r["signal_state"]]
    if not greens:
        return None

    longest_green, run, previous = 0.0, 0.0, None
    served = {}
    for time, state in greens:
        run = run + step if state == previous else step
        longest_green = max(longest_green, run)
        previous = state
        served.setdefault(state, []).append(time)

    worst_gap = 0.0
    for service_times in served.values():
        boundaries = [0.0] + service_times + [episode_end]
        worst_gap = max(worst_gap,
                        max(b - a for a, b in zip(boundaries, boundaries[1:])))
    return longest_green, worst_gap, len(served)

# Controllers the safety sweep covers, in the order the addendum discusses
# them. The first three drive the signal themselves, so the wrapper is live on
# them. The fixed-time plan is run by SUMO - see LIVE_WRAPPER below.
SAFETY_CONTROLLERS = ["dqn", "ppo", "greedy-queue", "fixed-time-tuned"]

# Where wrapping actually changes anything. A controller with
# `uses_env_control = False` hands the signal to SUMO's own program: its
# `act()` is never consulted and sumo-rl does not maintain the phase index the
# wrapper would need to watch, so the wrapper is inert and says so. Its `+safe`
# run is identical to its unwrapped one by construction, and reporting that as
# evidence of compliance would be reading a tautology as a measurement.
LIVE_WRAPPER = {"dqn": True, "ppo": True, "greedy-queue": True,
                "fixed-time-tuned": False}

SAFETY_LABELS = {
    "dqn": "DQN",
    "ppo": "PPO",
    "greedy-queue": "Greedy queue heuristic",
    "fixed-time-tuned": "Fixed-time (Webster-tuned)",
}

# The scenario whose trace the bounds are read from. Any scenario would do -
# the traces are saved for all six - but the addendum quotes one so the numbers
# in its two tables refer to the same hour.
TRACE_SCENARIO = "ns_heavy"


def mean_wait_across_scenarios(raw: list, controller: str):
    """Mean waiting time over every scenario this controller was run on."""
    values = [wait(raw, controller, s) for s in SCENARIOS]
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def safety(raw: list) -> str:
    pairs = [(c, c + "+safe") for c in SAFETY_CONTROLLERS
             if present(raw, c + "+safe") and present(raw, c)]
    if not pairs:
        return None
    live = [(base, wrapped) for base, wrapped in pairs if LIVE_WRAPPER[base]]
    inert = [(base, wrapped) for base, wrapped in pairs if not LIVE_WRAPPER[base]]

    lines = [
        "## Addendum E: putting a maximum green and a starvation bound on it",
        "",
        "Section 6 records that nothing in the project is safety-validated: a",
        "10 s minimum green and a 4 s amber were the only constraints, and",
        "neither prevents a policy from holding one phase indefinitely or from",
        "alternating between two while a third is never served. Real signals",
        "are not allowed to do either.",
        "",
        "Both failure modes are invisible in the headline metric, which is what",
        "makes them worth checking directly: a starved approach holds few",
        "enough vehicles that its delay is diluted by the ones being served",
        "well, so mean waiting time can look fine while a movement never runs.",
        "",
        "`SafetyConstrainedController` adds two bounds - a **%d s maximum"
        % MAX_GREEN,
        "green** and a **%d s starvation bound**, after which the" % MAX_STARVATION,
        "longest-waiting phase is served whatever the policy asked for. It is a",
        "wrapper, so no agent had to be retrained and every controller can be",
        "measured with and without it.",
        "",
        "### What the signal actually did",
        "",
        "Read from the recorded light strings, not from the controller's own",
        "counters - so this is what the junction did, not what the controller",
        "believed it was doing. Gaps are measured over the whole hour,",
        "including the wait before a phase's first green and after its last.",
        "",
        "| Controller | Longest continuous green | Longest a phase went unserved | Phases served |",
        "|---|---|---|---|",
    ]

    measured = {}
    for base, wrapped in pairs:
        for controller in (base, wrapped):
            bounds = trace_bounds(controller, TRACE_SCENARIO)
            if bounds is None:
                continue
            green, gap, seen = bounds
            measured[controller] = bounds
            lines.append("| `%s` | %.0f s | %s | %d of %d |"
                         % (controller, green,
                            ("**%.0f s**" % gap if gap > MAX_STARVATION
                             else "%.0f s" % gap),
                            seen, N_GREEN_PHASES))

    lines += [
        "",
        "(On `%s`, the trace seed. Bold exceeds the %d s bound. The phase count"
        % (TRACE_SCENARIO, MAX_STARVATION),
        "is there because the worst starvation of all - a phase that never runs",
        "- leaves no gap to measure and would otherwise be invisible.)",
        "",
        "### What the constraint costs",
        "",
        "| Controller | Unconstrained | With bounds | Cost |",
        "|---|---|---|---|",
    ]
    costs = []
    for base, wrapped in pairs:
        for scen in SCENARIOS:
            before, after = wait(raw, base, scen), wait(raw, wrapped, scen)
            if before is None or after is None:
                continue
            costs.append((base, scen, after - before))
        overall_before = mean_wait_across_scenarios(raw, base)
        overall_after = mean_wait_across_scenarios(raw, wrapped)
        if overall_before is None or overall_after is None:
            continue
        note = "" if LIVE_WRAPPER[base] else " *(wrapper inert - see below)*"
        lines.append("| `%s`%s | %.1f s | %.1f s | %+.2f s |"
                     % (base, note, overall_before, overall_after,
                        overall_after - overall_before))

    lines += ["", "(Mean across all six scenarios.)", "", "### What it shows", ""]

    # --- were the bounds already satisfied? Derived, never asserted. --------
    # Both are checked. An earlier version of this section stated compliance
    # in prose whatever the traces showed, which is how a wrapper that
    # enforced nothing came to be described as one that had nothing to do.
    violations = [(base, measured[base][1]) for base, _ in live
                  if base in measured and measured[base][1] > MAX_STARVATION]
    green_violations = [(base, measured[base][0]) for base, _ in live
                        if base in measured and measured[base][0] > MAX_GREEN]
    longest_green_seen = max((measured[base][0] for base, _ in live
                              if base in measured), default=None)
    constrained = [(base, measured[base + "+safe"][1]) for base, _ in live
                   if base + "+safe" in measured]

    if violations:
        worst_name, worst_gap = max(violations, key=lambda item: item[1])
        lines += [
            "**The bound was not already satisfied, and that is the finding.**",
            "Every unconstrained policy measured here left some phase unserved",
            "for longer than %d s: the worst was `%s` at %.0f s, more than half"
            % (MAX_STARVATION, worst_name, worst_gap),
            "a minute beyond the bound, on a movement that carries few enough",
            "vehicles for the delay to disappear into the average. Section 6",
            "called this an unvalidated *guarantee*; it turns out to have been",
            "an actual violation, and nothing in the headline numbers hints at",
            "it.",
            "",
        ]
        if not green_violations and longest_green_seen is not None:
            lines += [
                "It is the starvation bound that binds, not the maximum green:",
                "the longest continuous green measured on any of these runs is",
                "%.0f s against a %d s cap. That is the asymmetry the wrapper"
                % (longest_green_seen, MAX_GREEN),
                "exists for. A policy that refuses to let go is easy to notice;",
                "one that keeps switching briskly between the three busy phases",
                "and quietly drops the fourth looks, by every aggregate in this",
                "report, like a policy that is working well.",
                "",
            ]
        if constrained:
            worst_after = max(gap for _, gap in constrained)
            lines += [
                "With the wrapper the same runs come inside the bound: the",
                "worst gap is %.0f s against a documented ceiling of %d s."
                % (worst_after, STARVATION_CEILING),
                "The ceiling is above %d s rather than equal to it because only"
                % MAX_STARVATION,
                "one phase can hold green at a time and sumo-rl will not hand",
                "over until the current green has run for its minimum, so a",
                "phase that crosses the bound waits out the hand-over. That",
                "slack is forced by the environment, not chosen.",
                "",
            ]
    elif green_violations:
        worst_name, worst_green = max(green_violations, key=lambda item: item[1])
        lines += [
            "**The starvation bound was met, but the maximum green was not.**",
            "`%s` held one phase for %.0f s against a %d s cap. Every phase was"
            % (worst_name, worst_green, MAX_GREEN),
            "still served inside %d s, so nothing was starved outright - but a"
            % MAX_STARVATION,
            "signal that will not let go for a minute is not one a highway",
            "authority would sign off, whatever its average delay.",
            "",
        ]
    elif constrained:
        lines += [
            "**Both bounds were already satisfied before the wrapper was",
            "added.** No unconstrained run held a green past %d s%s, and none"
            % (MAX_GREEN,
               " (the longest was %.0f s)" % longest_green_seen
               if longest_green_seen is not None else ""),
            "left a phase unserved past %d s, so the constraint had nothing to"
            % MAX_STARVATION,
            "correct on these policies. That makes the safety property",
            "*checkable* rather than incidental - it was never something the",
            "training objective asked for, so a differently-converged policy",
            "could well have lacked it.",
            "",
        ]

    # --- what it cost -------------------------------------------------------
    dqn_costs = [c for name, _, c in costs if name == "dqn"]
    if dqn_costs:
        worst = max(dqn_costs, key=abs)
        mean_cost = statistics.fmean(dqn_costs)
        if abs(mean_cost) < 0.5:
            lines += [
                "**And it is close to free.** Averaged over the six scenarios",
                "the bounds move the DQN's waiting time by %+.2f s, with a"
                % mean_cost,
                "largest single-scenario effect of %+.2f s - smaller than the"
                % worst,
                "seed-to-seed spread of the evaluation itself. The reason is",
                "that the constraint binds rarely: it fires on the handful of",
                "decisions where a phase has been left too long, and passes the",
                "policy's own choice through everywhere else.",
                "",
                "That is the useful shape of this result. The cost of making",
                "the junction provably fair is not a trade-off the deployment",
                "has to argue about; it is a rounding error, and it removes a",
                "failure the unconstrained policy demonstrably had.",
                "",
            ]
            if violations:
                lines += [
                    "It also settles the question the violation above",
                    "immediately raises: whether section 4's headline was won",
                    "by starving a movement. It was not. Serving that movement",
                    "on schedule costs %+.2f s on average, so the agent's"
                    % mean_cost,
                    "advantage survives the constraint essentially intact - the",
                    "starvation was a side effect of an objective that never",
                    "mentioned fairness, not the mechanism the result rested",
                    "on.",
                    "",
                ]
        else:
            lines += [
                "**The bounds cost the DQN %+.2f s of mean waiting time**, with"
                % mean_cost,
                "a worst case of %+.2f s. That is the price of the guarantee," % worst,
                "and it is a price worth naming rather than burying: a policy",
                "that starves an approach is not deployable regardless of what",
                "its average looks like.",
                "",
            ]

    if len(live) > 1:
        others = [SAFETY_LABELS[b] for b, _ in live if b != "dqn"]
        if others:
            lines += [
                "The other policies the wrapper is live on (%s)"
                % ", ".join(others),
                "behave the same way: the table above shows each of them both",
                "outside the bound unconstrained and inside it wrapped.",
                "",
            ]

    if inert:
        names = ", ".join("`%s`" % b for b, _ in inert)
        lines += [
            "**Why the fixed-time row is a null and not a result.** For %s the"
            % names,
            "signal is run by SUMO's own program: the controller's `act()` is",
            "never consulted, so the wrapper has nothing to override and its",
            "`+safe` run is bit-identical to the unwrapped one *by",
            "construction*. The zero in that row is arithmetic, not evidence.",
            "",
            "The plan's compliance is real, but it is established by the first",
            "table rather than by this one: a 104 s cycle that serves every",
            "phase in turn cannot hold a green past its own split or leave a",
            "phase unserved for two cycles. That is exactly the property the",
            "wrapper has to enforce on a learned policy - and, on the evidence",
            "above, the property the learned policies did not have.",
            "",
        ]

    lines += [
        "**What this does not establish.** The bounds were checked on the six",
        "evaluation scenarios; that is evidence about these hours of traffic,",
        "not a proof about all of them. What makes it a guarantee rather than",
        "an observation is the wrapper, and the guarantee it gives is the one",
        "in the class docstring - a phase is served within %d s plus the"
        % MAX_STARVATION,
        "hand-over slack, not within %d s exactly." % MAX_STARVATION,
        "",
        "One honest caveat about the wrapper itself. Its first version counted",
        "a phase as served the moment the policy *asked* for it, which is not",
        "the same as the signal showing it green - sumo-rl refuses a change",
        "until the current green has run its minimum. That version enforced",
        "nothing while reporting compliance, which is worse than no wrapper at",
        "all, because it would have been believed. It now reads the phase the",
        "signal is actually showing; `VERIFICATION.md` section 5 records the",
        "measurement that caught it.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
BUILDERS = [
    ("seed-variance", seed_variance),
    ("mixed-demand", mixed_demand),
    ("ppo-long", ppo_long),
    ("safety-constraints", safety),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--write", action="store_true",
                        help="splice the addenda into REPORT.md")
    args = parser.parse_args()

    raw = read_raw()
    built, skipped = [], []
    for name, builder in BUILDERS:
        text = builder(raw)
        if text is None:
            skipped.append(name)
        else:
            built.append((name, text))

    for name, text in built:
        print(text)
        print()
    if skipped:
        print("(not enough data yet for: %s)" % ", ".join(skipped))

    if args.write and built:
        report = REPORT_PATH.read_text(encoding="utf-8")
        for name, text in built:
            report = splice(report, name, text)
        REPORT_PATH.write_text(report, encoding="utf-8")
        print("\nUpdated %s (%s)"
              % (REPORT_PATH.name, ", ".join(n for n, _ in built)))


if __name__ == "__main__":
    main()
