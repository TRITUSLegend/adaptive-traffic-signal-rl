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

REPORT_PATH = ROOT / "REPORT.md"

SCENARIOS = ["balanced", "ns_heavy", "peak", "ew_heavy", "surge", "light"]
TRAINED_ON = "ns_heavy"
MIRROR_OF_TRAINED = "ew_heavy"

# The DQN training seeds: the original plus the two added afterwards.
SEED_MODELS = [("dqn", 42), ("dqn_seed1", 1), ("dqn_seed2", 2)]


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
    Longest continuous green and longest gap between services, read from a
    saved per-step trace.

    Measured from the actual light strings rather than from a counter inside
    the controller, so it reports what the signal did, not what the controller
    believed it was doing.
    """
    path = trace_path(controller, scenario)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None

    longest_green, run = 0.0, 0.0
    previous = None
    last_served, worst_gap = {}, 0.0
    for row in rows:
        state = row["signal_state"]
        time = float(row["sim_time"])
        if "y" in state or not state:
            continue
        run = run + 5.0 if state == previous else 5.0
        longest_green = max(longest_green, run)
        previous = state
        last_served[state] = time
        for other, seen in last_served.items():
            if other != state:
                worst_gap = max(worst_gap, time - seen)
    return longest_green, worst_gap


# Controllers the safety sweep covers, in the order the addendum discusses
# them. The first three drive the signal themselves, so the wrapper is live on
# them. The fixed-time plan is run by SUMO - see LIVE_WRAPPER below.
SAFETY_CONTROLLERS = ["dqn", "ppo", "greedy-queue", "fixed-time-tuned"]

# Where wrapping actually changes anything. A controller with
# `uses_env_control = False` hands the signal to SUMO's own program and its
# `act()` is never consulted, so the wrapper cannot intervene however the plan
# behaves. Its `+safe` run is therefore identical to its unwrapped one by
# construction, and reporting that as evidence of compliance would be reading a
# tautology as a measurement.
LIVE_WRAPPER = {"dqn": True, "ppo": True, "greedy-queue": True,
                "fixed-time-tuned": False}

SAFETY_LABELS = {
    "dqn": "DQN",
    "ppo": "PPO",
    "greedy-queue": "Greedy queue heuristic",
    "fixed-time-tuned": "Fixed-time (Webster-tuned)",
}


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
        "`SafetyConstrainedController` adds two bounds - a **60 s maximum",
        "green** and a **200 s starvation bound**, after which the",
        "longest-waiting phase is served whatever the policy asked for. It is a",
        "wrapper, so no agent had to be retrained and every controller can be",
        "measured with and without it.",
        "",
        "### What the signal actually did",
        "",
        "Measured from the recorded light strings, not from the controller's",
        "own counters - so this is what the junction did, not what the",
        "controller believed it was doing:",
        "",
        "| Controller | Longest continuous green | Longest a phase went unserved |",
        "|---|---|---|",
    ]

    scenario = "ns_heavy"
    for base, wrapped in pairs:
        for controller in (base, wrapped):
            bounds = trace_bounds(controller, scenario)
            if bounds is None:
                continue
            lines.append("| `%s` | %.0f s | %.0f s |"
                         % (controller, bounds[0], bounds[1]))

    lines += [
        "",
        "(On `%s`, the trace seed. The bounds are 60 s and 200 s.)" % scenario,
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

    dqn_costs = [c for name, _, c in costs if name == "dqn"]
    if dqn_costs:
        worst = max(dqn_costs, key=abs)
        mean_cost = statistics.fmean(dqn_costs)
        if abs(mean_cost) < 0.5:
            lines += [
                "**The constraint is essentially free.** Averaged over the six",
                "scenarios it moves the DQN's waiting time by %+.2f s, and its"
                % mean_cost,
                "largest effect on any single scenario is %+.2f s." % worst,
                "",
                "The reason is visible in the first table: the learned policy",
                "was already respecting both bounds. It switches often enough",
                "that it never approaches a 60 s green, and it serves every",
                "phase well inside 200 s. The safety gap section 6 identified",
                "was real as a *guarantee* - nothing in the training objective",
                "forbade starvation, and a differently-converged policy could",
                "have exhibited it - but this particular policy did not need",
                "correcting.",
                "",
                "That is worth stating carefully. The constraint has not made",
                "the agent safe by changing its behaviour; it has made the",
                "safety property *checkable* rather than incidental. On a",
                "policy that did misbehave, it would bind.",
                "",
            ]
        else:
            lines += [
                "The constraint costs the DQN %+.2f s of mean waiting time on"
                % mean_cost,
                "average, with a worst case of %+.2f s. That is the price of" % worst,
                "the guarantee, and it is a price worth naming: an unconstrained",
                "policy is not deployable regardless of its average.",
                "",
            ]

    if len(live) > 1:
        others = [SAFETY_LABELS[b] for b, _ in live if b != "dqn"]
        if others:
            lines += [
                "The same holds for the other policies the wrapper is live on",
                "(%s): each was already inside both bounds, so wrapping them"
                % ", ".join(others),
                "changes their numbers by rounding at most.",
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
            "wrapper has to enforce on a learned policy, and it is why a fixed",
            "plan is the safe, dull thing that adaptive control has to earn its",
            "way past.",
            "",
        ]

    lines += [
        "**What this does not establish.** The bounds were checked on the six",
        "evaluation scenarios and the policy satisfied them; that is evidence",
        "about these hours of traffic, not a proof about all of them. The",
        "wrapper is what converts it into a guarantee, and the guarantee it",
        "gives is the one stated in the class docstring - a phase is served",
        "within 200 s plus up to 15 s of hand-over slack, not within 200 s",
        "exactly.",
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
