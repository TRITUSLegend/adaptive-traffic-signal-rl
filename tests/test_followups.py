"""
Tests for the addendum generators in eval/followups.py.

WHY THESE TESTS EXIST
---------------------
These four functions write prose that goes straight into REPORT.md, and the
prose states conclusions: "a genuine reduction", "the deficit is gone", "the
constraint is essentially free". Which sentence gets written is decided by
arithmetic on the evaluation data, so a sign error or a mis-picked branch does
not produce a crash - it produces a report that confidently says the opposite
of what was measured. That is the failure these tests exist to catch.

They also check something duller and just as easy to get wrong: that the
markdown tables are well formed. A header with five columns above a divider
with four renders as a mangled table in every markdown viewer, and nothing in
the pipeline would have noticed - that exact mismatch was live in the PPO
addendum until it was caught here.

Everything is built from synthetic evaluation rows, so the assertions are
about the generator's logic and not about how any particular training run
happened to turn out.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.followups as followups  # noqa: E402
import eval.paths as paths  # noqa: E402

SCENARIOS = ["balanced", "ns_heavy", "peak", "ew_heavy", "surge", "light"]
SEEDS = [100, 101, 102, 103, 104]


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def rows_for(controller: str, waits: dict) -> list:
    """
    Five per-seed rows per scenario, spread symmetrically about the mean.

    The jitter is deterministic and sums to zero, so `wait()` returns exactly
    the number asked for and a test can assert on it without slack.
    """
    out = []
    for scenario, mean in waits.items():
        for offset, seed in zip((-0.2, -0.1, 0.0, 0.1, 0.2), SEEDS):
            out.append({
                "controller": controller,
                "scenario": scenario,
                "seed": str(seed),
                "mean_waiting_time_s": str(mean + offset),
            })
    return out


def flat(value: float) -> dict:
    return {s: value for s in SCENARIOS}


def with_scenarios(base: float, **overrides) -> dict:
    waits = flat(base)
    waits.update(overrides)
    return waits


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """Point both LOGS_DIR names at a scratch directory."""
    monkeypatch.setattr(followups, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(paths, "LOGS_DIR", tmp_path)
    return tmp_path


def write_curve(logs_dir: Path, name: str, episodes: int, final_queue: float):
    path = logs_dir / ("training_curve_%s.csv" % name)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["episode", "mean_queue_veh"])
        writer.writeheader()
        for episode in range(1, episodes + 1):
            writer.writerow({"episode": episode, "mean_queue_veh": final_queue})


STATES = ("GGrrrrGGrrrr", "rrGrrrrrGrrr", "rrrGGrrrrGGr", "rrrrrGrrrrrG")


def write_trace(logs_dir: Path, controller: str, scenario: str,
                holds=(30, 30, 30, 30), sequence=None):
    """
    A trace that shows each phase in turn for the given number of seconds.

    The first hold sets the worst unserved gap, because every later phase has
    been waiting since t=0 - which is exactly the quantity addendum E reports,
    so a test can dial a compliant or a violating trace by changing one number.

    `sequence`, as (phase index, seconds) pairs, gives full control. It is
    needed for the case the two bounds pull apart: starving one phase while
    every green stays short takes a long cycle of the other three, which no
    single pass over four holds can express.
    """
    path = logs_dir / paths.trace_filename(controller, scenario)
    steps = sequence if sequence is not None else list(enumerate(holds))
    time = 0.0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sim_time", "signal_state"])
        writer.writeheader()
        for phase, hold in steps:
            for _ in range(int(hold) // 5):
                time += 5.0
                writer.writerow({"sim_time": time,
                                 "signal_state": STATES[phase]})


# ---------------------------------------------------------------------------
# Markdown tables must be well formed
# ---------------------------------------------------------------------------
def tables_in(text: str) -> list:
    """Every run of consecutive table lines, as lists of rows of cells."""
    tables, current = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            current.append([c.strip() for c in stripped.strip("|").split("|")])
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def assert_tables_are_well_formed(text: str, least: int = 1):
    tables = tables_in(text)
    assert len(tables) >= least, "expected at least %d table(s)" % least
    for table in tables:
        header, divider = table[0], table[1]
        assert all(set(cell) <= set("-: ") and cell for cell in divider), (
            "second row of a table is not a divider: %r" % (divider,))
        width = len(header)
        for index, row in enumerate(table):
            assert len(row) == width, (
                "row %d has %d cells, header has %d:\n  header %r\n  row    %r"
                % (index, len(row), width, header, row))


# ---------------------------------------------------------------------------
# Addendum B - seed variance
# ---------------------------------------------------------------------------
def test_one_training_seed_is_not_enough_to_say_anything(logs):
    raw = rows_for("dqn", flat(20.0))
    assert followups.seed_variance(raw) is None


def test_a_model_missing_a_scenario_does_not_count_as_evaluated(logs):
    """
    `present` requires every scenario. A half-finished evaluation must not be
    averaged in as though it were a third agent.
    """
    raw = rows_for("dqn", flat(20.0))
    raw += rows_for("dqn_seed1", {"balanced": 19.0, "ns_heavy": 20.0})
    assert followups.seed_variance(raw) is None


def test_three_seeds_produce_the_per_seed_table(logs):
    for name, episodes in (("dqn", 170), ("dqn_seed1", 170), ("dqn_seed2", 170)):
        write_curve(logs, name, episodes, 11.0)
    raw = (rows_for("dqn", flat(20.0))
           + rows_for("dqn_seed1", flat(20.4))
           + rows_for("dqn_seed2", flat(19.6)))

    text = followups.seed_variance(raw)

    assert "Addendum B" in text
    assert "seed 42" in text and "seed 1" in text and "seed 2" in text
    assert_tables_are_well_formed(text, least=2)


def test_the_spread_reported_is_the_spread_between_agents(logs):
    """
    The number the addendum leads with is the standard deviation ACROSS the
    trained agents on the worst scenario. Pinning it stops the sentence
    quietly starting to describe something else - seed-to-seed traffic
    variation, say, which is a different and much smaller quantity.
    """
    for name in ("dqn", "dqn_seed1", "dqn_seed2"):
        write_curve(logs, name, 170, 11.0)
    raw = (rows_for("dqn", with_scenarios(20.0, ew_heavy=20.0))
           + rows_for("dqn_seed1", with_scenarios(20.0, ew_heavy=24.0))
           + rows_for("dqn_seed2", with_scenarios(20.0, ew_heavy=22.0)))

    text = followups.seed_variance(raw)

    # sd of (20, 24, 22) = 2.00, and every other scenario has sd 0.
    assert "largest on `ew_heavy` at 2.00 s" in text
    assert "a range of 4.0 s" in text


# ---------------------------------------------------------------------------
# Addendum C - mixed demand
# ---------------------------------------------------------------------------
def mixed_raw(base_penalty_to: float, mixed_penalty_to: float) -> list:
    """Both agents at 20 s on the trained scenario, differing on the mirror."""
    raw = rows_for("dqn", with_scenarios(20.0, ns_heavy=20.0,
                                         ew_heavy=base_penalty_to))
    raw += rows_for("dqn_mixed", with_scenarios(20.0, ns_heavy=20.0,
                                                ew_heavy=mixed_penalty_to))
    raw += rows_for("greedy-queue", with_scenarios(22.0))
    return raw


def test_mixed_demand_is_skipped_until_the_agent_is_evaluated(logs):
    assert followups.mixed_demand(rows_for("dqn", flat(20.0))) is None


def test_a_smaller_directional_penalty_is_called_a_reduction(logs):
    text = followups.mixed_demand(mixed_raw(25.0, 22.0))
    assert "+25.0%" in text and "+10.0%" in text
    assert "a genuine reduction" in text
    assert "no improvement" not in text
    assert_tables_are_well_formed(text, least=2)


def test_a_larger_directional_penalty_is_reported_as_a_negative_result(logs):
    """
    The branch that matters most. If mixed training did not help, the addendum
    has to say so - this is the sentence a generator would be most tempted to
    get wrong.
    """
    text = followups.mixed_demand(mixed_raw(25.0, 28.0))
    assert "no improvement" in text
    assert "a genuine reduction" not in text
    assert "negative result" in text


def test_the_held_out_scenarios_are_labelled_as_held_out(logs):
    text = followups.mixed_demand(mixed_raw(25.0, 22.0))
    for scenario in ("ew_heavy", "surge", "light"):
        line = [ln for ln in text.splitlines()
                if ln.startswith("| `%s`" % scenario)]
        assert line and "held out" in line[0], scenario
    trained = [ln for ln in text.splitlines() if ln.startswith("| `ns_heavy`")]
    assert trained and "trained on" in trained[0]


# ---------------------------------------------------------------------------
# Addendum D - the longer PPO run
# ---------------------------------------------------------------------------
def ppo_raw(long_peak: float, long_surge: float) -> list:
    raw = rows_for("ppo", with_scenarios(20.0, peak=52.0, surge=47.0))
    raw += rows_for("ppo_long", with_scenarios(20.0, peak=long_peak,
                                               surge=long_surge))
    raw += rows_for("dqn", with_scenarios(20.0, peak=40.0, surge=38.0))
    return raw


@pytest.fixture
def ppo_curves(logs):
    write_curve(logs, "ppo", 170, 12.0)
    write_curve(logs, "ppo_long", 400, 11.0)
    write_curve(logs, "dqn", 170, 11.5)
    return logs


def test_ppo_long_is_skipped_until_it_is_evaluated(logs):
    assert followups.ppo_long(rows_for("ppo", flat(20.0))) is None


def test_a_gap_that_mostly_closed_supports_the_original_reading(ppo_curves):
    text = followups.ppo_long(ppo_raw(long_peak=42.0, long_surge=39.0))
    assert "400 episodes against" in text and "the original 170" in text
    assert "supports section 5.3's reading" in text
    assert "only partly supports" not in text
    assert_tables_are_well_formed(text, least=2)


def test_a_gap_that_barely_moved_is_reported_as_only_partial(ppo_curves):
    text = followups.ppo_long(ppo_raw(long_peak=51.0, long_surge=46.0))
    assert "only partly supports" in text
    assert "supports section 5.3's reading" not in text


def test_an_overtake_is_described_as_an_overtake(ppo_curves):
    """
    If the longer run passes the DQN the gap is negative, and the percentage
    sentence would read '-Nan% of it closed' or worse. It must say the deficit
    is gone, and must still count as the deficit having closed.
    """
    text = followups.ppo_long(ppo_raw(long_peak=37.0, long_surge=35.0))
    assert "the deficit is gone" in text
    assert "supports section 5.3's reading" in text
    assert "of it closed" not in text, "the percentage branch ran on a negative gap"


# ---------------------------------------------------------------------------
# Addendum E - the safety bounds
# ---------------------------------------------------------------------------
def safety_raw(cost: float = 0.0) -> list:
    raw = []
    for base in followups.SAFETY_CONTROLLERS:
        raw += rows_for(base, flat(20.0))
        raw += rows_for(base + "+safe", flat(20.0 + cost))
    return raw


def test_safety_is_skipped_until_a_wrapped_run_exists(logs):
    assert followups.safety(rows_for("dqn", flat(20.0))) is None


def write_safety_traces(logs, unwrapped_first_hold, wrapped_first_hold):
    """Traces for every controller in the sweep, with a chosen worst gap."""
    for controller in followups.SAFETY_CONTROLLERS:
        write_trace(logs, controller, followups.TRACE_SCENARIO,
                    holds=(unwrapped_first_hold, 30, 30, 30))
        write_trace(logs, controller + "+safe", followups.TRACE_SCENARIO,
                    holds=(wrapped_first_hold, 30, 30, 30))


def test_a_free_constraint_is_reported_as_free(logs):
    write_safety_traces(logs, 30, 30)

    text = followups.safety(safety_raw(cost=0.0))

    assert "close to free" in text
    assert_tables_are_well_formed(text, least=2)


def test_a_bound_that_was_already_met_is_not_called_a_violation(logs):
    write_safety_traces(logs, 30, 30)
    text = followups.safety(safety_raw(cost=0.0))
    assert "Both bounds were already satisfied" in text
    assert "not already satisfied" not in text


def test_a_bound_violation_is_reported_as_a_violation(logs):
    """
    THE CLAIM THAT MUST FOLLOW THE DATA. If the unconstrained policy left a
    phase unserved past the bound, the addendum has to say so - the earlier
    version of this section asserted compliance in prose whatever the traces
    showed, and the traces showed 315 s against a 200 s bound.
    """
    starved = followups.MAX_STARVATION + 120
    write_safety_traces(logs, starved, 30)

    text = followups.safety(safety_raw(cost=0.0))

    _, gap, _ = followups.trace_bounds("dqn", followups.TRACE_SCENARIO)
    assert gap > followups.MAX_STARVATION, "the fixture must actually violate"

    assert "not already satisfied" in text
    assert "Both bounds were already satisfied" not in text
    assert "%.0f s" % gap in text, "the prose quotes the measured gap"
    assert "**%.0f s**" % gap in text, "and the table flags it as over bound"


def test_the_gap_counts_the_wait_before_a_phases_first_green(logs):
    """
    Hand-computed. Phase 0 holds green for 100 s and the other three get 10 s
    each, so the last of them waits 125 s for its first green. A version that
    measured only the intervals BETWEEN services would score that phase 0 - it
    has no earlier service to measure from - which is precisely backwards.
    """
    write_trace(logs, "probe", "ns_heavy", holds=(100, 10, 10, 10))

    longest_green, gap, seen = followups.trace_bounds("probe", "ns_heavy")

    assert seen == 4
    assert longest_green == 100.0
    assert gap == 125.0


def test_a_phase_that_never_runs_shows_up_in_the_count(logs):
    """The gap cannot express total starvation, so the count has to."""
    write_trace(logs, "probe", "ns_heavy", holds=(100, 10, 10, 0))

    _, _, seen = followups.trace_bounds("probe", "ns_heavy")

    assert seen == 3, "a phase that never ran must not be counted as served"


def test_the_phase_count_is_reported_next_to_the_gap(logs):
    """A phase that never runs leaves no gap to measure, so the count is the
    only place total starvation could show up."""
    write_safety_traces(logs, 30, 30)
    text = followups.safety(safety_raw(cost=0.0))
    assert "%d of %d" % (len(STATES), followups.N_GREEN_PHASES) in text


def test_the_bounds_quoted_are_the_bounds_the_wrapper_enforces(logs):
    """A retyped literal here would let the report describe a constraint the
    code does not apply."""
    from eval.simulation import SafetyConstrainedController

    assert followups.MAX_GREEN == SafetyConstrainedController.MAX_GREEN
    assert followups.MAX_STARVATION == SafetyConstrainedController.MAX_STARVATION
    assert followups.STARVATION_CEILING > followups.MAX_STARVATION

    write_safety_traces(logs, 30, 30)
    text = followups.safety(safety_raw(cost=0.0))
    assert "%d s maximum" % followups.MAX_GREEN in text
    assert "%d s starvation bound" % followups.MAX_STARVATION in text


def test_a_costly_constraint_is_reported_as_a_price(logs):
    text = followups.safety(safety_raw(cost=3.0))
    assert "essentially free" not in text
    assert "the price of" in text


def test_the_fixed_time_row_is_marked_as_an_inert_wrapper(logs):
    """
    THE HONESTY CHECK. SUMO runs the fixed-time program, so the wrapper's
    `act()` is never called and its `+safe` run is identical by construction.
    Reporting that zero as evidence of compliance would be reading a tautology
    as a measurement.
    """
    text = followups.safety(safety_raw())
    line = [ln for ln in text.splitlines()
            if ln.startswith("| `fixed-time-tuned`")]
    assert line and "wrapper inert" in line[0]
    assert "arithmetic, not evidence" in text


def test_the_live_wrappers_are_not_marked_inert(logs):
    text = followups.safety(safety_raw())
    for controller in ("dqn", "ppo", "greedy-queue"):
        line = [ln for ln in text.splitlines()
                if ln.startswith("| `%s`" % controller)
                and " s |" in ln]
        assert line, controller
        assert "inert" not in line[0], controller


def test_every_controller_in_the_safety_sweep_is_classified(logs):
    """A new entry in SAFETY_CONTROLLERS must declare whether it is live."""
    for controller in followups.SAFETY_CONTROLLERS:
        assert controller in followups.LIVE_WRAPPER
        assert controller in followups.SAFETY_LABELS


# ---------------------------------------------------------------------------
# All four, together
# ---------------------------------------------------------------------------
def test_the_whole_addendum_set_renders_valid_markdown(logs):
    for name in ("dqn", "dqn_seed1", "dqn_seed2", "dqn_mixed", "ppo",
                 "ppo_long"):
        write_curve(logs, name, 400 if name == "ppo_long" else 170, 11.0)
    write_safety_traces(logs, followups.MAX_STARVATION + 100, 30)

    raw = (rows_for("dqn", with_scenarios(20.0, ns_heavy=20.0, ew_heavy=25.0,
                                          peak=40.0, surge=38.0))
           + rows_for("dqn_seed1", flat(20.4))
           + rows_for("dqn_seed2", flat(19.6))
           + rows_for("dqn_mixed", with_scenarios(20.0, ns_heavy=20.0,
                                                  ew_heavy=22.0))
           + rows_for("greedy-queue", flat(22.0))
           + rows_for("ppo", with_scenarios(20.0, peak=52.0, surge=47.0))
           + rows_for("ppo_long", with_scenarios(20.0, peak=42.0, surge=39.0))
           + rows_for("fixed-time-tuned", flat(30.0)))
    for base in followups.SAFETY_CONTROLLERS:
        raw += rows_for(base + "+safe", flat(20.1))

    built = [builder(raw) for _, builder in followups.BUILDERS]
    assert all(text is not None for text in built), "an addendum was skipped"
    for text in built:
        assert_tables_are_well_formed(text)
        assert text.startswith("## Addendum ")


def test_a_maximum_green_violation_is_reported_even_when_nothing_starves(logs):
    """
    The two bounds fail independently. A policy can serve every phase promptly
    and still hold one of them for a minute, and the section has to say which
    bound was broken rather than reaching for whichever sentence is nearby.
    """
    long_green = followups.MAX_GREEN + 40
    for controller in followups.SAFETY_CONTROLLERS:
        write_trace(logs, controller, followups.TRACE_SCENARIO,
                    holds=(long_green, 10, 10, 10))
        write_trace(logs, controller + "+safe", followups.TRACE_SCENARIO,
                    holds=(30, 10, 10, 10))

    text = followups.safety(safety_raw(cost=0.0))

    assert "maximum green was not" in text
    assert "Both bounds were already satisfied" not in text
    assert "not already satisfied" not in text


def test_the_binding_bound_is_named_when_only_starvation_fails(logs):
    """
    When the greens are short and a phase is still starved - which is what the
    real policies do - the section should say so, because that is the case the
    headline metrics cannot show.
    """
    # Phases 0-2 cycle briskly for six minutes; phase 3 is served once at the
    # end. Nothing is held long, and one movement is starved - the exact shape
    # the real policies turned out to have.
    starving = [(i % 3, 30) for i in range(12)] + [(3, 30)]
    for controller in followups.SAFETY_CONTROLLERS:
        write_trace(logs, controller, followups.TRACE_SCENARIO,
                    sequence=starving)
        write_trace(logs, controller + "+safe", followups.TRACE_SCENARIO,
                    holds=(30, 30, 30, 30))

    longest_green, gap, _ = followups.trace_bounds("dqn",
                                                   followups.TRACE_SCENARIO)
    assert longest_green <= followups.MAX_GREEN, "no green may be long here"
    assert gap > followups.MAX_STARVATION, "but one phase must be starved"

    text = followups.safety(safety_raw(cost=0.0))

    assert "starvation bound that binds" in text
    assert "maximum green was not" not in text


def test_a_violation_that_is_cheap_to_fix_says_the_headline_survives(logs):
    """
    A reader who has just been told the agent starved a movement will ask
    whether that is how it won. The section has to answer, and only from the
    cost it measured - which is why the sentence lives inside the branch where
    both a violation and a small cost are established.
    """
    write_safety_traces(logs, followups.MAX_STARVATION + 120, 30)

    text = followups.safety(safety_raw(cost=0.0))

    assert "won" in text and "by starving a movement" in text
    assert "It was not." in text


def test_a_clean_run_does_not_claim_the_headline_was_questioned(logs):
    """No violation, no question to answer - the paragraph must not appear."""
    write_safety_traces(logs, 30, 30)

    text = followups.safety(safety_raw(cost=0.0))

    assert "by starving a movement" not in text
