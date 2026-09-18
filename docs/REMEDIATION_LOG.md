# Remediation log / resume checkpoint

**Purpose.** This file is the resume point for the remediation and follow-up
pass. If work is interrupted, read this first: it records what is finished,
what is running, and the exact next step. `VERIFICATION.md` is the *evidence*
document for the grader; this is the *working* document.

**Started:** 2026-09-07, against the tree tagged `pre-remediation-audited`.
**Last updated:** 2026-09-19, after the Phase 3 chain completed and its addenda
were published into `REPORT.md`.

---

## Ground rules being followed

- Do **not** regenerate the committed route files. They are the demand every
  published number was measured on and they predate the seeding fix, so they
  cannot be reproduced by the current generator. `generate_routes.py` now
  refuses to overwrite them without `--force`.
- Do **not** retrain or overwrite `agent/models/dqn.zip` or `ppo.zip`. Phase 3
  models are saved under new names.
- Do not touch the reward, the observation space, the environment constants, or
  the experimental design. The audit validated those.
- Addenda are **appended** to `REPORT.md`. The original narrative and its
  numbers are not rewritten; where an addendum qualifies a conclusion it says
  so in the addendum.

---

## Phase 1 — critical path: COMPLETE

| # | Task | Defects | State |
|---|---|---|---|
| 0 | `git init`, commit, tag `pre-remediation-audited` | D2 | done |
| 1 | Deterministic route seeding + `--force` guard | D1 | done, verified |
| 2 | Generated tables spliced into `REPORT.md` | D4 | done, idempotent |
| 3 | Small-defect sweep | D5–D9 | done |
| 4 | Test suite in `tests/` | D3 | done, all pass |

Details of each fix and its evidence are in `VERIFICATION.md`.

## Phase 2 — verification: COMPLETE

| Check | State |
|---|---|
| Smoke test 5/5 after changes | pass |
| Both models load and predict | pass (dqn→2, ppo→1 on a fixed probe observation) |
| Route determinism across 3 fresh processes | pass (identical sha256) |
| Overwrite guard refuses, files untouched | pass (exit 1, `git status` clean) |
| Test suite passes | pass |
| Mutation check — tests actually catch a break | pass (5/5 breaks caught, on an isolated copy) |
| Full 150-episode sweep re-run and reconciled | pass — 150/150 bit-for-bit identical |
| `REPORT.md` numbers trace to the CSVs | pass — enforced by `test_report_matches_data.py` |
| `VERIFICATION.md` written | pass |

## Phase 3 — the five follow-up experiments: COMPLETE

| # | Item | State | Artifacts |
|---|---|---|---|
| 1 | 2 extra DQN training seeds | **DONE** — both trained and evaluated; addendum B in `REPORT.md` | `dqn_seed1.zip`, `dqn_seed2.zip` |
| 2 | Mixed-demand training | **DONE** — addendum C in `REPORT.md` | `dqn_mixed.zip` |
| 3 | Paired significance test | **DONE** — addendum A in `REPORT.md` | `eval/significance.py` |
| 4 | PPO 400 episodes | **DONE** — addendum D in `REPORT.md` | `ppo_long.zip` |
| 5 | Max-green / starvation cap | **DONE** — wrapper, tests and evaluation; addendum E in `REPORT.md` | `SafetyConstrainedController` |

### The chain that was run

`results/logs/runs/chain.sh`, launched 2026-09-08 16:59, five sequential steps.
It **completed all five steps at 2026-09-08 20:09:55**; `chain_status.txt` ends
with `CHAIN COMPLETE`. Each step's own output is in its own log in the same
directory.

    1. python agent/train_dqn.py --seed 2 --tag seed2                (~55 min)
    2. python agent/train_mixed.py                                   (~55 min)
    3. python agent/train_ppo.py --episodes 400 --tag long           (~60 min)
    4. python eval/evaluate.py --controllers dqn_seed2 dqn_mixed ppo_long --append
    5. python eval/evaluate.py --controllers dqn+safe ppo+safe
           greedy-queue+safe fixed-time-tuned+safe --append

**If the chain had died part-way**, `chain_status.txt` names the last step that
finished. A training run that was interrupted leaves no model (the save happens
at the end, or on KeyboardInterrupt), so restart that step from the top; its
curve CSV holds whatever episodes had been flushed and is overwritten cleanly
on restart. Steps 4 and 5 are `--append` and are safe to re-run: an episode is
keyed by `(controller, scenario, seed)` and a re-run replaces its row. This
guidance is retained because it still applies to any future re-run.

### After the chain finished: DONE 2026-09-19

    python eval/followups.py --write     # addenda B-E      - done
    python eval/make_tables.py --write   # left REPORT.md byte-identical - done
    python eval/make_plots.py            # figures byte-identical - done
    python -m pytest                     # 276 passed - done

`VERIFICATION.md` was updated earlier in the same pass (test counts are now
self-checking via `tests/test_documented_counts.py`, and the F-series defects
are written up in its section 5). The addenda publication and the Phase 3
artifacts are committed.

**Not attempted, deliberately:** the multi-intersection corridor. Both the
original brief and the audit deprioritise it, and starting it would trade a
finished single-junction result for an unfinished network one.

---

## Defects found *during* Phase 3, and how they were fixed

These are not audit findings; they are bugs in the Phase 3 code itself, found
before any of it produced a published number. Recorded here and in
`VERIFICATION.md` because three of the four would have failed silently.

**F4 — the safety wrapper enforced nothing.** The worst of them.
`SafetyConstrainedController` reset a phase's starvation clock when the wrapped
controller *asked* for that phase, but sumo-rl refuses a phase change while the
current green is younger than `yellow + min_green`, so a request is often not a
service. Measured on one `balanced` hour: the phase asked for was not the phase
then green on 573 of 720 decisions; the wrapper's own worst clock read 175 s
against its 200 s bound and fired zero overrides, while the junction left a
movement unserved for 245 s. `greedy-queue+safe` produced a trace bit-identical
to `greedy-queue` — which is what a wrapper that never fires looks like, and
had been read as the policy already being safe. Fixed by advancing the clocks
from `signal.green_phase`, the phase actually showing green. The unconstrained
policies turn out to violate the bound on every scenario checked (245-340 s);
wrapped they come to 210-215 s, at a cost under half a second of mean waiting
time. Addendum E's conclusion is now derived from the traces instead of
asserted in prose.

**F1 — the safety wrapper could not wrap a fixed-time plan.**
`SafetyConstrainedController.on_episode_start` read `len(signal.green_phases)`.
sumo-rl only builds that list when it is driving the signal; under a fixed-time
plan SUMO runs the program, the attribute is absent, and the wrapper raised
`AttributeError` before the episode took a step. `dqn+safe` therefore worked and
`fixed-time-tuned+safe` could not be evaluated at all. Fixed by reading
`num_green_phases`, which is published in both modes. The stub in
`tests/test_safety_constraints.py` now deliberately omits `green_phases`, so
every test in that module exercises the fixed-time shape.

**F2 — the trace filename was built in three places and they disagreed.**
`eval/evaluate.py` wrote `trace_greedy_queue+safe_ns_heavy.csv`;
`eval/followups.py` looked for `trace_greedy_queue_safe_ns_heavy.csv`. A
missing trace is a legitimate state, so `trace_bounds` returned None and the
row was dropped — addendum E would have been published with the constrained
controllers missing from the one table that exists to show them. Fixed by
`eval/paths.py`, a single naming function all three scripts import;
`tests/test_output_naming.py` asserts identity of the function object and
round-trips a trace from writer to reader.

**F3 — a generated table would have crashed on the new controller names.**
`insertion_backlog_table` iterated every controller in `evaluation_raw.csv` and
looked each up in `LABELS`, which only knows the report's five. Adding
`dqn_seed2` or `dqn+safe` with a non-zero backlog would have raised `KeyError`
at the last step of the pipeline, after the training had already run. The table
is now restricted to the five it describes, like every other generated table.

---

## Design decisions taken for Phase 3, so a resume does not re-litigate them

**Item 1 (extra seeds).** No code change needed to train: `train_dqn.py`
already takes `--seed` and `--tag`, which write `dqn_seed<N>.zip` and a
matching curve. Evaluating them needed a small additive change to
`eval/evaluate.py`, whose `load_policy`/`build_factory` recognised only the
literal names `dqn` and `ppo`. The extension infers the algorithm from the
name's prefix and leaves `CONTROLLER_ORDER` — the default sweep — alone, so the
validated default path is unchanged.

**Item 3 (significance).** Implemented without SciPy: `eval/significance.py`
carries its own regularised incomplete beta (for Student's t) and an exact
Wilcoxon by enumerating all 2^n sign assignments. Both are covered by
`tests/test_significance.py` against published critical values and hand-counted
cases, because a hand-rolled p-value that is quietly wrong would be worse than
no p-value at all. Key finding already established: at n=5 a two-sided Wilcoxon
has a floor of 2/32 = 0.0625 and *cannot* reach 0.05, so the paired t-test is
the primary result and the floor is stated beside the number.

**Item 5 (max green / starvation).** Built as a *wrapper controller*, not as a
change to the environment or to any trained policy: `SafetyConstrainedController`
wraps another controller, passes its action through unchanged in the normal
case, and overrides it when the current phase has held green past the cap or
when some phase has gone unserved past a starvation bound. This keeps the
constraint measurable — the same policy can be run with and without it, so the
cost of the safety layer is a number rather than an assertion — and means no
model has to be retrained to gain it.

**Which controllers get a `+safe` run, and why one of them is a null.** The
sweep covers `dqn`, `ppo`, `greedy-queue` and `fixed-time-tuned`. The first
three drive the signal themselves, so the wrapper is live on them. The
fixed-time plan is run by SUMO's own program and its `act()` is never
consulted, so its `+safe` run is bit-identical *by construction*. It is still
run and still reported, but addendum E says plainly that the zero in that row
is arithmetic rather than evidence, and establishes the plan's compliance from
its trace instead. Presenting an inert wrapper as a passing safety check would
be the kind of quiet overstatement this whole pass exists to remove.

---

## If you are resuming

1. `git log --oneline` — see which commits landed.
2. `cat results/logs/runs/chain_status.txt` — see how far the chain got.
3. `python -m pytest` — should be all green before anything else.
4. Find the first row above whose state is not "done" and continue from it.
5. Long training runs write their curve CSV every 10 episodes, so a partial run
   leaves usable evidence in `results/logs/training_curve_*.csv` even though it
   leaves no model.
