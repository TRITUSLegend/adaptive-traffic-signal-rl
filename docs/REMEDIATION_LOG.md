# Remediation log / resume checkpoint

**Purpose.** This file is the resume point for the remediation pass. If work is
interrupted, read this first: it records what is finished, what is running, and
the exact next step. `VERIFICATION.md` is the *evidence* document for the
grader; this is the *working* document.

**Started:** 2026-09-07, against the tree tagged `pre-remediation-audited`.

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

---

## Phase 1 — critical path: COMPLETE

| # | Task | Defects | State |
|---|---|---|---|
| 0 | `git init`, commit, tag `pre-remediation-audited` | D2 | done |
| 1 | Deterministic route seeding + `--force` guard | D1 | done, verified |
| 2 | Generated tables spliced into `REPORT.md` | D4 | done, idempotent |
| 3 | Small-defect sweep | D5–D9 | done |
| 4 | Test suite, 128 tests in `tests/` | D3 | done, all pass |

Details of each fix and its evidence are in `VERIFICATION.md`.

## Phase 2 — verification: IN PROGRESS

| Check | State |
|---|---|
| Smoke test 5/5 after changes | pass |
| Both models load and predict | pass (dqn→2, ppo→1 on a fixed probe observation) |
| Route determinism across 3 fresh processes | pass (identical sha256) |
| Overwrite guard refuses, files untouched | pass (exit 1, `git status` clean) |
| Test suite passes | pass (128) |
| Mutation check — tests actually catch a break | pass (5/5 breaks caught, on an isolated copy) |
| Full 150-episode sweep re-run and reconciled | running |
| `REPORT.md` numbers trace to the CSVs | pending |
| `VERIFICATION.md` written | pending |

Test count is now 155 (`tests/test_significance.py` added with Phase 3 item 3).

## Phase 3 — all five items, greenlit to run in order

The user asked for every item, in the order below, however long it takes, with
a checkpoint written here after each so the work can be resumed.

| # | Item | Est. | State | Artifacts |
|---|---|---|---|---|
| 1 | 2 extra DQN training seeds | ~1.8 h | not started | `dqn_seed<N>.zip`, curves |
| 2 | Mixed-demand training | ~1.5 h | not started | `dqn_mixed.zip` |
| 3 | Paired Wilcoxon significance test | ~0.5 h | not started | `eval/significance.py` |
| 4 | PPO 400 episodes | ~1.2 h | not started | `ppo_long.zip` |
| 5 | Max-green / starvation cap | ~1.5 h | not started | controller + eval |

**Not attempted, deliberately:** the multi-intersection corridor. Both the
original brief and the audit deprioritise it, and starting it would trade a
finished single-junction result for an unfinished network one.

### Design decisions taken for Phase 3, so a resume does not re-litigate them

**Item 1 (extra seeds).** No code change needed to train: `train_dqn.py`
already takes `--seed` and `--tag`, which write `dqn_seed<N>.zip` and a
matching curve. Evaluating them does need a small additive change to
`eval/evaluate.py`, whose `load_policy`/`build_factory` currently recognise
only the literal names `dqn` and `ppo`. The extension infers the algorithm from
the name's prefix and leaves `CONTROLLER_ORDER` - the default sweep - alone, so
the validated default path is unchanged.

**Item 3 (significance).** Implemented without SciPy: `eval/significance.py`
carries its own regularised incomplete beta (for Student's t) and an exact
Wilcoxon by enumerating all 2^n sign assignments. Both are covered by
`tests/test_significance.py` against published critical values and hand-counted
cases, because a hand-rolled p-value that is quietly wrong would be worse than
no p-value at all. Key finding already established: at n=5 a two-sided Wilcoxon
has a floor of 2/32 = 0.0625 and *cannot* reach 0.05, so the paired t-test is
the primary result and the floor is stated beside the number.

**Item 5 (max green / starvation).** To be built as a *wrapper controller*, not
as a change to the environment or to any trained policy: `MaxGreenController`
wraps another controller, passes its action through unchanged in the normal
case, and overrides it when the current phase has held green past the cap or
when some phase has gone unserved past a starvation bound. This keeps the
constraint measurable - the same policy can be run with and without it, so the
cost of the safety layer is a number rather than an assertion - and means no
model has to be retrained to gain it.

Each completed item is appended to `REPORT.md` as a clearly marked addendum.
The original report's narrative and numbers are not rewritten.

---

## If you are resuming

1. `git log --oneline` — see which commits landed.
2. `python -m pytest` — should be all green before anything else.
3. Find the first row above whose state is not "done" and continue from it.
4. Long training runs write their curve CSV incrementally, so a partial run
   leaves usable evidence in `results/logs/training_curve_*.csv`.
