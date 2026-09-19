# CLEANUP_AUDIT.md — Phase 1, read-only

Produced 2026-09-20 against commit `b693aee`. **Nothing has been changed.**
Every finding below is risk-tiered; only LOW items are proposed for automatic
removal in Phase 2.

Headline: the repository is in good order. One 230 MB regenerable directory
dominates the disk, the caches are back, and there is **no dead code, no
orphaned scratch file, and no leftover synthetic data**. The real findings are
documentation drift and two specific test-coverage gaps.

---

## 1. File inventory

Total excluding `.venv` and `.git`: **270 MB**, of which one directory is 230 MB.

| Path | Size | Notes |
|---|---|---|
| `results/corridor/tripinfo/` | **230 MB** (230 files) | Gitignored, regenerable — **LOW**, see §8 |
| `corridor/routes/` | 17 MB | **PROTECTED** — corridor demand |
| `networks/routes/` | 13 MB | **PROTECTED** — irreplaceable, §5 rule 1 |
| `results/logs/` | 4.0 MB | 78 traces + CSVs — **PROTECTED** |
| `results/corridor/logs/` | 2.0 MB | 28 traces + CSVs + `runs/` — **PROTECTED** |
| `tests/` | 1.2 MB | 23 modules, 513 tests |
| `agent/` | 856 KB | 6 models — **PROTECTED** |
| `corridor/networks/` | 433 KB | generated nets incl. `cologne3` actuated |
| `results/plots/`, `results/corridor/plots/` | 676 KB + 316 KB | **PROTECTED** — published figures |
| `.pytest_cache/` | 55 KB | **LOW** |
| `.ruff_cache/` | 19 KB | **LOW** |
| `__pycache__/` × 6 | ~100 KB | **LOW** |

### Scratch, backup and orphan scan — clean

Searched for `*.bak`, `*_old.*`, `*.orig`, `*.tmp`, `*~`, `*.swp`, `*.rej`,
`nohup.out`, stray `.log.N`, `*.egg-info`, `build/`, `dist/`.
**Zero hits.** The only matches were `.pyc` files inside `__pycache__`.

### Leftover synthetic data — confirmed cleaned

Earlier sessions fabricated datasets to exercise the table and figure
generators. Verified gone: no `*probe*`, no `*synthetic*`, no orphaned
`ma2c_probe.pt`, and `git log` shows **zero commits ever touched** any
synthetic CSV. `git status` is clean.

---

## 2. Dead code — none

Checked by AST parse plus cross-reference against all `.py`, `.md` and `.sh`
files, rather than by naming convention.

**No module is dead.** Every module reporting zero imports is either an
entry-point script a human runs directly (`train_dqn.py`, `train_ppo.py`,
`train_mixed.py`, `run_baseline.py`, `train_independent.py`, `train_ma2c.py`,
`make_corridor_plots.py`, `smoke_test.py`) or a test collected by pytest rather
than imported. `corridor/vec_env.py` looked orphaned by name but is imported by
`corridor/train_independent.py`.

**One unused top-level function**, and it is protected:

| Symbol | File | Referenced anywhere? |
|---|---|---|
| `evaluation_route_files()` | `networks/generate_routes.py:295` | No — code, docs, shell all zero |

> **HIGH risk — do not remove without your decision.** `networks/generate_routes.py`
> is one of the sixteen files hashed by `tests/test_corridor_isolation.py`, and
> the ABSOLUTE PROTECTIONS list says not to touch what that test hashes.
> Deleting three lines would fail the isolation test and require editing the
> hash table — the exact "someone tidied a protected file" event that test
> exists to catch. It is a three-line pure function with a docstring, costs
> nothing to keep, and is a plausible companion to `training_route_files()`
> which *is* used. **Recommendation: keep it.**

---

## 3. Unused imports and variables — none

`ruff check .` reports **All checks passed** across the whole tree with the rule
set pinned in `pyproject.toml`. Nothing for Phase 3 to auto-fix.

---

## 4. Duplicated logic — flagged only, no refactor proposed

Measured as identical non-blank, non-comment lines between each
single-intersection module and its corridor counterpart:

| Pair | Lines | Identical |
|---|---|---|
| `eval/paths.py` ↔ `corridor/paths.py` | 45 / 101 | 7 |
| `eval/evaluate.py` ↔ `corridor/evaluate_corridor.py` | 371 / 276 | 35 |
| `eval/make_tables.py` ↔ `corridor/make_corridor_tables.py` | 376 / 404 | 36 |
| `eval/make_plots.py` ↔ `corridor/make_corridor_plots.py` | 459 / 339 | 0 |
| `eval/simulation.py` ↔ `corridor/env.py` | 795 / 335 | 0 |

**MEDIUM — and my recommendation is to leave all of it alone.** The overlap is
shallow (mostly CSV read/write boilerplate and argparse scaffolding), and the
deeper files share nothing. More importantly, extracting a shared layer would
mean *editing `eval/` files to make the corridor reuse them* — which is
precisely what `CORRIDOR_PLAN.md` §7 and PROJECT_CONTEXT §12 forbid, and what
`test_corridor_isolation.py` hashes against. The duplication is the price of an
isolation guarantee that has already proved its worth.

Where sharing was safe, it already happens: `corridor/env.py` **imports**
`DELTA_TIME`, `YELLOW_TIME`, `MIN_GREEN`, `negative_queue_reward`,
`parse_tripinfo` and `quiet_stdout` from `eval/simulation.py` rather than
copying them, so there is exactly one definition of each constant in the repo.

---

## 5. Stale documentation

All findings are in `PROJECT_CONTEXT.md` and `README.md`. **Neither report is
affected** — `REPORT.md` and `CORRIDOR_REPORT.md` are accurate.

| # | Location | Claim | Reality | Risk |
|---|---|---|---|---|
| 5.1 | `README.md` (whole file) | No mention of the corridor extension at all | Half the project is missing from the user-facing entry point | **MEDIUM** |
| 5.2 | `PROJECT_CONTEXT.md:274` | file map says `tests/ 276 tests` | 513 | LOW |
| 5.3 | `PROJECT_CONTEXT.md:409` | `# 276 tests, ~3 s` | 513, ~6 s | LOW |
| 5.4 | `PROJECT_CONTEXT.md` §10 table | lists 11 test modules | 23 exist | LOW |
| 5.5 | `PROJECT_CONTEXT.md:243` | `docs/REMEDIATION_LOG.md (STALE Phase 3 table - see 3.2)` | that log was fixed on 2026-09-19, and §3.2 no longer exists | LOW |
| 5.6 | `PROJECT_CONTEXT.md:452,454` | §11 points at `§3.3` and `§3.1` | §3 was rewritten to a single §3.0; both anchors are dangling | LOW |
| 5.7 | `PROJECT_CONTEXT.md:87` | "Verified-green as of 2026-09-19" | verified 2026-09-20 | LOW |
| 5.8 | `PROJECT_CONTEXT.md` §6 file map | no `corridor/` entry | the corridor is ~18 MB of the tree | LOW |

`docs/PROGRESS.md` and `docs/REMEDIATION_LOG.md` carry old numbers, but both are
explicitly historical records of a moment, and `PROJECT_CONTEXT.md` already
states it overrides them. **Not stale — correct as history.** Leave alone.

`VERIFICATION.md`'s per-module table is complete (23 of 23) because
`test_documented_counts.py` enforces it. `PROJECT_CONTEXT.md`'s equivalent table
is *not* enforced, which is exactly why it drifted — see §6.3.

---

## 6. Test coverage gaps

513 tests across 23 modules; every source module that has behaviour worth
pinning is imported by at least one test, with two exceptions that matter.

### 6.1 `corridor/paths.py` has no assertions on its naming — MEDIUM

`eval/paths.py` exists because of **defect F2**: trace filenames were built in
three places, two disagreed, and because a missing trace is a legitimate state
the mismatch silently dropped rows from the one table that existed to show them.
`tests/test_output_naming.py` pins that fix directly — asserting function
identity and round-tripping a filename.

`corridor/paths.py` is the corridor's equivalent single naming authority, with
strictly more names to get wrong (controllers × scenarios × failure rates ×
corridor sizes). Tests *use* it as a helper but **nothing asserts its
behaviour**. The F2 failure mode is fully available and unguarded.

**Recommendation: add tests in Phase 4** (additive only, permitted by the
prompt).

### 6.2 `agent/training.py` has no test — MEDIUM, recommend leaving

The shared single-intersection training loop, route rotation and logging. No
test imports it. It is, however, frozen single-intersection code that produced
published numbers, is hashed by `test_corridor_isolation.py`, and is exercised
end-to-end by every training run. Adding tests is safe (additive) but low value
this late. **Recommendation: note it, do not act.**

### 6.3 `PROJECT_CONTEXT.md`'s test table is not machine-checked — LOW

`test_documented_counts.py` checks `README.md` and `VERIFICATION.md`. It does
not check `PROJECT_CONTEXT.md`, and that table is the one that drifted (finding
5.4). The project's own stated lesson is "documents are checked by running, not
by reading". **Recommendation: either extend the counts test to cover it, or
stop quoting per-module counts there. I lean to the latter — one enforced table
is better than two, one of which lies.** Flagging for your decision rather than
acting.

### Regression protection for known past defects — all present

| Defect | Guard | Present |
|---|---|---|
| D1 route seeding | `test_route_determinism.py` | yes |
| D4 report provenance | `test_report_matches_data.py`, `test_corridor_report_matches_data.py` | yes |
| D6 silent-zero metric | `test_corridor_metrics.py` | yes |
| F1/F4 safety wrapper | `test_safety_constraints.py` | yes |
| F2/F3 naming, table guards | `test_output_naming.py` | yes (single only — see 6.1) |
| Corridor geometry inversion | `test_corridor_network.py` | yes |
| Heuristic bypassing detectors | `test_detector_dependence.py` | yes |
| MA2C within-step policy leak | `test_ma2c.py` | yes |
| Time-limit bootstrapping | `test_episode_boundary.py` | yes |
| MA2C critic starvation | `test_ma2c.py` (learning test) | yes |
| Sweep interruption | `test_sweep_resume.py` | yes |

---

## 7. Caches and build artifacts

| Item | Size | Gitignored | Risk |
|---|---|---|---|
| `__pycache__/` × 6 | ~100 KB | yes | LOW |
| `.pytest_cache/` | 55 KB | yes | LOW |
| `.ruff_cache/` | 19 KB | yes | LOW |
| `*.egg-info`, `build/`, `dist/` | — | — | none exist |

All three regenerate on the next test or lint run. They are already ignored, so
removing them changes nothing tracked.

---

## 8. The 230 MB question — `results/corridor/tripinfo/`

**LOW risk, and the single biggest item in the repository.**

Same argument that justified deleting the single-intersection equivalent
(PROJECT_CONTEXT §8), verified again for the corridor:

- `run_corridor_episode` writes each tripinfo XML and parses it into metrics
  **within the same episode** (`corridor/env.py`), then never reads it again.
- Tests use `tmp_path` fixtures, never the real directory.
- Every metric it produced is permanently in `corridor_evaluation_raw.csv`,
  `corridor_robustness_raw.csv` and `corridor_transfer_raw.csv`.
- `.gitignore:28` already declares it regenerable.
- Re-running the evaluation recreates it; no verification path is lost.

Removing it takes the tree from **270 MB to ~40 MB**.

---

## 9. Proposed Phase 2 actions (LOW only)

| # | Action | Reclaims |
|---|---|---|
| 1 | `rm -f results/corridor/tripinfo/*.xml` | 230 MB |
| 2 | `rm -rf .pytest_cache .ruff_cache` | 74 KB |
| 3 | remove the 6 `__pycache__/` directories outside `.venv` | ~100 KB |

Nothing else. No source file, no test, no doc, no protected artifact.

## 10. Held for your decision

| Item | Tier | Ask |
|---|---|---|
| `evaluation_route_files()` unused (§2) | **HIGH** | It is in a hashed, protected file. Recommend keeping. |
| `README.md` never mentions the corridor (§5.1) | MEDIUM | May I add a section? It is additive prose, no data. |
| Duplication between `eval/` and `corridor/` (§4) | MEDIUM | Recommend leaving — refactoring would breach the isolation guarantee. |
| `corridor/paths.py` untested (§6.1) | MEDIUM | Recommend adding tests in Phase 4 — additive, guards a defect that already bit once. |
| `PROJECT_CONTEXT.md` test table unenforced (§6.3) | LOW | Recommend deleting the per-module counts there rather than duplicating an enforced table. |

## 11. Nothing found in these categories

- No editor swap files, backups, or `.orig`/`.rej` conflict remnants.
- No orphaned experiment scripts.
- No leftover synthetic/debug data.
- No unused imports or variables (`ruff` clean).
- No dead module.
- No file in `results/` or `corridor/` that is unreferenced by code or docs.
- No discrepancy in either report's data.
