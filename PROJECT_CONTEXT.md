# PROJECT_CONTEXT.md — complete state of this project

**Read this file first.** It is the single orientation document for anyone —
human or AI — picking this project up cold. It states what the project is, what
is finished, what is not, and the rules that must not be broken. Everything here
was verified against the working tree on **2026-09-18**, not copied from older
notes.

Where this file disagrees with `docs/PROGRESS.md` or `docs/REMEDIATION_LOG.md`,
**this file is correct** — those two are historical records that were written
mid-flight and have drifted.

---

## 1. What the project is

A deep-reinforcement-learning agent that controls the traffic lights at a
simulated four-way intersection in [SUMO](https://eclipse.dev/sumo/), plus a
measurement harness that compares it fairly against conventional signal control.

- **Domain:** adaptive traffic signal control, single isolated junction.
- **Agent:** DQN (primary) and PPO (cross-check), via `stable-baselines3` on a
  `sumo-rl` environment.
- **Decision loop:** every 5 simulated seconds the agent observes queues and
  densities and chooses which of 4 green phases to run next.
- **Objective:** it is given only `-(total queued vehicles)/8` as reward. No
  traffic engineering is encoded. It learns the rest over ~170 simulated hours.
- **Episode:** 3600 simulated seconds (one hour).
- **The real question the project answers:** not "can RL beat a bad fixed
  timer" (anything can), but how it compares against a *properly retimed*
  signal (Webster's method) and against a simple "serve the longest queue"
  heuristic.

### Headline result (mean waiting time per vehicle, seconds, 5 seeds each)

| Controller | balanced | ns_heavy* | peak | ew_heavy | surge | light |
|---|---|---|---|---|---|---|
| Fixed-time (untuned) | 29.6 | 52.6 | 72.1 | 46.5 | 75.5 | 28.7 |
| Fixed-time (Webster-tuned) | 30.5 | 31.8 | 46.5 | 32.7 | 45.3 | 23.6 |
| Greedy-queue heuristic | 18.5 | 22.3 | 52.7 | 22.2 | 50.8 | 13.4 |
| **DQN (learned)** | **19.7** | **20.6** | **39.4** | **24.9** | **37.6** | **14.0** |
| PPO (learned) | 19.2 | 19.8 | 51.7 | 25.1 | 46.6 | 15.0 |

\* `ns_heavy` is the only scenario the DQN trained on. All others are held out.

**DQN vs untuned fixed-time: −33% to −61% waiting time, with equal or slightly
higher throughput.**

### Three honesty caveats the project deliberately foregrounds

1. **The greedy heuristic is the real competition.** It beats the DQN on 3 of 6
   scenarios. The DQN's genuine edge is under **saturation** (`peak`, `surge`),
   where it is ~25% better. The project reports this rather than claiming
   "deep RL wins".
2. **The fixed-time baseline is flattered by an artefact.** Untuned fixed-time
   was so congested it failed to insert ~227 veh/h under `surge` and ~95 under
   `peak`. Vehicles never admitted accrue no waiting time, so the baseline's
   number is better than reality and the DQN's margin is an *underestimate*.
   DQN and PPO dropped none.
3. **The agent carries a directional bias from its training scenario:** 20.6 s
   on `ns_heavy` vs 24.9 s on the mirrored `ew_heavy` (+21%). The greedy
   heuristic is unaffected. Addendum C exists to test whether mixed-demand
   training removes this.

---

## 2. Completion status — the one-paragraph version

**The science and all compute are COMPLETE. One cheap publication step is not.**

Every model is trained, every controller is evaluated, every analysis script is
written, committed and tested. But the scripts that splice results into
`REPORT.md` were last run *before* the final training chain delivered its data,
so **three finished experiments (Addenda C, D, E) exist only as empty
placeholders in the report**, and Addendum B is missing its third seed. Closing
this costs about five minutes and zero retraining.

| Phase | Scope | State |
|---|---|---|
| Original build | 7 required deliverables + stretch goals 8, 10 | **complete, committed** |
| External audit remediation | defects D1–D9 | **complete, committed** |
| Phase 2 verification | mutation test, 150-episode bit-for-bit reproduction | **complete, committed** |
| Phase 3 experiments (A–E) | compute: training + evaluation | **complete** (chain finished 2026-09-08 20:09) |
| Phase 3 publication | splice addenda into `REPORT.md` | **NOT DONE — see §3** |
| Version control of Phase 3 outputs | models, traces, CSVs | **NOT DONE — all untracked** |
| Stretch goal 9 (multi-intersection corridor) | — | **deliberately not attempted** |

### Verified-green as of 2026-09-18

- `python -m pytest` → **276 passed**
- `python -m ruff check .` → **All checks passed**
- `results/logs/evaluation_raw.csv` → **390 rows** = 13 controllers × 6 scenarios × 5 seeds, no gaps
- All **6 trained models** present in `agent/models/`
- `results/logs/runs/chain_status.txt` → `CHAIN COMPLETE`

---

## 3. WHAT REMAINS TO BE DONE

### 3.1 Publish the finished addenda (CRITICAL — ~5 minutes, no compute)

This is the only thing standing between the project and "done".

**The problem.** The training chain wrote its data at `20:09` on 2026-09-08.
`followups.py`, `make_tables.py` and `make_plots.py` were last run between
`17:11` and `17:45` — *before* that data existed. So `REPORT.md` currently ends
with three empty generated regions:

```
<!-- BEGIN GENERATED: mixed-demand -->      <!-- END -->   <- Addendum C, EMPTY
<!-- BEGIN GENERATED: ppo-long -->          <!-- END -->   <- Addendum D, EMPTY
<!-- BEGIN GENERATED: safety-constraints --><!-- END -->   <- Addendum E, EMPTY
```

Addendum B is also stale: its table shows only seed 42 and seed 1, though
`dqn_seed2` is trained, evaluated, and already listed in the generator's
`SEED_MODELS`.

**Why this matters beyond tidiness.** `REPORT.md`'s own introduction promises
the reader a table of five addenda A–E, and `README.md` states that all five
"are appended to REPORT.md". Right now only A is complete and B is partial.
Addendum E is the one that documents the **F4 safety-wrapper defect** — the most
significant bug the project caught. The documents currently over-claim.

**Verified ready:** running `eval/followups.py` *without* `--write` on
2026-09-18 built all four addenda (B, C, D, E) cleanly with no "not enough
data" skips. The work is purely a splice.

**The fix, in order:**

```bash
./.venv/Scripts/python.exe eval/followups.py --write    # fills B (w/ seed 2), C, D, E
./.venv/Scripts/python.exe eval/make_tables.py --write  # MUST leave section 4-5 byte-identical
./.venv/Scripts/python.exe eval/make_plots.py           # figures also predate the chain
./.venv/Scripts/python.exe -m pytest                    # must stay 276 green
```

> The `make_tables.py --write` step is not busywork: leaving sections 4 and 5
> byte-identical is the project's designed check that an addendum has not
> quietly moved a number in the original report. If those sections change,
> stop and investigate.

### 3.2 Fix the stale Phase 3 status table (~2 minutes)

`docs/REMEDIATION_LOG.md` still describes the chain as in flight. Its Phase 3
table reads "seed2 **training**", "queued behind seed2", "queued third",
"evaluation queued". **All five items are finished.** The section "The chain
currently running" is also past tense now. Anyone resuming from that file will
be misled into re-running ~3 hours of training that is already done.

### 3.3 Commit the Phase 3 outputs (DECISION REQUIRED)

**Nothing from Phase 3 is under version control.** Currently untracked or
unstaged:

- 3 trained models (`dqn_mixed.zip`, `dqn_seed2.zip`, `ppo_long.zip`) — ~410 KB
- 36 new trace CSVs (`trace_*_{scenario}.csv`)
- 2 training curves (`training_curve_dqn_mixed.csv`, `training_curve_ppo_long.csv`)
- the chain logs in `results/logs/runs/`
- modified `evaluation_raw.csv`, `evaluation_summary.csv`, `training_curve_dqn_seed2.csv`

In a project whose entire discipline is data provenance, having the complete
experimental output sitting outside git is the largest live exposure. **Decide
explicitly** whether models and traces belong in git (they are small — the
models total ~410 KB) or in `.gitignore` with a documented regeneration recipe.
The existing `.gitignore` already reasons carefully about exactly this tradeoff
for route files; follow that precedent.

### 3.4 Repository hygiene (pending — see §8)

Regenerable artifacts still occupy the tree. `results/tripinfo/` alone is
**271 MB across 391 XML files** — 95% of the repository's non-`.venv` weight.
See §8 for the analysis and exact commands.

### 3.5 Explicitly NOT to be done

**The multi-intersection corridor (stretch goal 9 / REPORT §7.3).** Skipped on
purpose and documented consistently in three places. Both the original brief and
the external audit deprioritise it. Starting it would trade a finished
single-junction result for an unfinished network one. **Leave this decision
alone** unless the user explicitly reopens it.

---

## 4. Environment and how to run anything

| Thing | Value |
|---|---|
| Python | 3.13.5 in `.venv/` (the *system* Python is 3.9 and too old) |
| **Invoke as** | `./.venv/Scripts/python.exe <script>` |
| SUMO | 1.27.1, from the `eclipse-sumo` PyPI wheel — **no separate install, no admin rights needed** |
| torch | 2.13.0+cpu (from the PyTorch CPU index, avoids a ~2.5 GB CUDA download) |
| sumo-rl | 1.4.5 |
| stable-baselines3 | 2.7.1 |
| others | gymnasium 1.2.3, numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1 |
| Platform | Windows 11; git-bash and PowerShell both available |
| Git | branch `master`, 10 commits, HEAD `80cf159` |

**`sumo_setup.py` must be imported before any `traci` / `sumolib` / `sumo_rl`
import.** `sumo_rl` raises `ImportError` at import time if `SUMO_HOME` is unset.
Every entry-point script already does this; preserve it in anything new.

### Full pipeline from a clean clone

| # | Command | Does | Time |
|---|---|---|---|
| 1 | `networks/build_network.py` | Builds intersection + 4-phase signal plan | s |
| 2 | `networks/generate_routes.py --mixed` | 60 h of traffic across 6 scenarios | s |
| 3 | `smoke_test.py` | Toolchain check, 5 assertions | ~10 s |
| 4 | `baseline/run_baseline.py` | Fixed-time baselines | ~5 min |
| 5 | `agent/train_dqn.py` | Trains DQN | ~46 min |
| 6 | `agent/train_ppo.py` | Trains PPO | ~26 min |
| 7 | `eval/evaluate.py` | All controllers x all scenarios x 5 seeds | ~26 min |
| 8 | `eval/make_plots.py` | Figures | s |
| 9 | `eval/make_tables.py --write` | Splices tables into `REPORT.md` | s |
| 10 | `python -m pytest` | 276 tests | ~3 s |

Steps 5–7 are the long ones. Every script supports `--help`.

### Phase 3 follow-ups (already run — do not repeat)

```bash
agent/train_dqn.py --seed 1 --tag seed1      # ~50 min  -> dqn_seed1.zip
agent/train_dqn.py --seed 2 --tag seed2      # ~50 min  -> dqn_seed2.zip
agent/train_mixed.py                         # ~50 min  -> dqn_mixed.zip
agent/train_ppo.py --episodes 400 --tag long # ~60 min  -> ppo_long.zip
eval/evaluate.py --controllers dqn_seed1 dqn_seed2 dqn_mixed ppo_long --append
eval/evaluate.py --controllers dqn+safe ppo+safe greedy-queue+safe fixed-time-tuned+safe --append
eval/significance.py --write                 # addendum A
eval/followups.py --write                    # addenda B-E   <- THE PENDING STEP
```

`--append` merges new controllers into `evaluation_raw.csv` instead of replacing
it. An episode is keyed by `(controller, scenario, seed)`, so re-running one
replaces its row rather than duplicating it — append steps are safe to repeat.

A `+safe` suffix wraps any controller in the maximum-green and anti-starvation
bounds, which is how the **cost** of the safety layer is measured rather than
asserted.

---

## 5. NON-NEGOTIABLE GROUND RULES

Breaking any of these destroys published results or invalidates the comparison.

1. **NEVER regenerate the committed route files** in `networks/routes/`. They
   are the demand every published number was measured on, and they **predate the
   seeding fix, so the current generator cannot reproduce them**. They are
   irreplaceable. `generate_routes.py` refuses to overwrite them without
   `--force`; do not pass `--force`.
2. **NEVER retrain or overwrite `agent/models/dqn.zip` or `ppo.zip`.** These
   produced the report's headline numbers. Phase 3 models use new names.
3. **Do not touch** the reward function, the observation space, the environment
   constants, or the experimental design. The external audit validated these.
4. **Addenda are APPENDED to `REPORT.md`.** The original narrative and its
   numbers are never rewritten. Where an addendum qualifies an earlier
   conclusion, it says so *in the addendum*.
5. **Never report waiting time without throughput beside it.** A controller can
   fake excellent waiting time by refusing to admit vehicles. This is the
   project's central measurement discipline.
6. **Identical safety constraints for every controller** (4 s amber, 10 s min
   green). Differing amber times would be the easiest way to accidentally rig
   the comparison.

---

## 6. Architecture

### Fairness design (the thing that matters most)

- **One code path.** Fixed-time, the heuristic, and both RL agents all run
  through `eval/simulation.py:run_episode`. They are not merely measured with
  the same metrics — they are measured by the same code.
- **Same vehicles.** A `(scenario, seed)` pair names a file listing exactly
  which car arrives when and where it goes. Every controller drives the
  identical hour of traffic.
- **Unseen test data.** Training uses demand seeds 0–9; evaluation uses seeds
  100–104. No overlap.
- **A strong baseline.** Webster-tuned fixed-time is *given the demand in
  advance* — an advantage the RL agent never receives.

### The environment

- **Observation:** `Box(21,)` — 4 phase one-hot + 1 min-green flag + 8 lane
  densities + 8 lane queues (sumo-rl default).
- **Action:** `Discrete(4)` — which green phase to run next.
- **Reward:** `negative_queue_reward` = `-get_total_queued()/8`. Episode return
  is minus total vehicle-seconds of queueing, which *is* the reported objective.
- **`gamma=0.95`** ~ a 100 s horizon ~ one signal cycle.
- **Geometry:** 4-way, 250 m arms, 2 lanes per approach (lane 0 =
  straight+right, lane 1 = protected left), 12 controlled movements.
- **Signal:** 4 phases `GGrrrrGGrrrr` / `rrGrrrrrGrrr` / `rrrGGrrrrGGr` /
  `rrrrrGrrrrrG`; 104 s cycle (32 s through, 12 s left, 4 s amber).
- **Route rotation:** `RotatingRouteEnv` cycles route files across episodes so
  the agent cannot memorise one arrival sequence.

### Scenarios

`balanced` · `ns_heavy` (**the training scenario**) · `peak` · and the
robustness set `ew_heavy` (mirror of trained) · `surge` · `light`.

### File map

```
PROJECT_CONTEXT.md   THIS FILE - read first
README.md            User-facing setup and run instructions
REPORT.md            The deliverable: findings, method, limitations, addenda A-E
VERIFICATION.md      Evidence that audited defects D1-D9 and F1-F4 were fixed
docs/PROGRESS.md     HISTORICAL build log (superseded; design rationale valid)
docs/REMEDIATION_LOG.md  Working log (STALE Phase 3 table - see 3.2)

sumo_setup.py        Locates SUMO. Import before any traci/sumolib/sumo_rl.
smoke_test.py        5-check end-to-end toolchain verification
pyproject.toml       Python floor, pytest config, pinned ruff rule set

networks/
  nodes/edges/connections .xml     hand-written geometry
  build_network.py                 netconvert + explicit 4-phase program
  generate_routes.py               Poisson arrivals per scenario
  routes/                          60 route files - IRREPLACEABLE, see rule 1

baseline/
  fixed_time.py                    untuned + Webster-tuned controllers
  run_baseline.py                  standalone baseline runner

agent/
  training.py                      shared loop, route rotation, logging
  train_dqn.py / train_ppo.py      the two algorithms
  train_mixed.py                   DQN on three demand patterns (addendum C)
  models/                          dqn, ppo, dqn_seed1, dqn_seed2,
                                   dqn_mixed, ppo_long  (.zip)

eval/
  simulation.py                    run_episode + the +safe constraint wrapper
  evaluate.py                      the comparison sweep (--append mode)
  paths.py                         THE single place output filenames are decided
  make_plots.py / make_tables.py   figures / generated report tables
  significance.py                  paired t + exact Wilcoxon (addendum A)
  followups.py                     generates addenda B-E

tests/                             276 tests; documented counts self-check
results/
  logs/     per-episode CSVs, summaries, 78 trace CSVs, runs/ chain logs
  plots/    4 figures used in REPORT.md
  tripinfo/ raw SUMO per-vehicle XML - REGENERABLE, gitignored, see 8
```

---

## 7. Defects found and fixed — do not reintroduce

### From the external audit (D-series, all fixed; evidence in VERIFICATION.md)

`D1` non-reproducible route seeding · `D2` no version control · `D3` no tests ·
`D4` report claimed a provenance guarantee it lacked · `D5` `MIN_GREEN` defined
twice · `D6` honesty metric failed silently to zero · `D7/D8` docs and constants
· `D9` packaging hygiene.

### Found in the Phase 3 code itself (F-series) — three would have failed silently

- **F4 — the safety wrapper enforced nothing.** The worst.
  `SafetyConstrainedController` reset a phase's starvation clock when the
  wrapped controller *asked* for that phase — but sumo-rl refuses a phase change
  while the current green is younger than `yellow + min_green`, so **a request
  is not a service**. Measured: the phase asked for was not the phase then green
  on 573 of 720 decisions; the wrapper fired zero overrides while the junction
  actually starved a movement for 245 s against a 200 s bound.
  `greedy-queue+safe` produced a trace bit-identical to `greedy-queue`, which had
  been misread as "the policy is already safe" — it is what a wrapper that never
  fires looks like. **Fixed** by advancing clocks from `signal.green_phase`, the
  phase actually showing green.
- **F1 — the wrapper could not wrap a fixed-time plan.** It read
  `len(signal.green_phases)`, which sumo-rl only builds when *it* drives the
  signal. Fixed by reading `num_green_phases`, published in both modes.
- **F2 — trace filenames were built in three places and disagreed.** A missing
  trace is a legitimate state, so the row was silently dropped — Addendum E
  would have published with its key controllers missing. Fixed by `eval/paths.py`
  as the single naming authority.
- **F3 — a generated table would have crashed on the new controller names.**
  `KeyError` at the last pipeline step, after training had run.

### Earlier bugs whose fixes must be kept

- **Reward telescoping.** sumo-rl's default `diff-waiting-time` telescopes to
  ~0 over an episode and cannot produce a readable training curve. Replaced —
  see §6.
- **Phase counting.** `getPhase()` only moves under fixed-time; sumo-rl's
  `green_phase` only moves under RL control. Now counts green-to-green changes
  in the actual `getRedYellowGreenState()` string, which works for both.
- **Webster's method.** The original closed-form cycle produced plans *worse
  than untuned* (v/c 0.58 -> 0.81 on `balanced`), because clamping light
  left-turn phases up to the 10 s minimum inflated the cycle while through
  phases kept their original green. Replaced with a **constrained search over
  cycle lengths**.

### Statistical caveat that must always be stated

With n=5 paired observations, a two-sided exact Wilcoxon has a floor of
2/32 = **0.0625 and can never reach 0.05**, however large the effect. The paired
t-test is therefore the primary result, and this floor is stated beside every
Wilcoxon number. A `0.0625` means *every* seed favoured the same controller.

---

## 8. Repository hygiene — pending cleanup

Analysis done 2026-09-18. `results/tripinfo/` holds **391 XML files, 271 MB** —
about 95% of the repository's weight excluding `.venv`.

**These are safe to delete.** Verified by reading the code: `run_episode` writes
each tripinfo XML and parses it into metrics *within the same episode*
(`eval/simulation.py:718`); nothing reads them afterwards. The tests use
`tmp_path` fixtures, never the real directory. The metrics they produced are
permanently captured in `evaluation_raw.csv`. `.gitignore` already declares them
regenerable. Re-running `eval/evaluate.py` (~26 min) recreates them, so no
verification path is lost.

```bash
rm -f results/tripinfo/*.xml           # 271 MB, regenerable
rm -rf .pytest_cache .ruff_cache       # tool caches
find . -path ./.venv -prune -o -name "__pycache__" -type d -exec rm -rf {} +
```

Also worth doing: `.pytest_cache/` and `.ruff_cache/` are **not** in
`.gitignore` and should be.

**KEEP — do not mistake these for clutter:**

- `networks/routes/` (13 MB) — irreplaceable, see rule 1
- `results/logs/trace_*.csv` (78 files) — **Addendum E reads these**
- `results/plots/` — the figures in `REPORT.md`
- `results/logs/runs/` (48 KB) — chain evidence and the resume marker
- `agent/models/*.zip` — the trained policies

---

## 9. Design decisions settled — do not re-litigate

- Episode 3600 s; decision interval 5 s; yellow 4 s; min green 10 s.
- `gamma=0.95` ~ one signal cycle.
- **`additional_sumo_cmd` is `.split()` by sumo-rl**, so paths passed to SUMO
  must contain **no spaces**. This repo lives under "Design of Smart Cities",
  so `run_episode` `os.chdir`s to the repo root and passes a *relative* tripinfo
  path. **Do not "simplify" this away** — it will break on this machine.
- tripinfo uses `--tripinfo-output.write-unfinished true` to avoid survivorship
  bias: the worst-delayed vehicles are the ones still queued when the hour ends.
  Dropping them makes every controller look better and the congested ones most
  of all.
- `_quiet_stdout()` uses an **fd-1 redirect**, not Python's `redirect_stdout`,
  because SUMO's chatter comes from a child process.
- **Plotting:** rounded bar tops are built in **display/pixel** coordinates, not
  data coordinates (x is a scenario index, y is seconds — a data-unit radius is
  meaningless). Because of that, the figure must be **created at the same dpi it
  is saved at**; `FIG_DPI` pins both. Creating at 100 and saving at 160 moved
  every bar.
- Significance testing is implemented **without SciPy** — `eval/significance.py`
  carries its own regularised incomplete beta and an exact Wilcoxon by
  enumerating all 2^n sign assignments, both tested against published critical
  values.
- **`fixed-time-tuned+safe` is reported as a null by construction.** SUMO runs
  that program itself and the controller's `act()` is never consulted, so the
  wrapper is inert. It is still run and reported, but Addendum E says plainly
  that the zero is *arithmetic, not evidence*, and establishes compliance from
  the trace instead.
- PPO converges to roughly the same place as DQN, just more slowly (~11.7 vs
  ~11.1 vehicles) — the expected sample-efficiency cost of being on-policy.
  Lesson recorded at the time: **do not call a curve plateaued while it is still
  descending.**

---

## 10. Testing discipline

```bash
./.venv/Scripts/python.exe -m pytest        # 276 tests, ~3 s
./.venv/Scripts/python.exe -m ruff check .  # rule set pinned in pyproject.toml
```

The suite targets places where a bug would be **quiet rather than loud** — a
number that is wrong but plausible:

| Module | Tests | Guards |
|---|---|---|
| `test_webster.py` | 71 | Green allocation, delay formula, per-scenario plans |
| `test_followups.py` | 29 | Addendum generators: branch choice and table shape |
| `test_significance.py` | 27 | The hand-rolled t and Wilcoxon distributions |
| `test_route_determinism.py` | 26 | D1, seeding, arrival process, held-out seeds |
| `test_evaluate_merge.py` | 25 | Append-mode merge, tagged and `+safe` names |
| `test_safety_constraints.py` | 24 | F1, F4 — max green and the starvation bound |
| `test_network_phases.py` | 22 | Phase strings, exclusivity, amber derivation |
| `test_output_naming.py` | 19 | F2, F3 — trace filenames, generated-table guards |
| `test_report_matches_data.py` | 16 | D4, report-to-CSV provenance |
| `test_tripinfo.py` | 9 | Trip parsing, survivorship bias |
| `test_documented_counts.py` | 8 | This table and these totals, against the suite |

Two patterns worth preserving:

- **Documents are checked by running, not by reading.** `tests/conftest.py`
  records what pytest actually collected; `test_documented_counts.py` compares
  every count quoted in `README.md` and `VERIFICATION.md` against it and fails
  naming the file and the number to change. The counts had drifted twice before
  this existed.
- **Report-to-data provenance.** Every waiting-time cell in `REPORT.md` is
  parsed back out of the markdown and checked against the CSVs, so the report
  cannot drift from the data it describes.

`VERIFICATION.md` additionally records a **mutation check**: five deliberate
breaks were applied to an isolated copy of the repository, and the suite caught
5/5.

---

## 11. Quick orientation for an AI picking this up

1. Read this file. Then `REPORT.md` §1 and §6 for findings and limitations.
2. Run `./.venv/Scripts/python.exe -m pytest` — expect 276 green before touching
   anything.
3. Check `git status`. Expect the Phase 3 outputs described in §3.3 to be
   untracked.
4. The highest-value next action is **§3.1** — four commands, five minutes, and
   it closes the gap between what the documents promise and what they contain.
5. Obey §5. Rules 1 and 2 protect artifacts that **cannot be regenerated**.
