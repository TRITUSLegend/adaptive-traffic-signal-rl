# Build progress / resume notes

**Paused:** 2026-08-31, at the user's request.
**Resume by reading this file first.** It records what is done, what is in
flight, and the exact next steps.

---

## Environment (DONE — do not redo)

SUMO could **not** be found on this machine and there is no admin package
manager, but the `eclipse-sumo` PyPI wheel solved it with no admin rights.
**No human action was needed for the SUMO install.**

| Thing | Value |
|---|---|
| venv | `.venv/` built with `py -3.13` (system default Python is 3.9, too old) |
| Python | 3.13.5 |
| SUMO | 1.27.1, via `pip install eclipse-sumo` → `.venv/Lib/site-packages/sumo` |
| torch | 2.13.0+cpu (installed from the PyTorch CPU index to avoid a ~2.5 GB CUDA download) |
| sumo-rl | 1.4.5 |
| stable-baselines3 | 2.7.1 |
| gymnasium | 1.2.3, numpy 2.5.2, pandas 3.0.5, matplotlib 3.11.1 |

Run everything with `./.venv/Scripts/python.exe <script>`.

`sumo_setup.py` (repo root) locates SUMO and must be imported **before** any
`traci` / `sumolib` / `sumo_rl` import — `sumo_rl` raises `ImportError` at
import time if `SUMO_HOME` is unset.

---

## Completed and verified

1. **`sumo_setup.py`** — SUMO discovery (env var → pip wheel → PATH).
2. **`networks/{nodes,edges,connections}.*.xml`** — hand-written 4-way
   intersection, 250 m arms, 2 lanes per approach (lane 0 = straight+right,
   lane 1 = protected left).
3. **`networks/build_network.py`** — runs `netconvert`, then overwrites the
   auto-generated signal program with an explicit 4-phase plan.
   *Verified*: 12 controlled movements, phases
   `GGrrrrGGrrrr` / `rrGrrrrrGrrr` / `rrrGGrrrrGGr` / `rrrrrGrrrrrG`,
   104 s cycle (32 s through, 12 s left, 4 s amber).
4. **`networks/generate_routes.py`** — Poisson arrivals, 6 scenarios, 40 route
   files written. Train = `ns_heavy` seeds 0–9; eval = seeds 100–104 (disjoint).
   Scenarios: `balanced`, `ns_heavy` (train), `peak`, and robustness set
   `ew_heavy`, `surge`, `light`.
5. **`smoke_test.py`** — 5 checks, **all passing**.
6. **`eval/simulation.py`** — the shared episode runner + metrics. **Working.**

### Measured baselines (ns_heavy, seed 100, one episode, ~9 s wall)

| controller | mean wait | mean queue | throughput | switches | return |
|---|---|---|---|---|---|
| fixed-time | 47.2 s | 23.3 veh | 1746 | 138 | −2100.8 |
| greedy-queue | 23.4 s | 11.6 veh | 1749 | 236 | −1044.0 |
| random actions | — | ~56 veh | — | — | — |

Fixed-time's 138 switches match its 104 s cycle exactly — a good correctness check.

### Two real bugs found and fixed (keep these fixes)

- **Reward.** sumo-rl's default `diff-waiting-time` **telescopes to ≈0 over an
  episode** (measured: −2 for both controllers), so it cannot produce a
  meaningful training curve. Replaced with `negative_queue_reward` in
  `eval/simulation.py` = `-get_total_queued()/8`. Its episode return is minus
  the total vehicle-seconds of queueing, which *is* the objective. Justified at
  length in the docstring.
- **Phase counting.** `getPhase()` only moves under fixed-time; sumo-rl's
  `green_phase` only moves under RL control. Now counts green-to-green changes
  in the actual `getRedYellowGreenState()` string, which works for both.

Also: `_quiet_stdout()` (fd-1 redirect) suppresses SUMO's per-connection
chatter; a Python-level `redirect_stdout` does not work on a child process.

---

## Status after the resume

### DQN training: DONE
170 episodes, 45.9 min. Mean queue fell 56.0 -> 11.4 vehicles (-80%), with a
clear plateau from about episode 70. `agent/models/dqn.zip` saved,
`results/logs/training_curve_dqn.csv` written.

### Everything below is now built and verified
- `eval/simulation.py` refactor finished (`Controller` base class +
  `on_episode_start` hook; factories always receive the env).
- `baseline/fixed_time.py` - Webster-tuned fixed-time controller.
- `baseline/run_baseline.py` - standalone baseline runner.
- `eval/evaluate.py` - full sweep, plus an `--append` mode that merges a newly
  trained controller into an existing `evaluation_raw.csv` without re-running
  everything.
- `eval/make_plots.py` - four figures, validated categorical palette.
- `agent/train_ppo.py`, `requirements.txt`, `README.md`, `.gitignore`.

### Two plotting gotchas worth remembering
- Rounded bar tops must be built in **display/pixel** coordinates, not data
  coordinates (x is a scenario index, y is seconds - a data-unit radius is
  meaningless).
- Because those bars are in pixel space, the figure must be **created at the
  same dpi it is saved at**. Creating at the default 100 and saving at 160
  moved every bar. `FIG_DPI` now pins both.

### Interim findings (SUPERSEDED - kept only to explain the Webster fix)

The numbers recorded here mid-build came from the *first* Webster
implementation and a partial sweep. They are wrong and are retained only
because they are what exposed the bug: the tuned plan was scoring just -8.4%
against untuned on ns_heavy, and was *worse* than untuned on balanced.

Diagnosis: clamping the light left-turn phases up to the 10 s minimum green
inflated the cycle while the through phases kept the green they had been
allocated for the shorter cycle, so their share of the cycle fell (v/c 0.58 ->
0.81 on balanced). The closed-form Webster cycle assumes green can be split
freely in proportion to demand, which the minimum-green constraint forbids.

Fixed by searching cycle lengths and allocating green under the constraint.
**Final numbers are in REPORT.md and results/logs/ - use those, not anything
in this section.**

### PPO: converges to the same place as DQN, just more slowly
Mid-run it looked stuck around 24 vehicles, but that reading was premature - it
kept improving: 24.0 (ep 40) -> 18.1 (ep 70) -> 14.9 (ep 90) -> 11.7 (ep 150),
plateauing at ~11.7 against DQN's ~11.1.

This is the useful version of the stretch goal: two algorithms from different
families (off-policy value learning vs on-policy policy gradient) independently
land within half a vehicle of each other, which is good evidence the result
reflects the traffic problem rather than one algorithm's quirks. The slower
approach is the expected sample-efficiency cost of being on-policy - PPO
discards its experience after each update, DQN replays it.

Lesson for the writeup: do not call a curve plateaued while it is still
descending.

---

## Status: COMPLETE

All seven required deliverables plus stretch goals 8 (PPO) and 10 (robustness)
are built, run and written up. Stretch goal 9 (multi-intersection corridor) was
not attempted - the brief says to prioritise robustness over it, and robustness
is done.

Final artefacts: `REPORT.md`, `README.md`, 4 figures in `results/plots/`,
`results/logs/{evaluation_raw,evaluation_summary,baseline_summary}.csv`,
`results/logs/results_tables.md`, both trained models in `agent/models/`.

### Headline numbers (5 seeds per controller per scenario)
DQN vs untuned fixed-time, mean waiting time: -33.3% (balanced), -60.7%
(ns_heavy, trained), -45.3% (peak), -46.4% (ew_heavy), -50.3% (surge), -51.2%
(light). Throughput equal or slightly higher everywhere.

### Three things the report is careful about
1. The greedy heuristic beats the DQN on 3 of 6 scenarios; the DQN's real edge
   is under saturation (peak/surge, ~25% better than greedy). Reported as such
   rather than as "deep RL wins".
2. Untuned fixed-time failed to insert ~227 veh/h under `surge` and ~95 under
   `peak`. Those vehicles accrue no waiting time, so the baseline's numbers
   flatter it and the DQN's margin is an underestimate. DQN/PPO dropped none.
3. The agent carries a directional bias from training on ns_heavy: 20.6 s there
   vs 24.9 s on the mirrored ew_heavy (+21%). Greedy is unaffected.

### Two bugs caught by testing rather than shipped
- sumo-rl's default `diff-waiting-time` reward telescopes to ~0 per episode, so
  it cannot produce a readable training curve. Replaced with a negative-queue
  reward whose episode total IS the reported metric.
- The first Webster implementation produced plans *worse than untuned* on
  `balanced` (v/c 0.58 -> 0.81), because clamping light left-turn phases up to
  the 10 s minimum inflated the cycle while through phases kept their original
  green. Replaced the closed-form cycle with a constrained search over cycle
  lengths. The tuned plan now wins in 5 of 6 scenarios (and is correctly a wash
  on `balanced`, where symmetric demand needs no tuning).

## Decisions worth not re-litigating

- Episode = 3600 simulated s; decision interval 5 s; yellow 4 s (identical for
  every controller — this is what keeps the comparison fair); min green 10 s.
- `gamma=0.95` ≈ a 100 s horizon ≈ one signal cycle.
- Observation: sumo-rl default, `Box(21,)` = 4 phase one-hot + 1 min-green flag
  + 8 lane densities + 8 lane queues. Action: `Discrete(4)`.
- Route files rotate across episodes (`RotatingRouteEnv`) so the agent cannot
  memorise one arrival sequence.
- **`additional_sumo_cmd` is `.split()` by sumo-rl**, so paths passed to SUMO
  must contain no spaces. This repo lives under "Design of Smart Cities", so
  `run_episode` `os.chdir`s to the repo root and passes a *relative* tripinfo
  path. Do not "simplify" this away.
- tripinfo uses `--tripinfo-output.write-unfinished true` to avoid survivorship
  bias (the worst-delayed vehicles are the ones still queued at the hour's end).
