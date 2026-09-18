# CORRIDOR_PLAN.md — plan for the multi-intersection corridor extension

**Status: PROPOSAL. No code has been written and nothing has been trained.**
This document exists to be approved, amended or rejected before implementation
begins. Written 2026-09-19 against the tree at commit `e44aa24`.

**What this extension is for.** The assignment brief requires a corridor
deliverable, actuated baselines, multi-agent RL including MA2C, real-data-seeded
demand, and a robustness study (detector failure, demand shift, topology
transfer). The finished single-intersection project satisfies none of the first
four and only partially the fifth. This plan closes that gap **as new work in
new directories** — it modifies no existing single-intersection code, data,
model or report.

Facts below marked **[verified]** were checked against this machine today.
Everything else is an estimate and is labelled as such.

---

## 0. Summary of what I recommend

| Question | Recommendation |
|---|---|
| Corridor size | **3 signalised intersections**, synthetic arterial, 400 m spacing |
| Built how | Tile the existing hand-written geometry via a generator script |
| Real data | **RESCO `cologne3`** (bundled, real Cologne corridor) **[verified on disk]** + optionally Toronto/NYC open turning-movement counts to calibrate the Poisson generator |
| Transfer target | `cologne3` doubles as the **unseen topology** for zero-shot transfer |
| DQN/PPO multi-agent | **Independent learners** (IDQN, IPPO), one policy per intersection |
| MA2C | **From scratch**, ~500 lines PyTorch. Not in SB3; MARL libraries are a bad fit here |
| Actuated baseline | **SUMO-native `tlLogic type="actuated"`** (gap-based) **[verified supported]** |
| Compute | ~13–17 h for a 1-seed headline; **3 seeds is ~35–45 h and I do not recommend it** |
| Report | **New file `CORRIDOR_REPORT.md`.** `REPORT.md` is not touched at all |

The single design choice I am most pleased with: **`cologne3` serves as both the
real-data requirement and the topology-transfer target.** One real corridor, two
brief requirements, no contrivance.

---

## 1. Corridor topology

### What exists and can be reused

The current network is hand-written XML compiled by `netconvert`
(`networks/nodes.nod.xml`, `edges.edg.xml`, `connections.con.xml`, then
`build_network.py` overwrites the signal program with an explicit 4-phase plan).
Naming is systematic — `N2C` is inbound from north, `C2N` outbound — 2 lanes per
approach, lane 0 straight+right, lane 1 protected left, 12 controlled movements,
4 phases, 104 s cycle.

**That geometry is worth tiling rather than replacing.** It is validated by 22
tests in `test_network_phases.py`, its phase strings are known-correct, and the
existing agents were trained against exactly this movement structure — which is
what makes single→corridor transfer measurable at all.

### What I propose to build

A linear arterial: **3 signalised intersections** `J1 J2 J3` on a west–east
arterial, each a 4-way junction with the same 2-lane/4-phase structure as the
existing one.

```
        N1          N2          N3
        |           |           |
  W ——[J1]——400m——[J2]——400m——[J3]—— E
        |           |           |
        S1          S2          S3
```

- **Spacing 400 m** between adjacent signals. Realistic urban arterial spacing,
  and long enough to hold a queue without spillback into the upstream junction —
  the same reasoning the existing 250 m arms were chosen for. Spillback would
  silently suppress demand and flatter every controller equally, which is the
  measurement failure the original project already guards against.
- **Outer approaches 250 m**, reusing the existing arm length unchanged.
- **Side streets** (N/S arms) at each junction, 250 m, so each junction has a
  genuine competing movement rather than being a pass-through.

Built by a **generator script** (`corridor/build_corridor.py`) that emits
nodes/edges/connections for an arbitrary N-intersection corridor and then writes
each junction's 4-phase program explicitly, exactly as `build_network.py` does.
Parameterising N is what makes the topology-transfer experiment cheap: a
2-intersection and a 5-intersection corridor come free from the same generator.

**A note on what must be verified during implementation:** the existing
`nodes.nod.xml` header comment says "four 500 m-long approach arms" in one line
and "Why 250 m arms?" in the next. I have not resolved which is authoritative.
The generator will read the existing values programmatically rather than
hard-coding a number I guessed from a comment.

### Why 3 and not more

| N | Est. training time per run | Verdict |
|---|---|---|
| 2 | ~2.0 h | Too small — a 2-signal "corridor" barely tests coordination |
| **3** | **~3.0 h** | **Recommended.** Genuine upstream/middle/downstream roles; the middle junction has two signalised neighbours, which is the minimum for MA2C's neighbour-sharing to mean anything |
| 5 | ~5.5 h | Better science, but ×3 algorithms ×seeds becomes infeasible here |
| 7+ (`ingolstadt7`) | ~8 h+ | Out of budget for training; usable as an *evaluation-only* transfer target |

**3 is the smallest corridor where MA2C is not degenerate.** With 2
intersections every agent is an edge agent and "neighbour sharing" is just
"the other one". With 3, J2 has two neighbours and J1/J3 have one, so the
architecture's asymmetry is actually exercised.

---

## 2. Real-data seeding

### The PeMS mismatch — stated plainly

**PeMS does not fit this problem and I will not pretend it does.** PeMS
(California Performance Measurement System) is *freeway* loop-detector data:
mainline and ramp volumes, speeds and occupancies at 30 s/5 min aggregation. It
contains no intersections, no turning movements, no signal phasing and no
urban arterial geometry. Seeding an urban signalised corridor from freeway
mainline counts would be calibrating the wrong quantity from the wrong facility
type and calling it real. The brief names PeMS as an example, not a requirement;
the honest response is to substitute an urban source and say why.

### What I propose instead

**Primary — RESCO `cologne3`, already on this machine [verified]:**

| Scenario | Signals | Demand | Provenance |
|---|---|---|---|
| `cologne3` | 3 | **4,494 individual vehicles**, departures spanning 28,798 s | TAPAS Cologne real mobility model |
| `ingolstadt7` | 7 | 3,031 trips, 45 vehicle types | Real Ingolstadt network model |

Both ship inside `sumo-rl` at `.venv/Lib/site-packages/sumo_rl/nets/RESCO/` and
are the standard RESCO benchmark (Ault & Sharon, NeurIPS 2021 Datasets &
Benchmarks). **Zero acquisition risk — no download, no API key, no licence
question.** This is real demand derived from a real city's mobility model, on a
real corridor's geometry.

**Secondary — open turning-movement counts, if network access is permitted:**

| Source | What it gives | Realistically obtainable? |
|---|---|---|
| **Toronto TMC** (City of Toronto Open Data) | 15-min turning-movement counts per intersection, per approach, by movement | **Best fit.** Directly gives per-approach arrival rates *and* turn fractions |
| NYC DOT Traffic Volume Counts | Hourly volumes per segment, by direction | Good for volumes, weak on turns |
| UK DfT road traffic counts | Annual/daily counts by direction | Too coarse for signal timing |
| UTD19 (ETH Zürich) | 40k+ urban loop detectors, 40 cities | Good, but detector-level not movement-level |

**How it would be used: calibration, not replay.** Concretely — take the 15-min
counts for a real arterial intersection, convert to per-approach hourly arrival
rates and turning fractions, and use those as the `lambda` and turn-probability
parameters of the *existing* Poisson generator's logic (reimplemented in
`corridor/generate_corridor_routes.py`, not by modifying the original).

**Why calibration rather than literal replay:**
1. The counts are 15-min aggregates, not vehicle-by-vehicle departures. Replay
   would require inventing the within-interval arrival process anyway.
2. The project's entire measurement discipline rests on running **many
   stochastic seeds per scenario** and reporting spread. A single literal replay
   is one sample with no variance estimate — it would discard the one habit that
   makes these numbers trustworthy.
3. Calibrated generation lets us build *demand-shift* scenarios (§5b) by moving
   a real base profile, which literal replay cannot do.

**Honest statement of what is approximated:** turning fractions and platoon
structure from upstream signals are the two things counts capture poorly. Under
calibration the arrival process at the corridor entry is Poisson, which is
defensible at the boundary but *wrong mid-corridor*, where real arrivals are
platooned by the upstream signal. **In our corridor that platooning is produced
by the simulation itself**, since mid-corridor arrivals come from the upstream
junction rather than from the generator — so the approximation is confined to
the corridor boundary, which is where Poisson is most defensible. This should be
stated in the report rather than left for a reader to notice.

**If network access is not available**, `cologne3` alone satisfies the
requirement and the plan does not change shape. I will flag the substitution
explicitly rather than quietly downgrading.

---

## 3. Multi-agent RL

### DQN and PPO: independent learners (say so plainly)

**The plan is independent learners — IDQN and IPPO — one policy per
intersection, each seeing only its own local observation.** This is the default
and I am naming it as such rather than implying something more sophisticated.

- Each junction gets its own SB3 `DQN` / `PPO` instance over a `Box(21,)`
  observation identical in shape to the single-intersection one.
- **Keeping the per-agent observation identical in shape to the existing one is
  a deliberate constraint**, because it is what makes two things possible: the
  existing `dqn.zip` can be dropped onto a corridor junction zero-shot, and a
  corridor policy can be evaluated on a *different-length* corridor (§5c). If
  the observation grew with corridor length, topology transfer would be
  impossible by construction.
- Known limitation to state in the report: independent learners face a
  non-stationary environment (each agent's neighbours are also learning). This
  is exactly the weakness MA2C is designed to address, which is why having both
  is the point rather than redundancy.

I will also record, but not necessarily run, the parameter-sharing variant (one
network, agent id in the observation) as a cheaper alternative if compute bites.

### MA2C: build it from scratch

**MA2C is not in stable-baselines3** and there is no maintained SB3-compatible
implementation. Options considered:

| Option | Assessment |
|---|---|
| **From-scratch PyTorch** | **Recommended.** ~500 lines. No new dependency |
| MARLlib / RLlib | Needs Ray. Heavy, and Ray-on-Windows-CPU is a known source of pain. Rejected |
| PyMARL / EPyMARL | Research code coupled to SMAC; adapting to PettingZoo+SUMO is comparable work to writing it, with worse readability |
| `sumo-rl` + PettingZoo `parallel_env` | **[verified present]** — this is the *environment* API we will build on, but it provides no MA2C algorithm |

**Precedent in this project supports from-scratch:** `eval/significance.py`
hand-rolls the Student-t and exact Wilcoxon distributions rather than adding
SciPy, and covers them with 27 tests against published critical values. The same
tradeoff applies — a small, tested, readable implementation beats a heavy
dependency on a CPU-only Windows box.

**What MA2C means concretely** (Chu et al., *IEEE T-ITS* 2020), and what I will
implement:
1. Independent actor–critic per intersection (A2C).
2. **Neighbour observation sharing** — each agent's input is its own observation
   concatenated with its neighbours', scaled by a spatial discount factor α.
3. **Neighbour policy sharing** — neighbours' most recent action distributions
   are appended to the input, which is what stabilises the non-stationarity.
4. **Spatially discounted reward** — each agent's reward blends its own queue
   with neighbours' queues, decayed by distance.

**Honest effort estimate: 2–4 working days** including tests. The algorithm
itself is perhaps a day; the rest is the neighbour-graph plumbing, making the
observation assembly deterministic and testable, and the tests. The original
paper also uses an LSTM — **I propose omitting the LSTM in v1** and flagging it
as a deviation, since recurrent state plus the existing 5 s decision interval
adds a debugging surface disproportionate to its likely benefit here. That is a
substitution, so per rule 8 it is flagged here rather than made quietly.

**Risk, stated up front:** MA2C is the single highest-risk item in this plan. If
it under-performs IPPO it will be tempting to read that as a bug. The mitigation
is to fix the evaluation protocol before training and report whatever comes out,
the way Addendum D reported that PPO's longer budget only partly supported the
original explanation.

---

## 4. Actuated baseline

### The greedy heuristic is not actuated, and should not be relabelled

`greedy-queue` picks whichever phase currently has the longest queue, every 5 s,
with global knowledge of all queues. A real actuated controller does something
structurally different: it runs a **phase sequence** and decides only *when to
terminate the current green*, based on **detector gap-out** — green extends
while vehicles keep crossing a detector within a gap threshold, and terminates
on gap-out or max-out.

Relabelling greedy-queue as "actuated" would be exactly the kind of quiet
overstatement this project has twice removed (F4; the `fixed-time-tuned+safe`
null). **A new controller is needed.** Greedy-queue stays, honestly labelled a
queue-greedy heuristic.

### What I propose

**Use SUMO's native `<tlLogic type="actuated">`.** Verified supported in SUMO
1.27.1 on this machine (`netconvert --tls.default-type` **[verified]**). It is
gap-based, standard, documented, and auto-generates the lane-area detectors it
needs. Parameters to set explicitly and report: `max-gap` (≈3.0 s),
`detector-gap`, `min-dur` = 10 s and `max-dur` = 60 s to match the project's
existing min-green and the max-green used by the safety wrapper.

**Two consequences that must be handled, not discovered later:**

1. **SUMO drives an actuated program itself**, exactly as it drives the
   fixed-time program — the controller's `act()` is never consulted. So the
   `+safe` wrapper is **inert on it by construction**, the same null that
   Addendum E documents for `fixed-time-tuned+safe`. The corridor report must
   say this in the same terms rather than presenting a passing safety check.
2. **Actuated control is detector-dependent, and that is the whole point of the
   detector-failure experiment** (§5a). Fixed-time uses no detectors and is
   therefore immune by construction; actuated and the RL agents are not. This
   asymmetry is the most interesting result the robustness study can produce,
   and §5a is designed around it.

A NEMA-style dual-ring controller is available in SUMO too, but I do not
recommend it for v1: more configuration surface, and the gap-based actuated
logic is what "actuated baseline" conventionally means in this literature.

---

## 5. Robustness experiments

This is the brief's novelty requirement and should be designed first, not bolted
on. **The comparison point throughout is fixed-time's degradation under the
identical perturbation**, not the DRL controller's absolute degradation.

For every experiment, report three numbers per controller: clean performance,
perturbed performance, and **the margin over fixed-time at each perturbation
level**. A controller that degrades badly but still beats fixed-time is a
different finding from one that degrades past it, and only the margin
distinguishes them.

### (a) Detector failure

**Mechanism.** A wrapper sits between the environment and the controller and
corrupts the observation before the controller sees it. The simulation itself is
untouched — the vehicles are still there, the controller just cannot see them.
That separation is what makes it a *sensing* failure rather than a demand change.

**Failure modes** (both, reported separately — they are not equivalent):

| Mode | Behaviour | Why it matters |
|---|---|---|
| **Stuck-at-zero** | Failed lane reports 0 queue / 0 density | The dangerous one: an approach that looks empty may never be served |
| **Freeze / last-value** | Failed lane holds its last good reading | Milder, and closer to a real comms dropout |

**Failure rates** p ∈ {0.0, 0.1, 0.25, 0.5} of lanes failed, chosen per episode
from a **fixed RNG seeded by (scenario, seed, p)** so that *every controller sees
the identical set of failed detectors on the identical hour of traffic*. This is
the corridor version of the existing "same vehicles" fairness rule, and it is
what makes the comparison paired.

**Who is affected by construction — state it, do not discover it:**

| Controller | Detector-dependent? | Expected degradation |
|---|---|---|
| Fixed-time | **No** — runs a clock | **Zero, by construction.** Not a virtue; it cannot respond to demand at all |
| Actuated | Yes — gap-out needs detectors | Real degradation |
| Greedy-queue | Yes — reads queues | Real degradation |
| IDQN / IPPO / MA2C | Yes — queues+densities are the observation | Real degradation; the question is how gracefully |

**The honest framing:** fixed-time's immunity is the same kind of "null by
construction" as `fixed-time-tuned+safe`. The finding is not "fixed-time is
robust" — it is *"the adaptive controllers retain an X s advantage over
fixed-time even with half their detectors dead"*, or they do not, in which case
there is a crossover point and that crossover is the result worth reporting.

### (b) Demand shift

Train on one demand profile, evaluate on shifted ones without retraining:

| Shift | What changes | Precedent |
|---|---|---|
| Volume scaling | ×0.7, ×1.3, ×1.6 of calibrated base | New |
| Directional reversal | Arterial-dominant → cross-street-dominant | Directly extends Addendum C's `ns_heavy`→`ew_heavy` finding |
| Temporal | Flat → sharp peak within the hour | New |
| Incident | One link capacity-reduced mid-episode | New; the most realistic |

Addendum C already established the single-intersection version of this result
(directional penalty +20.8% → +1.2% under mixed-demand training), so the
corridor study has a baseline expectation to compare against rather than
starting blind.

### (c) Topology transfer (zero-shot)

Train on the 3-intersection synthetic corridor; evaluate **with no retraining**
on:

| Target | Differs how | Source |
|---|---|---|
| 2-intersection corridor | Shorter; both agents are edge agents | Same generator |
| 5-intersection corridor | Longer; three interior agents | Same generator |
| Corridor at 250 m spacing | Same N, tighter coupling, more spillback risk | Same generator |
| **`cologne3`** | **Real geometry, real demand, irregular junctions** | **Bundled [verified]** |

`cologne3` is the strong test: genuinely not drawn by us, with junction
geometry and phase structure we did not choose.

**The design constraint this imposes** — repeating it because it is easy to
violate early and expensive to fix late: **each agent's observation must be
fixed-dimension and local.** Any design where an agent sees the whole corridor
makes transfer to a different corridor length impossible. This is a constraint on
§3, driven by §5c, and it is why the per-agent observation stays `Box(21,)`.

**Expected difficulty, stated in advance:** `cologne3`'s junctions may not have
4 phases or 2 lanes per approach. Where the action space differs, zero-shot
transfer of a `Discrete(4)` policy is undefined. Mitigations, in order of
preference: (i) evaluate only on `cologne3` junctions whose phase count matches;
(ii) define an explicit action-projection and document it as a deviation; (iii)
if neither is defensible, report the synthetic-corridor transfers and state
plainly that real-topology transfer was not achievable — that is a legitimate
negative result, not a failure to be hidden. **I will confirm `cologne3`'s
per-junction phase structure before committing to this experiment**, and report
back if it forces option (iii).

---

## 6. Compute budget

### Measured baseline from this machine **[verified]**

| Run | Episodes | Wall time | Per episode | Throughput |
|---|---|---|---|---|
| DQN (seed 2) | 170 | 52.7 min | 18.6 s | 38.7 steps/s |
| PPO (long) | 400 | 49.7 min | 7.5 s | 96.6 steps/s |

DQN is ~2.5× slower per episode than PPO — it takes a gradient step per
environment step, PPO batches.

### Corridor scaling estimate

Cost scales with vehicles simulated (≈3× for 3 junctions) and with learning
work (≈3× for 3 independent agents, though these overlap since SUMO stepping is
shared). **Estimate ≈3.5× single-intersection per episode**, which is an
estimate and not a measurement:

| Run | Est. per episode | 170 episodes |
|---|---|---|
| IDQN, 3 agents | ~65 s | **~3.1 h** |
| IPPO, 3 agents | ~26 s | **~1.2 h** |
| MA2C, 3 agents | ~35 s | **~1.7 h** |

### Totals

| Scope | Training | Evaluation + robustness | **Total** |
|---|---|---|---|
| **1 seed per algorithm** | ~6.0 h | ~7–11 h | **~13–17 h** |
| 3 seeds per algorithm | ~18 h | ~17–27 h | **~35–45 h** |

The evaluation range is wide because the robustness sweep multiplies fast:
6 controllers × 4 failure rates × 2 failure modes × scenarios × seeds. That
product, not the training, is the real budget risk.

### Recommendation

**Run 1 seed per algorithm for the headline, then 3 seeds for the single best
DRL controller only.** ~17–20 h total, and it keeps a variance estimate where it
matters. Justification is already in the repo: Addendum B measured seed-to-seed
spread at 0.18–3.03 s depending on scenario, so a single seed is defensible for
a *ranking* but not for a claimed *margin*. Report both facts and cite Addendum B
for the magnitude of seed noise.

**Reduced fallback if time is shorter than that**, in the order I would cut:

1. Drop the ×0.7 and ×1.6 volume shifts, keep ×1.3 and directional reversal — saves ~3 h.
2. Drop the freeze/last-value detector mode, keep stuck-at-zero — halves the failure sweep.
3. Drop the 5-intersection transfer target, keep 2-intersection and `cologne3`.
4. Cut training to 120 episodes — the single-intersection curves plateau by ~70,
   so 120 is defensible; state it.
5. Last resort: 2-intersection corridor. **This weakens MA2C to the point of
   being degenerate** (§1) and I would rather cut seeds than do this.

I would **not** cut: the actuated baseline, detector failure, or `cologne3`
transfer. Those are the three things the brief specifically asks for that the
current repo lacks.

---

## 7. File layout

**Nothing below modifies, overwrites or writes into any existing
single-intersection directory.** All corridor work is confined to a new
top-level `corridor/`, a new `results/corridor/`, new test modules, and one new
report file.

```
corridor/                          ALL NEW
  __init__.py
  build_corridor.py                generates an N-intersection corridor network
  generate_corridor_routes.py      calibrated Poisson demand per scenario
  env.py                           multi-agent env over sumo-rl parallel_env
  controllers.py                   fixed-time, actuated, greedy, RL wrappers
  ma2c.py                          from-scratch MA2C (§3)
  train_idqn.py / train_ippo.py / train_ma2c.py
  evaluate_corridor.py             the corridor sweep
  robustness.py                    detector failure, demand shift, transfer
  perturbations.py                 the observation-corrupting wrapper (§5a)
  paths.py                         single naming authority (the F2 lesson)
  networks/                        generated corridor nets - NOTE: nested inside
    corridor3/ corridor2/ corridor5/   corridor/, NOT the root networks/
  routes/
  models/                          idqn_*.zip, ippo_*.zip, ma2c_*.pt

tests/                             NEW MODULES ONLY, no existing test touched
  test_corridor_network.py         phase strings, connectivity, spacing
  test_corridor_routes.py          determinism, calibration arithmetic
  test_perturbations.py            detector failure is reproducible and paired
  test_ma2c.py                     neighbour assembly, spatial discount
  test_corridor_isolation.py       asserts single-intersection files unchanged

results/corridor/                  NEW
  logs/  plots/

CORRIDOR_REPORT.md                 NEW top-level report
```

### Protecting the existing work

- **`REPORT.md` is not touched at all.** Not a pointer line, not a section.
  Reason beyond rule 4: `tests/test_report_matches_data.py` parses REPORT.md and
  checks every waiting-time cell against the CSVs, and
  `eval/make_tables.py --write` must keep leaving it byte-identical. Adding
  corridor content is an unnecessary interaction with a guarantee that currently
  holds. Discoverability is handled by pointers in `README.md` and
  `PROJECT_CONTEXT.md` instead.
- **Read-only reuse is allowed and intended.** `corridor/` may *import* from
  `sumo_setup.py` and from `eval/simulation.py` (e.g. the `Controller` base
  class, metric helpers) but will not modify them.
  `tests/test_corridor_isolation.py` will assert this by hashing the
  single-intersection source files, so an accidental edit fails the suite rather
  than being noticed later. This is the same move `test_documented_counts.py`
  makes for documentation.
- **`networks/routes/` is never written to.** Corridor demand goes in
  `corridor/routes/`. Rule 1 is structurally unbreakable because no corridor
  code has the old path.
- **No existing model is loaded for writing.** `agent/models/dqn.zip` will be
  *read* for the single→corridor transfer experiment, never written.

### One coupling that will bite if not planned for

`tests/test_documented_counts.py` checks the test counts quoted in `README.md`
and `VERIFICATION.md` against what pytest actually collected. **Adding corridor
test modules will change the total from 276 and fail that test** — correctly, by
design. The fix is to update the documented counts in the same commit that adds
the tests. Noting it here so it reads as expected behaviour rather than a
regression.

Also: `docs/REMEDIATION_LOG.md` currently ends with "**Not attempted,
deliberately:** the multi-intersection corridor." That line becomes false once
this work starts and will need a one-line update pointing here.

---

## 8. Open questions for you

1. **Seeds: 1 or 3?** I recommend 1 per algorithm plus 3 for the best, per §6.
2. **Network access for open data?** If yes I will calibrate from Toronto TMC.
   If no, `cologne3` alone carries the real-data requirement and I will say so
   in the report.
3. **Is `cologne3` acceptable as *the* real-data component**, given it is bundled
   with a dependency rather than downloaded by us? I think it is stronger, not
   weaker — it is a published benchmark others can reproduce exactly.
4. **MA2C without the LSTM** for v1 — acceptable? (§3, flagged per rule 8.)
5. **Corridor size 3** — confirm, or push to 5 and accept ~1.8× compute.

---

## 9. What happens on approval

In order, with a checkpoint after step 3 before any long training run:

1. `corridor/build_corridor.py` + `test_corridor_network.py` — geometry and
   phase plans verified the way the original network was.
2. `corridor/generate_corridor_routes.py` + determinism tests.
3. `corridor/env.py`, `controllers.py` (incl. actuated), `perturbations.py`, and
   a **short smoke run** of every controller on one corridor hour — fixed-time
   and actuated first, since they need no training and will surface geometry
   errors immediately. **Checkpoint here.**
4. `ma2c.py` + tests, before any training, since it is the highest-risk item.
5. Training runs, longest first.
6. Evaluation, robustness sweep, `CORRIDOR_REPORT.md`.

**I am stopping here and awaiting your review.** No corridor code has been
written, no dependency added, nothing trained.

---

## 10. Resolution of the open questions (appended 2026-09-19, post-approval)

Approved: plan as proposed, **1 seed**, **3-intersection corridor**. Questions 1
and 5 of §8 are therefore settled. The remaining three were delegated to my
judgment; here is what I chose and why. This section is appended — nothing above
has been rewritten.

### Q2 — network access for open data: NOT USED, and not needed

**Decision: calibrate from `cologne3` itself. No download.**

Rather than fetch Toronto TMC counts, I extracted the real per-entry flows from
the `cologne3` route file already on disk. Measured today:

| Quantity | Value |
|---|---|
| Busiest hour (06:31–07:59, real morning peak) | **2,856 veh/h** across the 3-junction corridor |
| Dominant entry #1 | 550 veh/h (19.3%) |
| Dominant entry #2 | 454 veh/h (15.9%) |
| Distinct entry edges | 33; the top 10 carry 85% of demand |

The two dominant entries are the arterial ends; the long tail is side streets.
That is exactly the quantity needed to calibrate the synthetic corridor, and it
comes from real mobility data with **zero acquisition risk and full
reproducibility** — any reader with `sumo-rl` installed can recompute these four
numbers.

This is strictly better than a download: an external CSV would have to be
vendored into the repo and trusted, whereas this derives from a dependency the
project already pins. The resulting calibration is recorded in
`corridor/generate_corridor_routes.py` with the arithmetic shown.

### Q3 — is `cologne3` acceptable as *the* real-data component: YES

Being bundled with a dependency is a strength, not a weakness. It is the
published RESCO benchmark (Ault & Sharon, NeurIPS 2021 D&B), derived from the
TAPAS Cologne mobility model, and it is used unmodified. Under Q2 it now does
**three** jobs rather than two: real-data calibration source, zero-shot
topology-transfer target, and the real-demand evaluation scenario itself.

**The honest limit, which the report will state:** `cologne3` calibrates the
*volumes and the arterial/side-street split*. It does not give us turning
fractions for our synthetic geometry, so the existing 60/20/20 straight/left/
right split is carried over from the single-intersection study unchanged. That
is an assumption inherited, not measured.

### Q4 — MA2C without the LSTM in v1: CONFIRMED

Omitted, as flagged in §3. The neighbour-observation and neighbour-policy
sharing plus the spatially discounted reward are the parts that make MA2C
*multi-agent*; the LSTM addresses partial observability within an agent and is
orthogonal to the coordination question this study asks. Recurrent state would
also interact with the 5 s decision interval and the episode-boundary reset in
ways that need their own tests to trust.

**This is a deviation from the published algorithm and will be labelled as such
in `CORRIDOR_REPORT.md`**, not buried. If MA2C underperforms, the missing LSTM
is a candidate explanation and the report will say so rather than concluding
that MA2C does not work.

### Calibration actually adopted

| Scenario | Arterial (each end) | Side street (each of 6) | Total | Role |
|---|---|---|---|---|
| **`calibrated`** | 500 veh/h | 200 veh/h | 2,200 veh/h | **Training scenario**; derived from cologne3 above |
| `balanced` | 350 veh/h | 350 veh/h | 2,800 veh/h | Symmetric — the case fixed-time is designed for |
| `peak` | 650 veh/h | 260 veh/h | 2,860 veh/h | `calibrated` × 1.3 |
| `reverse` | 200 veh/h | 500 veh/h | 3,400 veh/h | Demand shift: side streets dominate |

`calibrated`'s 2,200 veh/h sits just under cologne3's real 2,856 veh/h, and the
arterial kerb lane runs at v/c ≈ 0.73 against the 550 veh/h capacity the
single-intersection study derived — busy enough to separate controllers, below
the spillback regime that would invalidate the measurement.
