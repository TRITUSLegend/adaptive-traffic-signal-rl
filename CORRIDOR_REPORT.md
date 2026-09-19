# Adaptive Signal Control on a Multi-Intersection Corridor

A deep-reinforcement-learning study of coordinated traffic signal control along
a three-junction arterial, with fixed-time and actuated baselines, three
multi-agent learners, and a robustness study covering detector failure, demand
shift and zero-shot transfer to an unseen real-world topology.

**This document is additive.** It does not modify, restate or supersede
`REPORT.md`, the finished single-intersection study — that work, its numbers and
its addenda A–E are untouched, and `tests/test_corridor_isolation.py` hashes its
sources to keep them that way. Where this study and that one can be compared,
the comparison is drawn here.

---

> ## Read this before any number below
>
> **One training seed. No significance testing was performed. Every margin in
> this report is a point estimate that a future multi-seed run could revise.**
>
> The single-intersection study measured how much a result moves when only the
> training seed changes: between 0.18 s and 3.03 s of mean waiting time
> depending on the scenario (`REPORT.md`, Addendum B). Differences smaller than
> that are not distinguishable from initialisation noise, and nothing in this
> document should be read as though they were.
>
> This is a deliberate scope decision, not an oversight. The compute budget was
> measured before committing to it: three algorithms at three seeds would be
> roughly 35–45 hours on this machine against 13–17 for one, and one seed across
> three algorithms was judged more informative than three seeds of one.
> `CORRIDOR_PLAN.md` §6 records the reasoning.
>
> Rankings that hold across several scenarios are the trustworthy part of what
> follows. Individual margins are not.

---

## 1. What the brief asked for, and where this lands

| Requirement | Where it is met |
|---|---|
| Microscopic simulation (SUMO) | §2.1 — a generated 3-junction arterial |
| DQN with queue state and delay reward | §2.3, §4 — independent DQN |
| Policy-gradient method (PPO/A2C) | §2.3, §4 — independent PPO, and MA2C's actor-critic |
| Multi-agent RL (MA2C) for coordination | §2.3, §4 — written from scratch, §7 for the deviation |
| **Fixed-time baseline** | §2.3 — the same 104 s plan the original study used |
| **Actuated baseline** | §2.3 — SUMO's gap-based actuated control |
| Demand from real flow profiles | §2.2 — calibrated from `cologne3` (TAPAS Cologne) |
| Avg waiting time, queue, throughput, delay reduction | §5 |
| Convergence (reward curve) | §5.1 |
| **Robustness: detector failure** | §6.1–6.2 |
| **Robustness: demand shift** | §5.2 — the `reverse` scenario |
| **Robustness: unseen topology** | §6.3 — corridor2, corridor5 and real `cologne3` |

The brief is explicit that clean-conditions delay reduction is not the point —
"DQN/PPO-on-SUMO is heavily published already" — and that robustness is what
makes the topic worth doing. §6 is therefore the centre of this report, not an
appendix to §5.

---

## 2. What was built

### 2.1 The corridor

A linear west–east arterial with three signalised junctions, each also served by
a north and a south side street:

```
        N1          N2          N3
        |           |           |
  W ——[J1]——400m——[J2]——400m——[J3]—— E
        |           |           |
        S1          S2          S3
```

- **400 m between signals.** Long enough to hold a queue without spilling back
  into the upstream junction. Spillback would suppress demand outside the
  simulation and flatter every controller equally — the measurement failure the
  original study guards against with its insertion-backlog metric, which is
  reported here too.
- **250 m outer arms**, the length the single-intersection network uses.
- **2 lanes per approach**: lane 0 straight + right, lane 1 a protected left.
- **Four phases per junction**, mutually exclusive, so every phase is
  collision-free and there are no permissive give-way lefts.

The geometry is *tiled from the existing single-intersection network* rather
than drawn afresh, and the result is checkable: each junction's compiled phase
strings come out identical to the validated single-intersection plan —
`GGrrrrGGrrrr`, `rrGrrrrrGrrr`, `rrrGGrrrrGGr`, `rrrrrGrrrrrG`, on a 104 s
cycle. That identity is not decoration. It is what makes a junction here present
the same action space and movement structure the original agents trained on, and
therefore what makes transfer between the two studies meaningful.

**Three junctions, and why not more.** Three is the smallest corridor in which
MA2C is not degenerate: J2 has two signalised neighbours while J1 and J3 have
one, so the architecture's asymmetric neighbour sharing is actually exercised.
With two junctions every agent is an edge agent and "neighbour sharing" reduces
to "the other one".

### 2.2 The demand, and why not PeMS

**PeMS does not fit this problem, and the report does not pretend otherwise.**
It is *freeway* loop-detector data — mainline and ramp volumes at 30 s/5 min
aggregation — with no intersections, no turning movements and no urban arterial
geometry. Calibrating a signalised corridor from freeway mainline counts would
be measuring the wrong quantity on the wrong facility type. The brief names PeMS
as an example rather than a requirement, so an urban source is substituted and
the substitution is stated.

Demand is instead calibrated from **`cologne3`**, the RESCO benchmark corridor
(Ault & Sharon, NeurIPS 2021 Datasets & Benchmarks) derived from the TAPAS
Cologne mobility model and bundled with `sumo-rl`. Measured from its route file,
reproducible with `python corridor/generate_corridor_routes.py --provenance`:

| Quantity | Value |
|---|---|
| Busiest hour (06:31–07:59, real morning peak) | **2,856 veh/h** across three junctions |
| Dominant entry #1 | 550 veh/h (19.3% of demand) |
| Dominant entry #2 | 454 veh/h (15.9%) |
| Distinct entry edges | 33; the top 10 carry 85% |

The two dominant entries are the arterial ends and the long tail is side
streets — a heavy arterial fed by many light side roads. Rounded to 500 veh/h
per arterial end and 200 per side stub, that gives 2,200 veh/h, deliberately a
little under the real 2,856, with the arterial kerb lane at v/c ≈ 0.73 against
the 550 veh/h capacity the original study derived: loaded enough to separate
controllers, below the regime where spillback invalidates the measurement.

| Scenario | Arterial (each end) | Side (each of 6) | Total | Role |
|---|---|---|---|---|
| `calibrated` | 500 veh/h | 200 veh/h | 2,200 | **Training scenario** |
| `balanced` | 350 | 350 | 2,800 | Symmetric — what fixed-time is designed for |
| `peak` | 650 | 260 | 2,860 | `calibrated` × 1.3 |
| `reverse` | 200 | 500 | 3,400 | **Demand shift**: side streets dominate |

**Calibration, not replay**, for three reasons: the counts are aggregates rather
than vehicle-by-vehicle departures, so replay would invent the within-interval
arrival process anyway; a single replay is one sample with no variance estimate,
discarding the multi-seed discipline the whole project rests on; and the
demand-shift scenarios are the base profile *moved*, which a fixed recording
cannot provide.

**What is assumed rather than measured:** the 60/20/20 straight/left/right split
is inherited unchanged from the single-intersection study. `cologne3` gives
volumes, not turning fractions for our geometry. Arrivals at the corridor
boundary are Poisson, which is defensible at an entry point; mid-corridor
arrivals are platooned by the upstream signal, but that platooning is produced
by the simulation itself rather than by the generator, so the approximation is
confined to the boundary.

### 2.3 The controllers

| Controller | What it does | Who drives it |
|---|---|---|
| **Fixed-time** | The 104 s cycle, unchanged from the original study | SUMO |
| **Actuated** | Gap-based: green extends while vehicles keep arriving within `max-gap` 3.0 s, bounded by min 10 s / max 60 s | SUMO |
| **Greedy queue** | Serves whichever phase has the longest queue, per junction, every 5 s | us |
| **Independent DQN** | One policy per junction, local observation only | us |
| **Independent PPO** | Same, on-policy | us |
| **MA2C** | Actor-critic per junction with neighbour observation sharing, neighbour policy sharing and spatially discounted reward | us |

**The greedy heuristic is not an actuated controller and is not labelled as
one.** It picks a phase from global queue knowledge every 5 s; an actuated
controller runs a fixed phase *sequence* and decides only when to terminate the
current green, from detector gap-out. Relabelling one as the other would be the
kind of quiet overstatement this project has twice had to remove. The actuated
baseline is SUMO's own implementation, so it is standard and independently
specified rather than ours.

**A consequence to note before reading §6:** SUMO drives both fixed-time and
actuated itself, so their `act()` is never consulted and they cannot be
perturbed at the observation level at all. Their bounds come from the program's
own `minDur`/`maxDur`, pinned to the project's MIN_GREEN and MAX_GREEN, so every
controller here still pays an identical amber and minimum-green cost. §6.2
returns to what that structural immunity does and does not mean.

The maximum-green and anti-starvation wrapper from the single-intersection study
(`REPORT.md`, Addendum E) is available for corridor controllers and is covered by
tests, but **no `+safe` sweep was run here and none is reported**. Whether the
corridor's learned policies violate the starvation bound the way the
single-intersection ones did is an open question this study does not answer.

---

## 3. How the comparison is kept fair

The same four guarantees the single-intersection study rests on:

- **One code path.** All six controllers run through
  `corridor/env.py:run_corridor_episode`. They are not merely measured with the
  same metrics — they are measured by the same code.
- **The same vehicles.** A `(scenario, seed)` pair names a file listing exactly
  which car arrives when and where it is going. Every controller drives the
  identical hour.
- **The same safety constraints.** 4 s amber and 10 s minimum green for
  everything, the actuated plan included, whose bounds are pinned to the
  project's constants rather than to SUMO's defaults. Differing amber times
  would be the easiest way to rig a comparison.
- **Unseen demand.** Training used seeds 0–9; every number here is measured on
  seeds 100–104.

And one the corridor adds:

- **The same dead detectors.** In §6.1 the failed detector set is drawn from an
  RNG seeded by `(scenario, seed, rate, mode)` and by nothing identifying the
  controller, so every controller faces the identical fault on the identical
  hour. Without that, "degradation" would partly measure a difference in the
  perturbation.

**Throughput is never reported without waiting time beside it.** A controller
can make delay look excellent by refusing to admit vehicles; on a corridor it
can also do so by starving the link that feeds its neighbour. Every table below
carries throughput, and the insertion backlog is recorded per episode so that
suppressed demand is visible rather than silently improving a number.

---

## 4. Did the learners converge?

<!-- BEGIN GENERATED: convergence -->
| Run | Episodes | Mean queue, first 10 (veh) | Mean queue, last 10 (veh) | Change |
|---|---|---|---|---|
| Independent DQN | 170 | 110.6 | 16.1 | -85% |
| Independent PPO | 170 | 82.3 | 15.3 | -81% |
| MA2C | 170 | 118.8 | 26.1 | -78% |
<!-- END GENERATED: convergence -->

![Corridor training curves](results/corridor/plots/corridor_training_curves.png)

---

## 5. Results under clean conditions

### 5.1 Every controller, every scenario

<!-- BEGIN GENERATED: headline -->
| Controller | Scenario | Mean wait (s) | Queue (veh) | Throughput (veh/h) | Not inserted |
|---|---|---|---|---|---|
| Fixed-time | `calibrated` | 84.2 ± 7.6 | 51.4 | 2072 | 1 |
|   Actuated (gap-based) | `calibrated` | 34.3 ± 2.2 | 21.0 | 2114 | 1 |
|   Greedy queue heuristic | `calibrated` | 29.1 ± 1.7 | 17.7 | 2114 | 1 |
|   Independent DQN | `calibrated` | 25.1 ± 0.5 | 15.2 | 2120 | 1 |
|   Independent PPO | `calibrated` | 24.7 ± 0.6 | 15.1 | 2120 | 1 |
|   MA2C | `calibrated` | 232.4 ± 10.3 | 140.3 | 1860 | 22 |
| Fixed-time | `balanced` | 70.1 ± 7.3 | 55.4 | 2700 | 0 |
|   Actuated (gap-based) | `balanced` | 36.1 ± 1.0 | 28.6 | 2747 | 0 |
|   Greedy queue heuristic | `balanced` | 31.7 ± 1.5 | 25.0 | 2746 | 0 |
|   Independent DQN | `balanced` | 29.2 ± 1.3 | 23.1 | 2749 | 0 |
|   Independent PPO | `balanced` | 28.2 ± 1.0 | 22.3 | 2754 | 0 |
|   MA2C | `balanced` | 179.7 ± 8.5 | 137.4 | 2476 | 83 |
| Fixed-time | `peak` | 144.8 ± 8.8 | 109.0 | 2460 | 166 |
|   Actuated (gap-based) | `peak` | 41.7 ± 1.9 | 33.4 | 2764 | 0 |
|   Greedy queue heuristic | `peak` | 41.9 ± 1.3 | 33.5 | 2754 | 0 |
|   Independent DQN | `peak` | 32.7 ± 1.0 | 26.3 | 2773 | 0 |
|   Independent PPO | `peak` | 31.9 ± 0.9 | 25.6 | 2774 | 0 |
|   MA2C | `peak` | 233.2 ± 9.0 | 182.2 | 2446 | 62 |
| Fixed-time | `reverse` | 64.5 ± 13.0 | 61.6 | 3285 | 0 |
|   Actuated (gap-based) | `reverse` | 39.6 ± 1.5 | 37.8 | 3324 | 0 |
|   Greedy queue heuristic | `reverse` | 39.6 ± 4.8 | 37.8 | 3313 | 0 |
|   Independent DQN | `reverse` | 48.3 ± 19.4 | 46.3 | 3264 | 0 |
|   Independent PPO | `reverse` | 35.1 ± 3.7 | 33.6 | 3318 | 0 |
|   MA2C | `reverse` | 131.2 ± 7.7 | 119.6 | 3047 | 143 |
<!-- END GENERATED: headline -->

![Corridor comparison](results/corridor/plots/corridor_comparison.png)

### 5.2 Delay reduction against fixed-time

The brief's headline metric. Negative is an improvement.

<!-- BEGIN GENERATED: reduction -->
| Controller | `calibrated` | `balanced` | `peak` | `reverse` | Mean |
|---|---|---|---|---|---|
| Actuated (gap-based) | -59.2% | -48.4% | -71.2% | -38.5% | **-54.3%** |
| Greedy queue heuristic | -65.4% | -54.7% | -71.1% | -38.6% | **-57.5%** |
| Independent DQN | -70.2% | -58.4% | -77.4% | -25.1% | **-57.8%** |
| Independent PPO | -70.6% | -59.8% | -78.0% | -45.5% | **-63.5%** |
| MA2C | +175.9% | +156.3% | +61.0% | +103.5% | **+124.2%** |
<!-- END GENERATED: reduction -->

`reverse` is the demand-shift test: the side streets carry more than the
arterial the agents trained on favouring. It is the corridor analogue of the
`ew_heavy` test that exposed a 20.8% directional penalty in the
single-intersection study.

### 5.3 Throughput, so the reductions above can be read honestly

<!-- BEGIN GENERATED: throughput -->
| Controller | `calibrated` | `balanced` | `peak` | `reverse` |
|---|---|---|---|---|
| Fixed-time | 2072 | 2700 | 2460 | 3285 |
| Actuated (gap-based) | 2114 | 2747 | 2764 | 3324 |
| Greedy queue heuristic | 2114 | 2746 | 2754 | 3313 |
| Independent DQN | 2120 | 2749 | 2773 | 3264 |
| Independent PPO | 2120 | 2754 | 2774 | 3318 |
| MA2C | 1860 | 2476 | 2446 | 3047 |
<!-- END GENERATED: throughput -->

---

## 6. Robustness — the part the brief calls the point

### 6.1 Detector failure

A wrapper corrupts the observation before the controller sees it. **The
simulation is untouched**: the vehicles are still queueing and still crossing,
and every metric is still measured from the real simulation. The controller
simply cannot see some of them. That separation is what makes this a *sensing*
failure rather than a change in demand — deleting vehicles would change the
information and the problem at once, and the resulting number would answer
neither question.

Only the sixteen detector-derived values are corrupted (8 lane densities, 8 lane
queues). The phase one-hot and the minimum-green flag are the signal's own
state, not a measurement, so corrupting them would model a different fault. A
failed detector takes out both the density and the queue for its lane, because
those come from one physical sensor.

Two modes, reported separately because a controller can be robust to one and not
the other:

- **Stuck-at-zero** — the lane reports an empty road. The dangerous one: an
  approach that looks empty may never be served, so the fault actively misleads
  rather than merely withholding.
- **Freeze** — the lane holds its last good reading. Milder, and closer to a
  communications dropout than to a dead loop.

> **Which controllers this fault can reach, corrected by measurement.** The plan
> for this study predicted that fixed-time would be immune and that *actuated
> would degrade*, since gap-out depends on detectors. The data says otherwise:
> actuated reads **34.3 s at every failure rate**, exactly as flat as
> fixed-time.
>
> The reason is a limit of the perturbation, not a property of the controller.
> This wrapper corrupts **the controller's observation**, and SUMO drives
> actuated from **its own internal induction loops** — 19 of them, which
> `corridor/transfer.py` counts — which the wrapper never touches. So actuated
> is a null here *by construction*, in the same way fixed-time is, and its flat
> line is arithmetic rather than evidence of robustness.
>
> Reaching it would mean corrupting SUMO's own detectors, a different mechanism
> that is not built here. Until someone does, **this study says nothing about
> how gap-based actuated control behaves when its loops fail** — which is a
> shame, because that is the one baseline a practitioner would most want the
> answer for.

<!-- BEGIN GENERATED: detector -->
**Failure mode: stuck-at-zero**

| Controller | 0% dead | 12% dead | 25% dead | 50% dead | Change at 50% |
|---|---|---|---|---|---|
| Fixed-time | 84.2 | 84.2 | 84.2 | 84.2 | **+0%** |
| Actuated (gap-based) | 34.3 | 34.3 | 34.3 | 34.3 | **+0%** |
| Greedy queue heuristic | 29.1 | 33.2 | 174.1 | 348.8 | **+1098%** |
| Independent DQN | 25.1 | 33.1 | 184.3 | 560.7 | **+2136%** |
| Independent PPO | 24.7 | 32.3 | 222.5 | 332.0 | **+1242%** |
| MA2C | 232.4 | 251.8 | 269.8 | 307.3 | **+32%** |

**Failure mode: freeze (last value held)**

| Controller | 0% dead | 12% dead | 25% dead | 50% dead | Change at 50% |
|---|---|---|---|---|---|
| Fixed-time | 84.2 | 84.2 | 84.2 | 84.2 | **+0%** |
| Actuated (gap-based) | 34.3 | 34.3 | 34.3 | 34.3 | **+0%** |
| Greedy queue heuristic | 29.1 | 29.2 | 29.1 | 29.1 | **+0%** |
| Independent DQN | 25.1 | 25.0 | 25.2 | 24.6 | **-2%** |
| Independent PPO | 24.7 | 24.7 | 24.7 | 24.3 | **-2%** |
| MA2C | 232.4 | 231.3 | 231.3 | 230.1 | **-1%** |
<!-- END GENERATED: detector -->

![Detector failure](results/corridor/plots/corridor_detector_failure.png)

### 6.2 The margin over fixed-time, which is the actual question

**Fixed-time reads no detectors, so its degradation is zero by construction.**
That is not a virtue and must not be read as one — it cannot respond to demand
at all, which is why it is slow in the first place. It is the reference line,
the same way the inert `fixed-time-tuned+safe` row is arithmetic rather than
evidence in `REPORT.md`.

The question worth answering is therefore not how far an adaptive controller
falls, but whether it is still ahead of the clock when half its detectors are
dead. Negative means still faster than fixed-time.

<!-- BEGIN GENERATED: margin -->
| Controller | 0% dead | 12% dead | 25% dead | 50% dead |
|---|---|---|---|---|
| Actuated (gap-based) | -49.9 s | -49.9 s | -49.9 s | -49.9 s |
| Greedy queue heuristic | -55.1 s | -51.1 s | +89.8 s | +264.6 s |
| Independent DQN | -59.2 s | -51.2 s | +100.1 s | +476.5 s |
| Independent PPO | -59.5 s | -52.0 s | +138.2 s | +247.8 s |
| MA2C | +148.2 s | +167.5 s | +185.5 s | +223.1 s |
<!-- END GENERATED: margin -->

### 6.3 Zero-shot transfer to unseen topologies

Two kinds, which answer different questions and are not pooled.

**Clean transfer — `corridor2` and `corridor5`.** Built by the same generator,
so every junction has the same 4 phases and 8 lanes the policies trained on.
The observation projection is exactly a no-op here, so for the independent
learners the measured change is topology and nothing else: a different number
of junctions, a different neighbour structure, different interior-versus-edge
roles. This is the unconfounded measurement.

> **With one exception, which belongs to MA2C alone.** The independent learners
> share parameters, so one network drives a corridor of any length without
> adaptation. MA2C does not share parameters — it has a network *per junction* —
> so on `corridor5` its three trained agents are reused cyclically to cover five
> junctions. That is an approximation, and it means MA2C's `corridor5` number is
> not clean in the way the others are. Its `corridor2` number is, since two of
> its three agents simply go unused. This is the price of the specialisation
> that makes MA2C interesting, and it is the mirror image of the trade-off
> recorded in limitation 2.

**Real-topology transfer — `cologne3`.** Much stronger evidence, because we did
not draw it. But its junctions are structurally incompatible with ours:

| Junction | Green phases | Controlled lanes | Observation |
|---|---|---|---|
| `360082` | 3 | 5 | 14 values |
| `360086` | 4 | 6 | 17 values |
| `GS_cluster_…` | 4 | 8 | 21 values |

Our policies expect 4 phases and 8 lanes — 21 values. **Only one of the three
junctions fits.** So a projection is unavoidable: observations are zero-padded
or truncated to 21 values and actions are clamped into the phases a junction
actually has. Both are defined explicitly in `corridor/transfer.py` and tested
on hand-computable cases, and the projection is exactly a no-op on a corridor we
generated, which is what keeps the clean transfers clean.

> **A `cologne3` number mixes the difficulty of an unseen topology with the cost
> of the adaptation itself, and those two cannot be separated from that number
> alone.** The clean transfers exist to give an unconfounded reading beside it.

Two further caveats specific to this target. MA2C's agents are matched
positionally onto sorted junction ids, which is the true west-to-east order on a
corridor we built but an assumption on `cologne3`. And the actuated variant of
`cologne3` has detectors on 16 of its 19 controlled lanes — three lanes at the
cluster junction have none — so its actuation is partial there.

**The baselines need no adaptation at all.** SUMO drives fixed-time and actuated
from the target's own program, and the greedy heuristic reads whatever signal it
finds. Only the learned controllers are projected. That asymmetry is itself a
finding: it is a real deployment cost of a learned policy that a conventional
controller does not pay.

<!-- BEGIN GENERATED: transfer -->
| Controller | `corridor2` | `corridor5` | `cologne3` |
|---|---|---|---|
| Fixed-time | 58.1 s / 1740 veh | 172.9 s / 2611 veh | 22.8 s / 2856 veh |
| Actuated (gap-based) | 28.0 s / 1763 veh | 43.3 s / 2860 veh | 25.2 s / 2856 veh |
| Greedy queue heuristic | 23.3 s / 1761 veh | 43.0 s / 2849 veh | 307.3 s / 2819 veh |
| Independent DQN | 20.7 s / 1767 veh | 32.2 s / 2872 veh | 13012.5 s / 706 veh |
| Independent PPO | 20.5 s / 1768 veh | 32.4 s / 2873 veh | 15896.7 s / 504 veh |
| MA2C | 250.1 s / 1548 veh | 243.5 s / 2514 veh | 54.1 s / 2852 veh |
<!-- END GENERATED: transfer -->

![Topology transfer](results/corridor/plots/corridor_transfer.png)

---

## 6.4 MA2C: what the headline number is, and what it is not

MA2C is bottom of every table above, by a wide margin. That ranking is real and
is not softened here — but reporting it without the following would be
misleading, because most of the gap is **an interaction between this policy and
the evaluation protocol**, not a measure of how well it learned.

**It did learn.** Training went from 118.8 to 26.1 mean queue over 170
episodes, a curve indistinguishable in shape from the other two (§4). The
policy it arrived at is *soft*: mean entropy 1.197 against 1.386 for a uniform
policy over four phases.

**The protocol takes each agent's most probable phase.** That is what every
other controller here is held to, and for a sharp policy it is the right
reading. For this one it is destructive. Measured over 2000 states, MA2C's
argmax was phase 0 on 46% of them and phase 2 on 54% — **and phases 1 and 3 on
none**. Both left-turn movements are therefore never requested at all.

The traces show exactly that happening, which is why the signal-state column
exists:

<!-- BEGIN GENERATED: phase-service -->
| Controller | Green phases ever shown | Longest green (s) | Longest a phase went unserved (s) |
|---|---|---|---|
| Fixed-time | 4 of 4 | 35 | 95 |
| Actuated (gap-based) | 4 of 4 | 60 | 100 |
| Greedy queue heuristic | 4 of 4 | 65 | 320 |
| Independent DQN | 4 of 4 | 65 | 325 |
| Independent PPO | 4 of 4 | 55 | 450 |
| MA2C | 2 of 4 | 160 | 1760 |

Measured on `calibrated`, worst junction of 3, one evaluation hour each. For reference the single-intersection study's bounds are a 60 s maximum green and a 200 s starvation bound.
<!-- END GENERATED: phase-service -->

MA2C under the deterministic protocol shows **2 of 4 phases** and leaves one
unserved for **1760 s** — twenty-nine minutes. Nothing in a mean waiting time
would reveal that on its own; a starved approach carries too few vehicles to
move the average, which is the same blind spot Addendum E of the
single-intersection study had to reason around.

Reading the *same trained policy* by sampling instead — which is what it did
throughout training — gives a different picture:

<!-- BEGIN GENERATED: ma2c-protocol -->
| Action rule | `calibrated` | `balanced` | `peak` | `reverse` | Throughput |
|---|---|---|---|---|---|
| Deterministic (argmax) - the protocol used for every other controller | 232.4 | 179.7 | 233.2 | 131.2 | 2457 |
| Sampled from the policy - what it did while training | 42.4 | 46.6 | 52.1 | 62.3 | 2719 |
| *Fixed-time, for reference* | 84.2 | 70.1 | 144.8 | 64.5 | 2629 |
<!-- END GENERATED: ma2c-protocol -->

**Both numbers are reported because they answer different questions.** The
deterministic row is "how good is this policy's mode", and it is the one
comparable with the other five controllers. The sampled row is "how good is
this policy", and it is the one that reflects what was trained. Choosing only
the flattering one would be dishonest; choosing only the harsh one would blame
MA2C for a reading it never optimised for; and quietly switching protocol for a
single controller would break the comparison everything else rests on.

**What this does not excuse.** Even read generously, MA2C does not beat the
independent learners, and the plan predicted it might not — the implementation
omits the LSTM of the published algorithm (§7.3), and a policy whose competence
lives in its sampling noise is a weaker result than one whose mode is good. The
honest summary is that MA2C *worked*, *coordinated no better than agents that
never communicated*, and *is additionally fragile to a protocol the others
tolerate*.

### A finding that outlived MA2C

The phase-service table above was built to explain MA2C and ended up indicting
almost everything. Against the project's 200 s starvation bound, **every
adaptive controller here violates it**: the greedy heuristic at 320 s,
Independent DQN at 325 s, Independent PPO at 450 s. Only the two SUMO-driven
baselines stay inside — fixed-time at 95 s and actuated at 100 s — and they do
so because a cycle that serves every phase in turn structurally cannot starve
one.

This reproduces at corridor scale what Addendum E found at a single junction,
and it sharpens it: the learned controllers that *win* on mean waiting time are
the same ones that quietly leave a movement unserved for five to seven minutes.
The `+safe` wrapper from the single-intersection study would bound this, and is
implemented and tested, but **was not swept here** — so this report measures the
violation without offering the remedy.

---

## 7. Limitations

Stated as plainly as the single-intersection study states its own.

1. **One training seed, and no significance testing.** Repeated from the top
   because it conditions everything: every margin here is a point estimate.
   Addendum B of `REPORT.md` measured seed-to-seed spread at 0.18–3.03 s in the
   single-intersection case, and there is no reason to expect the corridor to be
   tighter. Rankings that hold across several scenarios are the trustworthy
   part; individual differences of a second or two are not.
2. **The independent learners share parameters.** `CORRIDOR_PLAN.md` §3 proposed
   one network per junction. What is implemented is one network trained on all
   three junctions' experience, with each junction still acting independently on
   its own local observation. The reason is that topology transfer is otherwise
   undefined — there is no principled answer to which of three policies drives
   junction 4 of a five-junction corridor. The cost is real: the three junctions
   occupy genuinely different roles and a shared network cannot specialise to
   them. MA2C, whose agents do not share parameters, is the contrast.
3. **MA2C has no LSTM.** The published algorithm (Chu et al., IEEE T-ITS 2020)
   puts a recurrent layer in each agent; v1 here does not, and
   `CORRIDOR_PLAN.md` §10 records the decision. If MA2C underperforms the
   simpler independent learners, the missing LSTM is a live candidate
   explanation and this report does not conclude that MA2C does not work.
4. **The demand calibration is partial.** `cologne3` supplies volumes and the
   arterial/side split. Turning fractions are inherited from the
   single-intersection study, not measured.
5. **One corridor geometry.** Uniform 400 m spacing, identical junctions,
   symmetric side streets. Real arterials are none of those things, which is
   part of why `cologne3` is included despite needing a projection.
6. **No signal coordination baseline.** A green-wave offset plan is the
   conventional engineering answer to a corridor and would be the natural
   further baseline. It is not implemented here.
7. **Detector failure is modelled at the observation, not the sensor.** Real
   failures have correlated causes — a cabinet loses power and takes an approach
   with it — whereas lanes here fail independently.
8. **The demand-shift study covers two of the four shifts planned.**
   `CORRIDOR_PLAN.md` §5b listed volume scaling, directional reversal, a
   temporal within-hour peak, and a mid-episode incident. Volume scaling
   (`peak`) and directional reversal (`reverse`) are implemented and reported;
   the temporal and incident shifts were dropped under the plan's own compute
   fallback and are not claimed. The incident case is the one most worth adding
   next, because it is the most realistic and the only one that tests recovery
   rather than a steady state.

---

## 8. Relationship to the single-intersection study

Nothing in `REPORT.md` is restated, revised or superseded by this document. The
two studies share a reward function, a decision interval, amber and minimum
green, an observation layout and an evaluation discipline — the corridor imports
those constants rather than copying them, so they cannot drift apart, and
`tests/test_corridor_isolation.py` fails if any single-intersection source or
`REPORT.md` itself is modified.

What differs is the problem: three coupled junctions where one junction's
decisions become its neighbour's arrivals, two baselines instead of one, and a
robustness study that the original explicitly listed as missing.

---

## 9. Reproducing this

```bash
python corridor/build_corridor.py --all              # corridor 2, 3 and 5
python corridor/generate_corridor_routes.py --all    # 70 route files
python corridor/train_independent.py --algo dqn      # ~3-5 h
python corridor/train_independent.py --algo ppo      # ~3-5 h
python corridor/train_ma2c.py                        # ~3-5 h
python corridor/evaluate_corridor.py                 # the clean sweep
python corridor/robustness.py --all                  # failure + transfer
python corridor/make_corridor_tables.py --write      # this report's tables
python corridor/make_corridor_plots.py               # its figures
python -m pytest                                     # the whole suite
```

Training seeds are fixed. Traffic simulation is stochastic, so every reported
cell is the mean of **5 independent hours** with the standard deviation beside
it — but that is variation between *hours of traffic*, not between trained
policies, which is the quantity a single training seed leaves unmeasured.
