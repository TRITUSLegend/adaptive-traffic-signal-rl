# Adaptive Traffic Signal Control Using Deep Reinforcement Learning

A reinforcement-learning agent that controls the traffic lights at a four-way
intersection in the [SUMO](https://eclipse.dev/sumo/) traffic simulator, and a
measurement harness that compares it against conventional fixed-time signal
control.

**Headline result:** across six demand scenarios the learned controller cut
mean waiting time per vehicle by **33-61%** against a standard fixed-time plan
(60.7% on the demand it was trained on), while slightly *increasing* the number
of vehicles that get through the junction.

Two caveats that the report deals with properly: a trivial "serve the longest
queue" heuristic captures most of that gain, with the learned agent pulling
clearly ahead only when the junction is saturated; and the fixed-time baseline's
own numbers are flattered by vehicles it was too congested to admit. Full
numbers, method and limitations are in **[REPORT.md](REPORT.md)**.

---

## What is actually going on here

A normal traffic light runs a fixed timing plan: 32 seconds of green for
north-south, then 32 for east-west, around and around, no matter what traffic
is actually waiting. It cannot know that the side street is empty while twenty
cars queue on the main road.

This project replaces that stopwatch with an agent that looks at the queues on
each approach every 5 seconds and picks which movement to give green to next.
Nobody tells it any traffic engineering. It is given a score - minus the number
of vehicles currently stopped - and it learns, over about 170 simulated hours,
which decisions make that score less bad.

The interesting question is not "can it beat a badly timed signal" (almost
anything can), so the project also measures it against a signal retimed
properly for the demand using Webster's method, and against a simple
"serve whoever has the longest queue" heuristic.

---

## Setup

Requires **Python 3.10+**. SUMO does *not* need to be installed separately -
it comes from PyPI as part of the requirements.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Optional but recommended: the small CPU-only PyTorch (~200 MB not ~2.5 GB).
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

Then confirm the toolchain works:

```bash
python smoke_test.py
```

It should end with `ALL CHECKS PASSED`. If it does not, it names the step that
failed and what to do about it.

> **Already have SUMO installed?** Set the `SUMO_HOME` environment variable and
> it will be used in preference to the pip-installed copy.

---

## Running the project

Every step is a standalone script. Run them in this order for a fresh clone.

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python networks/build_network.py` | Builds the intersection and its 4-phase signal plan | seconds |
| 2 | `python networks/generate_routes.py --mixed` | Generates 60 hours of random traffic across 6 demand scenarios | seconds |
| 3 | `python smoke_test.py` | Verifies SUMO, the packages, and a live simulation | ~10 s |
| 4 | `python baseline/run_baseline.py` | Runs the fixed-time baselines and reports their metrics | ~5 min |
| 5 | `python agent/train_dqn.py` | Trains the DQN agent | ~46 min |
| 6 | `python agent/train_ppo.py` | Trains the PPO agent (optional comparison) | ~26 min |
| 7 | `python eval/evaluate.py` | Runs every controller on every scenario, 5 seeds each | ~26 min |
| 8 | `python eval/make_plots.py` | Draws the figures used in the report | seconds |
| 9 | `python eval/make_tables.py --write` | Regenerates the results tables in REPORT.md | seconds |
| 10 | `python -m pytest` | Runs the test suite | ~2 s |

Steps 1, 2, 8, 9 and 10 are cheap; steps 5-7 are the long ones.

### The follow-up experiments (addenda A-E)

The report's original findings are steps 1-9. Five further experiments were run
afterwards, each aimed at a weakness the report itself named; they are appended
to REPORT.md as addenda rather than folded into it. Reproducing them costs
about another three hours:

```bash
python agent/train_dqn.py --seed 1 --tag seed1     # a second training seed  (~50 min)
python agent/train_dqn.py --seed 2 --tag seed2     # a third                 (~50 min)
python agent/train_mixed.py                        # mixed-demand training   (~50 min)
python agent/train_ppo.py --episodes 400 --tag long  # PPO, longer budget    (~60 min)

python eval/evaluate.py --controllers dqn_seed1 dqn_seed2 dqn_mixed ppo_long --append
python eval/evaluate.py --controllers dqn+safe ppo+safe greedy-queue+safe                                      fixed-time-tuned+safe --append

python eval/significance.py --write                # addendum A
python eval/followups.py --write                   # addenda B-E
```

`--append` merges the new controllers into `evaluation_raw.csv` instead of
replacing it, so the published 150-episode sweep is not re-run. A `+safe`
suffix on any controller wraps it in the maximum-green and anti-starvation
bounds, which is how the cost of the safety layer is measured rather than
asserted.

Useful flags:

```bash
python agent/train_dqn.py --episodes 400        # train longer for a better policy
python agent/train_dqn.py --episodes 20         # quick check that it runs
python eval/evaluate.py --scenarios ns_heavy    # evaluate one scenario only
python eval/evaluate.py --controllers fixed-time dqn
python eval/evaluate.py --seeds 100 101         # fewer repeats, faster
python networks/generate_routes.py --list       # show the demand scenarios
python baseline/fixed_time.py                   # print the Webster-tuned plans
```

Every script supports `--help`.

---

## Repository layout

```
networks/          The intersection and the traffic
  nodes / edges / connections .xml   hand-written geometry (2 lanes per approach)
  build_network.py                   compiles it and writes the 4-phase signal plan
  generate_routes.py                 generates Poisson vehicle arrivals per scenario
  routes/                            60 generated demand files (40 for the main
                                     study, 20 more for the mixed-demand agent)

baseline/          Non-adaptive controllers
  fixed_time.py                      untuned + Webster-tuned fixed-cycle signals
  run_baseline.py                    standalone baseline runner

agent/             The learned controllers
  training.py                        shared training loop, route rotation, logging
  train_dqn.py                       DQN
  train_ppo.py                       PPO
  train_mixed.py                     DQN on three demand patterns (addendum C)
  models/                            saved policies (.zip)

eval/              Measurement
  simulation.py                      the one episode runner every controller uses,
                                     plus the +safe constraint wrapper
  evaluate.py                        the comparison sweep
  paths.py                           the one place output filenames are decided
  make_plots.py                      figures
  make_tables.py                     generates the results tables in REPORT.md
  significance.py                    paired t and exact Wilcoxon (addendum A)
  followups.py                       generates addenda B-E from the same data

tests/             276 tests, run with `python -m pytest`
                   (the counts quoted in the docs are checked by the suite)

results/           Outputs
  logs/                              per-episode CSVs, summary tables, traces
  plots/                             the figures in REPORT.md
  tripinfo/                          raw SUMO per-vehicle trip reports

sumo_setup.py      Finds SUMO; imported before any SUMO import
smoke_test.py      End-to-end toolchain check
pyproject.toml     Python floor, pytest and ruff configuration
PROJECT_CONTEXT.md Full project state: status, remaining work, ground rules
REPORT.md          Findings, method, limitations, and addenda A-E
VERIFICATION.md    Evidence that the audited defects were fixed
docs/PROGRESS.md   Build log and design decisions
docs/REMEDIATION_LOG.md            Working log of the post-audit remediation
```

---

## How the comparison is kept fair

This matters more than any individual number, so it is worth stating plainly:

- **Same code path.** Fixed-time, the heuristic, and both RL agents are all run
  through one function (`eval/simulation.py:run_episode`). They are not just
  measured with the same metrics, they are measured by the same code.
- **Same vehicles.** A scenario and seed together name a file that lists
  exactly which car arrives when, and where it is going. Every controller
  drives the identical hour of traffic.
- **Same safety constraints.** Identical 4-second amber and 10-second minimum
  green for all controllers. Differing amber times would be the easiest way to
  accidentally rig the result.
- **Unseen test data.** The agent trains on demand seeds 0-9 and is evaluated
  on seeds 100-104. There is no overlap.
- **A strong baseline.** The untuned fixed-time plan is the one the brief asks
  for, but a Webster-tuned plan - given the demand in advance, which the RL
  agent never gets - is also reported.
- **Throughput is always reported.** A controller can make waiting time look
  wonderful by simply refusing to let cars into the junction, so waiting time
  is never shown without the throughput next to it.

## Tests

```bash
python -m pytest          # the test suite
python -m ruff check .    # lint; the rule set is pinned in pyproject.toml
```

The suite is fast (about three seconds) and is aimed at the places where a bug
would be *quiet* rather than loud - a number that is wrong but plausible:

- **Route determinism.** The seed must be a fixed function of scenario and
  seed number. It once was not, and the same command produced different
  traffic on every run.
- **Webster's method.** Allocated greens must fit inside the cycle they were
  computed for, which is the invariant the original closed-form version broke.
- **Report-to-data provenance.** Every waiting-time cell in REPORT.md is parsed
  back out of the markdown and checked against the CSVs, so the report cannot
  drift away from the data it describes.
- **Trip parsing.** Vehicles still queueing when the hour ends must be counted,
  or every controller looks better than it is and the congested ones most of all.
- **The statistics.** The hand-rolled t and Wilcoxon distributions are checked
  against published critical values and hand-counted cases.
- **The safety wrapper.** A phase may not be held past the maximum green or go
  unserved past the starvation bound, whatever the wrapped policy asks for.

`VERIFICATION.md` records the evidence, including a mutation check in which
deliberate breaks were applied to a copy of the repository to confirm the tests
actually catch them.

---

## Reproducibility

Random seeds are fixed for training (`--seed`, default 42), for vehicle
generation, and for SUMO itself. Even so, **traffic simulation is stochastic**:
two hours with identical average demand can differ by several seconds of mean
waiting time by chance alone. Never draw a conclusion from a single run. The
evaluation harness therefore averages **5 independent hours per controller per
scenario** and reports the standard deviation across them, so a difference can
be weighed against the noise it has to beat.
