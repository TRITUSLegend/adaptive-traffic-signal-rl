"""
Train a DQN on MIXED demand, to test whether varied training removes the
directional bias the single-scenario agent picked up.

Run:
    python agent/train_mixed.py                  # 170 episodes, ~46 min
    python agent/train_mixed.py --episodes 20    # quick check that it runs

Outputs:
    agent/models/dqn_mixed.zip
    results/logs/training_curve_dqn_mixed.csv

The original `agent/models/dqn.zip` is NOT touched - the two are meant to be
compared.

THE PROBLEM THIS ADDRESSES
--------------------------
REPORT.md section 4.4 reports the clearest weakness in the results. The DQN was
trained only on `ns_heavy` (600 veh/h north-south against 300 east-west), and
on `ew_heavy` - the identical demand mirrored - it does about 21% worse: 24.9 s
against 20.6 s. The greedy heuristic, which has no learned preference, is
unaffected (22.3 s versus 22.2 s). The agent has absorbed a preference for
north-south that does not transfer.

The obvious remedy is to train on more than one demand pattern, which is what
this script does.

WHAT IT TRAINS ON, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
`balanced`, `ns_heavy` and `peak` - the three evaluation scenarios - over the
same ten training seeds each, interleaved so consecutive episodes differ in
demand rather than arriving in blocks.

`ew_heavy`, `surge` and `light` are excluded, and that exclusion is the whole
point of the experiment being worth running. `ew_heavy` is the test that
exposes the bias; training on it would not fix the bias, it would erase the
measurement. Keeping the robustness set unseen is what lets the result mean
something.

Be clear about what this can therefore show. Of the three training scenarios,
one is symmetric and two are north-south heavy, so the agent still never sees
east-west traffic dominate. The hypothesis under test is the modest one - that
exposure to VARIED demand reduces the learned directional preference - not that
the agent is shown the answer and repeats it. A partial improvement is the
honest expectation.

Everything else is held identical to `train_dqn.py`: same hyperparameters, same
budget, same reward, same seed. The only difference is which route files the
episodes are drawn from, so any change in behaviour is attributable to the
demand rather than to the setup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.train_dqn import DEFAULT_EPISODES, build_model  # noqa: E402
from agent.training import STEPS_PER_EPISODE, train  # noqa: E402
from networks.generate_routes import (  # noqa: E402
    MIXED_TRAINING_SCENARIOS,
    mixed_training_route_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="training episodes, one simulated hour each "
                             "(default: %d, matching the single-scenario agent "
                             "so the comparison is fair)" % DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42, as in train_dqn.py)")
    parser.add_argument("--tag", type=str, default="mixed",
                        help="suffix for the model and curve filenames "
                             "(default: mixed)")
    args = parser.parse_args()

    routes = mixed_training_route_files()
    missing = [p.name for p in routes if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing %d mixed-demand route file(s), e.g. %s.\n"
            "Generate them without touching any existing file:\n"
            "    python networks/generate_routes.py --mixed --only-missing"
            % (len(missing), missing[0]))

    print("Mixed-demand training over %s." % ", ".join(MIXED_TRAINING_SCENARIOS))
    print("Held out and never trained on: ew_heavy, surge, light.")
    print("Each episode is %d decisions; budget = %d timesteps.\n"
          % (STEPS_PER_EPISODE, args.episodes * STEPS_PER_EPISODE))

    train("dqn", build_model, args.episodes, args.seed, tag=args.tag,
          routes=routes, demand="+".join(MIXED_TRAINING_SCENARIOS))


if __name__ == "__main__":
    main()
