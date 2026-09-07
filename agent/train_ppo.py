"""
Train a PPO agent on the same traffic signal, for comparison with the DQN.

Run:
    python agent/train_ppo.py                    # default 170 episodes (~45 min)
    python agent/train_ppo.py --episodes 400

Outputs:
    agent/models/ppo.zip
    results/logs/training_curve_ppo.csv

WHY COMPARE TWO ALGORITHMS
--------------------------
DQN and PPO come from the two main families of reinforcement learning, and they
fail in different ways, so running both is a check on the result rather than a
horse race:

  DQN is *off-policy*. It stores every experience in a replay buffer and reuses
  it many times, which makes it economical with simulation time - valuable here,
  because running SUMO is far slower than updating the network.

  PPO is *on-policy*. It must throw away its experience after each update and
  collect fresh data, so it needs more simulated hours to reach the same place.
  In exchange its updates are more stable, because it explicitly limits how far
  the policy may move in one step.

If both independently land near the same performance, that is good evidence the
result reflects the traffic problem rather than one algorithm's quirks. If PPO
lags at an equal episode budget, the likely explanation is sample efficiency,
not that the policy is wrong - and that is worth saying plainly in the report
rather than presenting it as a defeat for PPO.

Both agents get an identical budget, environment, reward and discount factor so
the comparison means something.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO  # noqa: E402

from agent.training import STEPS_PER_EPISODE, train  # noqa: E402

DEFAULT_EPISODES = 170  # matched to the DQN budget so the comparison is fair


def build_model(env, seed: int) -> PPO:
    """Construct PPO with hyperparameters chosen for this problem."""
    return PPO(
        "MlpPolicy",
        env,
        # Collect exactly two full episodes before each update. Aligning the
        # rollout with the episode boundary means every update sees complete
        # signal cycles rather than a fragment cut off mid-queue.
        n_steps=STEPS_PER_EPISODE * 2,
        batch_size=180,        # divides 1440 evenly, so no partial minibatch
        n_epochs=10,           # passes over each batch of collected experience

        learning_rate=3e-4,    # the standard PPO default; stable for a small MLP

        # Same horizon as the DQN (~100 s, about one signal cycle) so the two
        # agents are optimising the same objective.
        gamma=0.95,
        gae_lambda=0.95,

        # PPO's defining trick: refuse to move the policy more than 20% per
        # update, which prevents one unlucky batch from destroying a good policy.
        clip_range=0.2,

        # A small bonus for keeping the action distribution uncertain. Without
        # it PPO tends to commit early to a two-phase cycle and stop exploring
        # the left-turn phases at all.
        ent_coef=0.01,

        policy_kwargs=dict(net_arch=[64, 64]),  # same network size as the DQN
        seed=seed,
        verbose=0,
        device="cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="training episodes, one simulated hour each "
                             "(default: %d)" % DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility (default: 42)")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for the model/curve filenames")
    args = parser.parse_args()

    print("Each episode is %d decisions; budget = %d timesteps.\n"
          % (STEPS_PER_EPISODE, args.episodes * STEPS_PER_EPISODE))
    train("ppo", build_model, args.episodes, args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
