"""
Train a Deep Q-Network to control the traffic signal.

Run:
    python agent/train_dqn.py                    # default 170 episodes (~35 min on a laptop CPU)
    python agent/train_dqn.py --episodes 400     # longer run, better policy
    python agent/train_dqn.py --episodes 20      # quick smoke run

Outputs:
    agent/models/dqn.zip                         the trained policy
    results/logs/training_curve_dqn.csv          per-episode return, for the plot

WHAT IS A DQN, BRIEFLY
----------------------
The agent needs to answer one question at every decision point: "if I show
phase X now, how much total future queueing will follow?"  A DQN is a small
neural network that estimates exactly that number - the "Q-value" - for each of
the four phases, and the agent simply picks the phase with the best estimate.

The network starts out knowing nothing, so early on it acts almost at random
(this is the "exploration" phase).  Every decision and its outcome is stored in
a replay buffer, and the network is repeatedly retrained on random samples from
that buffer so its predictions match what actually happened.  Over time the
estimates get accurate, and acting greedily on accurate estimates is a good
control policy.

We use the implementation from stable-baselines3 rather than writing our own,
because a hand-rolled DQN is where subtle bugs hide and none of the project's
questions are about the algorithm's internals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import DQN  # noqa: E402

from agent.training import STEPS_PER_EPISODE, train  # noqa: E402

# Default training budget.
#
# Chosen by watching the learning curve rather than picked in advance: the
# return improves steeply for roughly the first 60 episodes, keeps creeping
# down until about 140, and is flat after that. 170 episodes leaves a clearly
# visible plateau at the end of the plot, which is the evidence we need that
# training had converged rather than being cut off early.
DEFAULT_EPISODES = 170


def build_model(env, seed: int) -> DQN:
    """Construct the DQN with hyperparameters chosen for this problem."""
    return DQN(
        "MlpPolicy",
        env,
        # --- how fast the network updates ---------------------------------
        # 5e-4 is a middle-of-the-road value for a small MLP. Much higher and
        # the Q-estimates chase noise from individual lucky episodes; much
        # lower and 170 episodes is not enough to converge.
        learning_rate=5e-4,

        # --- memory of past experience ------------------------------------
        # The replay buffer holds 100k transitions, more than a whole training
        # run produces, so nothing is forgotten. Training on random samples
        # from it (rather than on consecutive steps) breaks the correlation
        # between successive traffic states, which is what keeps the network
        # from oscillating.
        buffer_size=100_000,

        # Collect ~7 episodes of experience before the first gradient step, so
        # the first updates are not fitted to a nearly empty buffer.
        learning_starts=5_000,
        batch_size=64,

        # --- how far ahead the agent plans --------------------------------
        # gamma=0.95 with a 5 s decision interval means the agent effectively
        # cares about the next 100 seconds or so - roughly one signal cycle.
        # That is the right horizon: long enough to see that starving a
        # direction now creates a bigger queue later, short enough that the
        # value estimates stay learnable.
        gamma=0.95,

        # A gradient step after every environment step. The simulation, not the
        # network, is the bottleneck here (~12 ms per step versus ~1 ms per
        # update), so updating this often costs little wall-clock time and
        # extracts noticeably more from each simulated hour.
        train_freq=1,
        gradient_steps=1,

        # The target network is a frozen copy used to compute learning targets;
        # refreshing it only every 1000 steps stops the network from chasing
        # its own moving predictions.
        target_update_interval=1_000,

        # --- exploration ---------------------------------------------------
        # Start fully random, decay to 2% random over the first 35% of
        # training, then mostly exploit. The decay has to be slow enough that
        # the agent tries genuinely bad phase orderings early - that is how it
        # learns which ones are bad.
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,

        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=0,
        device="cpu",  # a 2-layer 64-unit MLP is slower on a GPU than on CPU
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="training episodes, one simulated hour each "
                             "(default: %d)" % DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for reproducibility (default: 42)")
    parser.add_argument("--tag", type=str, default="",
                        help="suffix for the model/curve filenames, for "
                             "running variants side by side")
    args = parser.parse_args()

    print("Each episode is %d decisions; budget = %d timesteps.\n"
          % (STEPS_PER_EPISODE, args.episodes * STEPS_PER_EPISODE))
    train("dqn", build_model, args.episodes, args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
