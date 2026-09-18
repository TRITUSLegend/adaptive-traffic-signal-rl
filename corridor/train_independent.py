"""
Train independent learners (IDQN / IPPO) on the corridor.

Run:
    python corridor/train_independent.py --algo dqn --episodes 170
    python corridor/train_independent.py --algo ppo --episodes 170
    python corridor/train_independent.py --algo dqn --episodes 3 --smoke

Output:
    corridor/models/idqn.zip  /  ippo.zip
    results/corridor/logs/training_curve_idqn.csv  /  _ippo.csv

WHAT "INDEPENDENT" MEANS HERE
-----------------------------
Each junction observes only its own approaches and chooses only its own phase,
with no communication - that is the defining property of independent learners,
and it is what MA2C is built to improve on.

The networks' PARAMETERS are shared across junctions: one policy is trained on
the experience of all three.  CORRIDOR_PLAN.md proposed one network per
junction; corridor/vec_env.py documents why this changed, the short version
being that topology transfer is undefined otherwise - there is no principled
answer to which of three policies drives junction 4 of a five-junction
corridor.  The deviation is flagged in CORRIDOR_REPORT.md rather than left for
a reader to infer.

HYPERPARAMETERS
---------------
Copied from agent/train_dqn.py and agent/train_ppo.py unchanged wherever they
transfer, so that a difference between the single-intersection and corridor
results is a difference in the PROBLEM rather than in the tuning.  gamma stays
0.95 - about a 100 s horizon, roughly one signal cycle.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sumo_setup  # noqa: E402,F401

from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from corridor import paths  # noqa: E402
from corridor.generate_corridor_routes import TRAINING_SCENARIO  # noqa: E402
from corridor.vec_env import make_training_vec_env  # noqa: E402

STEPS_PER_EPISODE = 720          # 3600 simulated seconds / 5 s per decision
DEFAULT_EPISODES = 170           # matches the single-intersection budget


class CurveCallback(BaseCallback):
    """
    Record one row per finished episode and flush it to disk as we go.

    Flushing every ten episodes means an interrupted run still leaves usable
    evidence - the same habit the single-intersection training loop has, and
    the reason a killed run there was still worth something.
    """

    def __init__(self, env, path: Path, label: str, verbose: int = 0):
        super().__init__(verbose)
        self.env = env
        self.path = path
        self.label = label
        self.started = time.time()
        self._written = 0

    def _on_step(self) -> bool:
        done = len(self.env.episode_returns)
        if done > self._written:
            for i in range(self._written, done):
                minutes = (time.time() - self.started) / 60.0
                if (i + 1) % 10 == 0 or i == 0:
                    print("  %s episode %4d | return %10.1f | mean queue %6.1f veh "
                          "| %5.1f min elapsed"
                          % (self.label, i + 1, self.env.episode_returns[i],
                             self.env.episode_queues[i], minutes), flush=True)
            self._written = done
            if done % 10 == 0:
                self._flush()
        return True

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["episode", "return", "mean_queue_veh"])
            for i, (ret, queue) in enumerate(zip(self.env.episode_returns,
                                                 self.env.episode_queues), 1):
                writer.writerow([i, round(ret, 3), round(queue, 4)])

    def close(self) -> None:
        self._flush()


def build_model(algo: str, env, seed: int):
    if algo == "dqn":
        return DQN(
            "MlpPolicy", env,
            learning_rate=5e-4,
            buffer_size=100_000,
            learning_starts=5_000,
            batch_size=64,
            gamma=0.95,
            train_freq=1,
            target_update_interval=1_000,
            exploration_fraction=0.35,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            policy_kwargs=dict(net_arch=[64, 64]),
            seed=seed,
            verbose=0,
        )
    if algo == "ppo":
        return PPO(
            "MlpPolicy", env,
            # With three agents stepping together, n_steps counts steps per
            # agent, so this is two episodes' worth of corridor experience.
            n_steps=STEPS_PER_EPISODE * 2,
            batch_size=180,
            learning_rate=3e-4,
            gamma=0.95,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[64, 64]),
            seed=seed,
            verbose=0,
        )
    raise SystemExit("unknown algorithm %r; choose dqn or ppo" % algo)


def train(algo: str, episodes: int, seed: int, n_junctions: int, tag: str = "",
          scenario: str = TRAINING_SCENARIO):
    label = "i" + algo
    env = make_training_vec_env(n_junctions=n_junctions, scenario=scenario)
    model = build_model(algo, env, seed)

    curve = paths.training_curve(label, tag)
    callback = CurveCallback(env, curve, label)

    # VecEnv counts one timestep per agent per step, so a corridor episode is
    # STEPS_PER_EPISODE * n_junctions timesteps.
    total = episodes * STEPS_PER_EPISODE * n_junctions
    print("Training %s for %d episodes (%d timesteps across %d junctions)"
          % (label, episodes, total, n_junctions))
    started = time.time()
    try:
        model.learn(total_timesteps=total, callback=callback, progress_bar=False)
    except KeyboardInterrupt:
        print("\ninterrupted - saving what has been learned so far")
    finally:
        callback.close()
        env.close()

    out = paths.model_file(label, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    minutes = (time.time() - started) / 60.0
    print("Finished in %.1f minutes." % minutes)
    print("  model -> %s" % out.relative_to(paths.ROOT).as_posix())
    print("  curve -> %s" % curve.relative_to(paths.ROOT).as_posix())
    if env.episode_queues:
        first = env.episode_queues[:10]
        last = env.episode_queues[-10:]
        print("  mean queue over first 10 episodes: %5.1f vehicles"
              % (sum(first) / len(first)))
        print("  mean queue over last  10 episodes: %5.1f vehicles"
              % (sum(last) / len(last)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train independent DQN or PPO on the corridor.")
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--junctions", type=int, default=3)
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true",
                        help="a very short run, to prove the wiring works")
    args = parser.parse_args()

    episodes = 2 if args.smoke else args.episodes
    tag = args.tag or ("smoke" if args.smoke else "")
    train(args.algo, episodes, args.seed, args.junctions, tag)


if __name__ == "__main__":
    main()
