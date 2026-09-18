"""
Train MA2C on the corridor.

Run:
    python corridor/train_ma2c.py --episodes 170
    python corridor/train_ma2c.py --episodes 2 --smoke

Output:
    corridor/models/ma2c.pt
    results/corridor/logs/training_curve_ma2c.csv

WHY THIS HAS ITS OWN LOOP
-------------------------
The independent learners run through stable-baselines3, which owns its
training loop and sees one flat vector per agent.  MA2C cannot: an agent's
input depends on its NEIGHBOURS' observations and on their most recent action
distributions, and its reward is shaped by its neighbours' rewards.  Those are
corridor-level quantities that have to be assembled between the environment
and the learner, which is precisely what SB3's interface does not expose.

So this is a plain n-step advantage actor-critic loop: roll out ROLLOUT steps,
bootstrap the return from the value of the state we stopped in, update every
agent, repeat.  The algorithm lives in corridor/ma2c.py and is tested
independently of this script - the mechanisms that make it multi-agent are
checked on hand-computable cases in tests/test_ma2c.py, rather than being
inferred from whether the curve goes down.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sumo_setup  # noqa: E402,F401

from corridor import paths  # noqa: E402
from corridor.env import make_corridor_env  # noqa: E402
from corridor.generate_corridor_routes import (  # noqa: E402
    TRAINING_SCENARIO,
    TRAINING_SEEDS,
)
from corridor.ma2c import MA2C, ROLLOUT  # noqa: E402
from eval.simulation import quiet_stdout  # noqa: E402

DEFAULT_EPISODES = 170


def empty_batch(ts_ids: list) -> dict:
    return {ts: {"inputs": [], "actions": [], "rewards": [], "bootstrap": 0.0}
            for ts in ts_ids}


def train(episodes: int, seed: int, n_junctions: int, tag: str = "",
          scenario: str = TRAINING_SCENARIO, quiet: bool = True):
    curve_path = paths.training_curve("ma2c", tag)
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    model = None
    returns, queues = [], []
    started = time.time()

    print("Training ma2c for %d episodes on a %d-junction corridor"
          % (episodes, n_junctions))
    try:
        for episode in range(episodes):
            # Rotate the demand so the policy cannot memorise one hour.
            route_seed = TRAINING_SEEDS[episode % len(TRAINING_SEEDS)]
            with quiet_stdout():
                env = make_corridor_env(scenario, route_seed, fixed_ts=False,
                                        n_junctions=n_junctions)
                observations = env.reset()

            if model is None:
                model = MA2C(list(env.ts_ids), seed=seed)
            model.reset_policies()

            batch = empty_batch(model.ts_ids)
            episode_return = 0.0
            queue_sum = 0.0
            steps = 0
            done = False

            while not done:
                actions = model.act(observations)
                # The inputs `act` actually used, not a rebuild of them. A
                # rebuild would be computed after the neighbour policies had
                # moved on, so the gradient would be evaluated at a state that
                # did not produce the action.
                inputs = model.last_inputs

                with quiet_stdout():
                    observations, rewards, dones, info = env.step(actions)
                done = dones["__all__"]

                shaped = model.shaped_rewards(rewards)
                for ts, value in inputs.items():
                    batch[ts]["inputs"].append(value)
                    batch[ts]["actions"].append(actions[ts])
                    batch[ts]["rewards"].append(shaped[ts])

                episode_return += float(sum(rewards.values()))
                queue_sum += float(info.get("system_total_stopped", 0.0))
                steps += 1

                if steps % ROLLOUT == 0 or done:
                    # Bootstrap from the value of the state we stopped in; zero
                    # at the end of an episode, where there is no future.
                    if done:
                        for ts in batch:
                            batch[ts]["bootstrap"] = 0.0
                    else:
                        with torch.no_grad():
                            for ts in batch:
                                if ts not in observations:
                                    continue
                                x = torch.from_numpy(
                                    model.build_input(ts, observations)).float()
                                _, value = model.nets[ts](x.unsqueeze(0))
                                batch[ts]["bootstrap"] = float(value.item())
                    model.update(batch)
                    batch = empty_batch(model.ts_ids)

            with quiet_stdout():
                env.close()

            returns.append(episode_return)
            queues.append(queue_sum / steps if steps else 0.0)

            if (episode + 1) % 10 == 0 or episode == 0:
                minutes = (time.time() - started) / 60.0
                print("  ma2c episode %4d | return %10.1f | mean queue %6.1f veh "
                      "| %5.1f min elapsed"
                      % (episode + 1, returns[-1], queues[-1], minutes),
                      flush=True)
            if (episode + 1) % 10 == 0:
                write_curve(curve_path, returns, queues)
    except KeyboardInterrupt:
        print("\ninterrupted - saving what has been learned so far")

    write_curve(curve_path, returns, queues)
    out = paths.model_file("ma2c", tag)
    if model is not None:
        model.save(out)

    minutes = (time.time() - started) / 60.0
    print("Finished in %.1f minutes." % minutes)
    print("  model -> %s" % out.relative_to(paths.ROOT).as_posix())
    print("  curve -> %s" % curve_path.relative_to(paths.ROOT).as_posix())
    if queues:
        first = queues[:10]
        last = queues[-10:]
        print("  mean queue over first 10 episodes: %5.1f vehicles"
              % (sum(first) / len(first)))
        print("  mean queue over last  10 episodes: %5.1f vehicles"
              % (sum(last) / len(last)))
    return out


def write_curve(path: Path, returns: list, queues: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["episode", "return", "mean_queue_veh"])
        for i, (ret, queue) in enumerate(zip(returns, queues), 1):
            writer.writerow([i, round(ret, 3), round(queue, 4)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MA2C on the corridor.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--junctions", type=int, default=3)
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    episodes = 2 if args.smoke else args.episodes
    tag = args.tag or ("smoke" if args.smoke else "")
    train(episodes, args.seed, args.junctions, tag)


if __name__ == "__main__":
    main()
