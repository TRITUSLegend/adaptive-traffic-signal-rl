"""
Shared training machinery for the reinforcement-learning controllers.

`train_dqn.py` and `train_ppo.py` are thin command-line wrappers around this
module, so both algorithms are guaranteed to train on the same environment,
the same demand, the same episode length and the same reward.  If they did not
share this code it would be very easy to accidentally give one of them an
advantage and then report the difference as an algorithmic result.

WHAT "TRAINING" MEANS HERE
-------------------------
The agent plays the same hour of traffic over and over.  Each time it reaches
a decision point (every 5 simulated seconds) it looks at the state of the
intersection and picks one of four signal phases.  At the end of the step it
receives a reward - minus the average queue per lane - and stores the
experience.  Periodically it adjusts its neural network so that actions which
led to shorter queues become more likely.  Nothing about traffic engineering is
programmed in; the agent only ever sees numbers and a score.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sumo_setup  # noqa: E402,F401  (configures SUMO paths before SUMO imports)
import gymnasium as gym  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from eval.simulation import (  # noqa: E402
    DELTA_TIME,
    EPISODE_SECONDS,
    MIN_GREEN,
    REWARD_FN,
    YELLOW_TIME,
    _quiet_stdout,
    make_env,
)
from networks.generate_routes import TRAINING_SCENARIO, training_route_files  # noqa: E402

MODELS_DIR = ROOT / "agent" / "models"
LOGS_DIR = ROOT / "results" / "logs"

# Decisions per episode: one simulated hour divided by the 5-second decision
# interval. Used to translate the friendlier `--episodes` flag into the
# timestep count stable-baselines3 actually wants.
STEPS_PER_EPISODE = EPISODE_SECONDS // DELTA_TIME  # 720


class RotatingRouteEnv(gym.Wrapper):
    """
    Give the agent a different hour of traffic on every episode.

    Without this the agent would replay one fixed list of vehicles for the
    whole of training, and the cheapest way to score well would be to memorise
    that exact sequence - "at 14 minutes 35 seconds, switch to east-west" -
    which is not a traffic controller, it is a recording. Cycling through ten
    independently generated hours of the same demand pattern forces it to learn
    a rule that responds to what it can actually see.

    Implementation note: sumo-rl reads its route file afresh from `_route`
    every time the simulation restarts, so swapping that attribute before
    `reset()` is all that is needed.
    """

    def __init__(self, env, route_files):
        super().__init__(env)
        self._routes = [str(p) for p in route_files]
        if not self._routes:
            raise ValueError("no training route files - run generate_routes.py")
        self._index = -1

    def reset(self, **kwargs):
        self._index = (self._index + 1) % len(self._routes)
        self.env.unwrapped._route = self._routes[self._index]
        with _quiet_stdout():
            return self.env.reset(**kwargs)


class EpisodeLogger(BaseCallback):
    """
    Record what happened in every training episode so we can plot the curve.

    We log the episode return, which for our reward is exactly minus the total
    vehicle-seconds of queueing in that hour. A curve that falls and then
    flattens is the evidence that the agent learned something and then stopped
    improving; a flat noisy line would mean it never did.
    """

    def __init__(self, csv_path: Path, algo: str, report_every: int = 10):
        super().__init__()
        self.csv_path = csv_path
        self.algo = algo
        self.report_every = report_every
        self.rows = []
        self._start = time.time()

    def _on_step(self) -> bool:
        # Monitor attaches an "episode" entry to info on the final step of an
        # episode; that is the only place SB3 exposes the completed return.
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            n = len(self.rows) + 1
            row = {
                "episode": n,
                "timesteps": int(self.num_timesteps),
                "return": float(episode["r"]),
                "length": int(episode["l"]),
                "wall_seconds": round(time.time() - self._start, 1),
                # Mean queue implied by the return, in vehicles: easier to read
                # than the raw return and directly comparable to the baseline.
                "mean_queue_veh": round(
                    -float(episode["r"]) * 8.0 / max(int(episode["l"]), 1), 2),
            }
            self.rows.append(row)
            if n % self.report_every == 0:
                window = self.rows[-self.report_every:]
                avg = sum(r["return"] for r in window) / len(window)
                queue = sum(r["mean_queue_veh"] for r in window) / len(window)
                print("  episode %4d | %7d steps | return(avg%d) %9.1f "
                      "| mean queue %5.1f veh | %5.1f min elapsed"
                      % (n, self.num_timesteps, self.report_every, avg, queue,
                         row["wall_seconds"] / 60.0))
                self._flush()
        return True

    def _flush(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)

    def _on_training_end(self) -> None:
        if self.rows:
            self._flush()


def build_training_env(seed: int):
    """The environment the agent learns in: ns_heavy demand, rotating seeds."""
    routes = training_route_files()
    missing = [p.name for p in routes if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing training route files (%s ...) - run:  "
            "python networks/generate_routes.py" % missing[0])

    with _quiet_stdout():
        env = make_env(
            scenario=TRAINING_SCENARIO,
            seed=seed,
            fixed_ts=False,      # the agent, not SUMO, drives the signal
            tripinfo=None,       # per-vehicle reports are only needed at eval time
            route_path=routes[0],
        )
    env = RotatingRouteEnv(env, routes)
    # Monitor is what records episode returns and lengths for the curve.
    return Monitor(env)


def describe_setup(algo: str, episodes: int, timesteps: int, seed: int) -> None:
    print("=" * 72)
    print("Training %s on the single-intersection traffic signal" % algo.upper())
    print("=" * 72)
    print("  scenario          %s (rotating over %d demand seeds)"
          % (TRAINING_SCENARIO, len(training_route_files())))
    print("  episode length    %d simulated seconds (%d decisions)"
          % (EPISODE_SECONDS, STEPS_PER_EPISODE))
    print("  decision interval %d s   yellow %d s   min green %d s"
          % (DELTA_TIME, YELLOW_TIME, MIN_GREEN))
    print("  reward            %s" % getattr(REWARD_FN, "__name__", REWARD_FN))
    print("  budget            %d episodes = %d timesteps" % (episodes, timesteps))
    print("  random seed       %d" % seed)
    print("=" * 72)


def train(algo: str, model_factory, episodes: int, seed: int, tag: str = "") -> Path:
    """
    Run a full training job and save the model plus its learning curve.

    `model_factory(env, seed)` builds the stable-baselines3 algorithm; keeping
    it a parameter is what lets DQN and PPO share this function.
    """
    timesteps = episodes * STEPS_PER_EPISODE
    describe_setup(algo, episodes, timesteps, seed)

    name = algo + (("_" + tag) if tag else "")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / name
    curve_path = LOGS_DIR / ("training_curve_%s.csv" % name)

    env = build_training_env(seed)
    model = model_factory(env, seed)

    logger = EpisodeLogger(curve_path, algo)
    started = time.time()
    try:
        model.learn(total_timesteps=timesteps, callback=logger, progress_bar=False)
    except KeyboardInterrupt:
        # Interrupting a long run should still leave a usable model behind
        # rather than throwing away the training done so far.
        print("\nInterrupted - saving the partially trained model.")
    finally:
        model.save(str(model_path))
        env.close()

    minutes = (time.time() - started) / 60.0
    print("\nFinished in %.1f minutes." % minutes)
    print("  model  %s.zip" % model_path)
    print("  curve  %s" % curve_path)
    if logger.rows:
        first = logger.rows[:10]
        last = logger.rows[-10:]
        start_q = sum(r["mean_queue_veh"] for r in first) / len(first)
        end_q = sum(r["mean_queue_veh"] for r in last) / len(last)
        print("  mean queue over first 10 episodes: %5.1f vehicles" % start_q)
        print("  mean queue over last  10 episodes: %5.1f vehicles" % end_q)
        if end_q < start_q:
            print("  -> improved by %.0f%% during training"
                  % (100.0 * (start_q - end_q) / start_q))
    return model_path
