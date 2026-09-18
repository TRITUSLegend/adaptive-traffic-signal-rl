"""
Present the corridor to stable-baselines3 as a vector of per-junction agents.

WHY A VecEnv
------------
Each junction observes only its own approaches (`Box(21,)`) and chooses only
its own phase (`Discrete(4)`) - identical in shape to the single-intersection
environment.  N junctions stepping one shared SUMO simulation is therefore
exactly the shape SB3 already understands as a vector of N environments, and
using that interface means DQN and PPO run through their own tested `learn()`
loop rather than through a hand-driven one.

THE LOCKSTEP PROPERTY THIS RELIES ON, AND WHY IT IS ASSERTED
------------------------------------------------------------
sumo-rl returns observations only for signals whose `time_to_act` is true, so
in general agents act asynchronously.  On this corridor they do not: every
signal starts at `begin_time` and every branch of `TrafficSignal.set_next_phase`
sets `next_action_time = sim_step + delta_time`, so signals that start together
stay together.  Measured on a `calibrated` hour: all three junctions appeared in
all 200 sampled steps.

That is a convenient fact, not a guarantee, and a silent violation would be
poisonous - SB3 would pair junction A's observation with junction B's reward
and train on transitions that never happened.  So it is checked on every step
(`_require_all_agents`) and raises rather than papering over a partial dict.

PARAMETER SHARING - A FLAGGED DEVIATION FROM THE PLAN
-----------------------------------------------------
CORRIDOR_PLAN.md section 3 proposed "independent learners, one policy per
intersection".  What is implemented is **independent learners with shared
parameters**: one network trained on the experience of all N junctions, with
each junction still acting independently on its own local observation at
execution time.

Two reasons, both recorded in the plan's appendix and in CORRIDOR_REPORT.md:

  1. **Topology transfer would otherwise be undefined.**  The brief requires
     zero-shot evaluation on a structurally different corridor.  With one
     network per junction there is no principled answer to which of the three
     policies drives junction 4 of a five-junction corridor, or which drives an
     irregular `cologne3` junction.  A shared policy transfers by construction.
  2. It is a standard MARL baseline in its own right, and it uses SB3's tested
     training loop instead of a hand-rolled substitute.

The cost is real and is stated rather than hidden: the three junctions occupy
genuinely different roles (two edge junctions, one interior), and a shared
network cannot specialise to them.  That is a limitation of this result, not a
neutral implementation detail.

ROUTE ROTATION
--------------
Training rotates through the training seeds' route files, so the policy cannot
memorise one hour of arrivals - the same guard `RotatingRouteEnv` provides in
the single-intersection study.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corridor import paths  # noqa: E402
from corridor.env import make_corridor_env  # noqa: E402
from corridor.generate_corridor_routes import (  # noqa: E402
    TRAINING_SCENARIO,
    TRAINING_SEEDS,
)
from eval.simulation import quiet_stdout  # noqa: E402


class CorridorVecEnv(VecEnv):
    """
    One SUMO corridor presented as `n_junctions` parallel agents.

    All sub-environments finish together (the episode ends when the simulated
    hour does), so the usual VecEnv auto-reset applies to every agent at once.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_junctions: int = 3, scenario: str = TRAINING_SCENARIO,
                 seeds: list = None, quiet: bool = True):
        self.n_junctions = n_junctions
        self.scenario = scenario
        self.seeds = list(seeds) if seeds else list(TRAINING_SEEDS)
        self._quiet = quiet
        self._episode = 0
        self._env = None
        self._ts_ids = None
        self._last_obs = None

        # Episode bookkeeping, read by the training scripts for the curve.
        self.episode_returns = []
        self.episode_queues = []
        self._return = 0.0
        self._queue_sum = 0.0
        self._queue_n = 0

        probe = self._open(self.seeds[0])
        ts_id = probe.ts_ids[0]
        observation_space = probe.observation_spaces(ts_id)
        action_space = probe.action_spaces(ts_id)
        self._close_env()

        super().__init__(n_junctions, observation_space, action_space)
        self.actions = None

    # -- environment lifecycle ------------------------------------------
    def _open(self, seed: int):
        with quiet_stdout():
            self._env = make_corridor_env(
                self.scenario, seed, fixed_ts=False,
                n_junctions=self.n_junctions)
        self._ts_ids = list(self._env.ts_ids)
        return self._env

    def _close_env(self) -> None:
        if self._env is not None:
            try:
                with quiet_stdout():
                    self._env.close()
            except Exception:       # noqa: BLE001 - teardown must not mask a real error
                pass
            self._env = None

    def _current_seed(self) -> int:
        """Rotate through the training route files, one per episode."""
        return self.seeds[self._episode % len(self.seeds)]

    def _require_all_agents(self, observations: dict) -> None:
        missing = set(self._ts_ids) - set(observations)
        if missing:
            raise RuntimeError(
                "sumo-rl returned observations for %d of %d junctions "
                "(missing %s). The corridor's agents are no longer acting in "
                "lockstep, so vectorised training would pair one junction's "
                "observation with another's reward. See the module docstring."
                % (len(observations), len(self._ts_ids), sorted(missing)))

    def _stack(self, observations: dict) -> np.ndarray:
        self._require_all_agents(observations)
        return np.asarray([observations[ts] for ts in self._ts_ids],
                          dtype=np.float32)

    # -- VecEnv API ------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._close_env()
        env = self._open(self._current_seed())
        with quiet_stdout():
            observations = env.reset()
        self._return = 0.0
        self._queue_sum = 0.0
        self._queue_n = 0
        self._last_obs = self._stack(observations)
        return self._last_obs

    def step_async(self, actions) -> None:
        self.actions = actions

    def step_wait(self):
        action_dict = {ts: int(a) for ts, a in zip(self._ts_ids, self.actions)}
        with quiet_stdout():
            observations, rewards, dones, info = self._env.step(action_dict)

        self._require_all_agents(observations)
        obs = self._stack(observations)
        reward_vec = np.asarray([float(rewards[ts]) for ts in self._ts_ids],
                                dtype=np.float32)
        done = bool(dones["__all__"])
        done_vec = np.asarray([done] * self.num_envs, dtype=bool)

        self._return += float(reward_vec.sum())
        self._queue_sum += float(info.get("system_total_stopped", 0.0))
        self._queue_n += 1

        infos = [{} for _ in range(self.num_envs)]
        if done:
            # SB3 expects the terminal observation in `info` and a fresh one
            # from `reset()` in its place, since the VecEnv auto-resets.
            self.episode_returns.append(self._return)
            self.episode_queues.append(
                self._queue_sum / self._queue_n if self._queue_n else 0.0)
            mean_queue = self.episode_queues[-1]
            for i in range(self.num_envs):
                infos[i]["terminal_observation"] = obs[i]
                infos[i]["episode"] = {"r": self._return, "l": self._queue_n,
                                       "queue": mean_queue}
            self._episode += 1
            obs = self.reset()

        self._last_obs = obs
        return obs, reward_vec, done_vec, infos

    def close(self) -> None:
        self._close_env()

    # -- the rest of the VecEnv contract --------------------------------
    #
    # SB3 requires these, but the corridor is one simulation rather than N
    # separate objects, so there is no per-sub-env attribute to get or set.
    def get_attr(self, attr_name: str, indices=None):
        indices = self._indices(indices)
        return [getattr(self, attr_name, None) for _ in indices]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        raise NotImplementedError(
            "CorridorVecEnv wraps a single simulation; per-agent attributes "
            "cannot be set independently.")

    def env_method(self, method_name: str, *args, indices=None, **kwargs):
        raise NotImplementedError(
            "CorridorVecEnv wraps a single simulation; there is no per-agent "
            "environment to call %r on." % method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._indices(indices))

    def _indices(self, indices):
        if indices is None:
            return list(range(self.num_envs))
        if isinstance(indices, int):
            return [indices]
        return list(indices)


def make_training_vec_env(n_junctions: int = 3,
                          scenario: str = TRAINING_SCENARIO,
                          seeds: list = None) -> CorridorVecEnv:
    """The training environment: the corridor over its rotating training hours."""
    missing = [paths.route_file(n_junctions, scenario, s)
               for s in (seeds or TRAINING_SEEDS)]
    absent = [p for p in missing if not p.exists()]
    if absent:
        raise FileNotFoundError(
            "%d training route files are missing (first: %s) - run:  "
            "python corridor/generate_corridor_routes.py --junctions %d"
            % (len(absent), absent[0], n_junctions))
    return CorridorVecEnv(n_junctions=n_junctions, scenario=scenario, seeds=seeds)
