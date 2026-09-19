"""
MA2C - multi-agent advantage actor-critic for the corridor, written from scratch.

WHY FROM SCRATCH
----------------
MA2C is not in stable-baselines3 and there is no maintained SB3-compatible
implementation.  The alternatives were a heavy MARL framework (MARLlib/RLlib
needs Ray, which is poor on a CPU-only Windows box) or research code coupled to
a different benchmark.  This project already hand-rolled its Student-t and
exact Wilcoxon distributions rather than adding SciPy, and covered them with
tests against published values; the same trade applies here.

WHAT MA2C IS (Chu et al., IEEE T-ITS 2020), AND WHAT IS IMPLEMENTED
-------------------------------------------------------------------
Independent actor-critics, one per intersection, made cooperative by three
mechanisms - all three are implemented:

  1. **Neighbour observation sharing.**  An agent's input is its own
     observation concatenated with its neighbours', scaled down by a spatial
     discount factor ALPHA.  The scaling says "your neighbour's queue matters
     to you, but less than your own".
  2. **Neighbour policy sharing.**  The neighbours' most recent action
     probabilities are appended to the input.  This is the part that addresses
     non-stationarity: an agent that can see what its neighbours are about to
     do is not chasing a moving target blindly.
  3. **Spatially discounted reward.**  Agent i is rewarded for its own queue
     plus ALPHA times its neighbours'.  Without this, three selfish agents
     optimise three junctions and nobody optimises the corridor.

DELIBERATE DEVIATION: NO LSTM
-----------------------------
The published algorithm puts an LSTM in each agent.  v1 here does not, and
CORRIDOR_PLAN.md section 10 records the decision.  The recurrent state
addresses partial observability *within* an agent and is orthogonal to the
coordination question this study asks; it would also interact with the 5 s
decision interval and the episode-boundary reset in ways needing their own
tests to trust.

This matters when reading the result: if MA2C underperforms the simpler
independent learners, the missing LSTM is a live candidate explanation and
CORRIDOR_REPORT.md says so rather than concluding that MA2C does not work.

WHY THE NEIGHBOUR INPUT IS PADDED
---------------------------------
J2 has two neighbours; J1 and J3 have one.  Inputs are padded to a fixed
MAX_NEIGHBOURS so that every agent's network has the same shape - which is
also what lets a trained MA2C policy be evaluated zero-shot on a corridor of a
different length, where junctions have different neighbour counts.  Padding is
zeros, which is the same value a neighbour with no queue and a uniform policy
would contribute, so it is a neutral filler rather than a signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Spatial discount on neighbour observations and rewards. Chu et al. report
# 0.75 as a good default; it is the weight of "my neighbour's problem" relative
# to "my problem".
ALPHA = 0.75

# Every junction in a linear corridor has at most two signalised neighbours.
MAX_NEIGHBOURS = 2

OBS_DIM = 21        # sumo-rl default per junction: phase one-hot + densities + queues
N_ACTIONS = 4       # the four green phases

GAMMA = 0.95        # ~100 s horizon ~ one signal cycle; matches the rest of the project

# Converts the discounted SUM of rewards into a discounted AVERAGE.
#
# Without it the critic cannot arrive. A per-step shaped reward around -5 over
# a 1/(1-GAMMA) = 20 step horizon implies returns near -100, and Adam moves a
# parameter about `lr` per step, so travelling from an initialisation near zero
# to -100 needs on the order of 150,000 updates. A 170-episode run has 6,120.
# Measured after four episodes the critic sat at -4.2 against a true return of
# -44, and a critic that never arrives is not merely slow: with V near zero the
# advantage is just the return, whose positional structure inside a rollout the
# policy then learns instead of anything about actions.
#
# Multiplying by (1 - GAMMA) rescales the return to the same units as one
# step's reward, which is a bounded, interpretable quantity the critic can
# reach in a few thousand updates. It is the same reasoning the single-
# intersection reward uses in dividing by its lane count - "purely numerical
# hygiene ... dividing by a constant cannot change which policy is optimal, it
# just rescales every value" - and it leaves the optimal policy untouched for
# the same reason. The actor is unaffected either way, because it consumes a
# NORMALISED advantage, in which any constant factor cancels.
REWARD_SCALE = 1.0 - GAMMA
HIDDEN = 64
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
ROLLOUT = 20        # n-step return length


def agent_input_dim() -> int:
    """Own observation + discounted neighbour observations + neighbour policies."""
    return OBS_DIM + MAX_NEIGHBOURS * OBS_DIM + MAX_NEIGHBOURS * N_ACTIONS


def neighbours(ts_ids: list) -> dict:
    """
    Adjacency along a linear arterial: each junction neighbours the next.

    `ts_ids` must be in corridor order, which is how `build_corridor.py` names
    them (J1, J2, ... west to east). For a network we did not build - cologne3
    in the transfer experiment - the ids are sorted and treated as a chain,
    which is an approximation and is flagged where it is used.
    """
    result = {}
    for i, ts in enumerate(ts_ids):
        near = []
        if i > 0:
            near.append(ts_ids[i - 1])
        if i + 1 < len(ts_ids):
            near.append(ts_ids[i + 1])
        result[ts] = near
    return result


class ActorCritic(nn.Module):
    """
    One agent's policy and value function, as two SEPARATE networks.

    WHY THEY DO NOT SHARE A BODY
    ----------------------------
    A shared trunk is the more common design and it is what this class had
    first. It failed here, twice, for the same underlying reason: actor and
    critic then share one `clip_grad_norm_`, and their losses are on wildly
    different scales.

    With a squared value loss the critic dominated - value loss 2895 against a
    policy loss of 105, so the clip scaled everything by 0.005 and the policy
    head received half a percent of its natural step. Switching the critic to
    smooth-L1 fixed that and broke the other side: the critic's gradient is
    then bounded at 1 per element, so with a typical error of 40 it needed on
    the order of 130,000 updates and had 6,000. Measured after four episodes it
    predicted -1.9 against a true return of -41.9, essentially still at its
    initialisation.

    A dead critic is not a slow critic. With V near zero the advantage is just
    the return, and n-step returns carry a positional artifact - a step early in
    a rollout accumulates more discounted reward than a step late in it. Once
    normalised, that becomes "actions late in a rollout are good", which is
    uncorrelated with whether the action was any good. The policy learned that
    artifact confidently, which is why the corridor starved its busiest
    movement rather than merely failing to improve.

    Separating the networks gives each its own gradient budget, so the critic
    can use a squared loss and converge quickly while the actor's step is
    unaffected by the critic's scale. It costs a few thousand extra parameters
    and removes the coupling entirely.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, N_ACTIONS),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


class MA2C:
    """
    The full multi-agent learner: one ActorCritic per junction.

    Parameters are NOT shared between agents - that is what makes this MA2C
    rather than the parameter-sharing independent learners in vec_env.py, and
    it is why MA2C can specialise to the interior junction's different role.
    """

    def __init__(self, ts_ids: list, alpha: float = ALPHA, seed: int = 0):
        self.ts_ids = list(ts_ids)
        self.alpha = alpha
        self.neighbours = neighbours(self.ts_ids)
        torch.manual_seed(seed)
        self.nets = {ts: ActorCritic(agent_input_dim()) for ts in self.ts_ids}
        self.optimisers = {
            ts: torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
            for ts, net in self.nets.items()}
        # Last action distribution per agent, for neighbour policy sharing.
        self._last_policy = {ts: np.full(N_ACTIONS, 1.0 / N_ACTIONS,
                                         dtype=np.float32)
                             for ts in self.ts_ids}
        # The inputs the most recent `act` actually used. The trainer stores
        # these rather than rebuilding, so the gradient is evaluated at the
        # state that produced the action.
        self.last_inputs = {}

    # -- input assembly --------------------------------------------------
    def build_input(self, ts: str, observations: dict) -> np.ndarray:
        """Own observation, then each neighbour's observation and policy."""
        own = np.asarray(observations[ts], dtype=np.float32)
        parts = [own]

        near = self.neighbours.get(ts, [])
        for i in range(MAX_NEIGHBOURS):
            if i < len(near) and near[i] in observations:
                parts.append(self.alpha *
                             np.asarray(observations[near[i]], dtype=np.float32))
            else:
                parts.append(np.zeros(OBS_DIM, dtype=np.float32))
        for i in range(MAX_NEIGHBOURS):
            if i < len(near):
                parts.append(self._last_policy[near[i]])
            else:
                parts.append(np.zeros(N_ACTIONS, dtype=np.float32))
        return np.concatenate(parts)

    def shaped_rewards(self, rewards: dict) -> dict:
        """
        Spatially discounted reward: a WEIGHTED MEAN of own and neighbours'.

        This is the mechanism that makes the agents cooperative rather than
        three selfish controllers sharing a road: an agent is rewarded for its
        own queue plus ALPHA times each neighbour's, so "my neighbour's problem
        matters to me, but less than mine".

        WHY IT IS DIVIDED BY THE TOTAL WEIGHT
        -------------------------------------
        The first version returned the raw weighted SUM, and MA2C diverged:
        mean queue climbed from 118 to 362 over ten episodes and saturated at
        gridlock, while the independent learners went from 115 to 39 over the
        same span. Two things were wrong, and the second is the worse one.

        **The scale was inflated.** An interior agent's reward was 2.5x the raw
        per-junction value, so its 20-step returns ran to about -160 against
        roughly -64 in the single-agent case - and the learning rate had been
        carried over from that single-agent setting unchanged.

        **The scale differed BY POSITION.** An interior junction has two
        neighbours and an edge junction one, so J2's value function operated on
        a 1.43x different scale from J1's for no reason other than where it sat
        in the corridor. The same algorithm was effectively tuned differently
        at different junctions, which is not a property anyone would choose.

        Dividing by the total weight fixes both while changing nothing about
        what the mechanism does: the RELATIVE weighting of own versus neighbour
        is identical, and a lone agent with no neighbours is untouched. It is a
        weighted mean rather than a weighted sum. Flagged in CORRIDOR_REPORT.md
        as a deviation from a literal reading of Chu et al., because the paper
        does not normalise explicitly.
        """
        shaped = {}
        for ts in self.ts_ids:
            total = float(rewards.get(ts, 0.0))
            weight = 1.0
            for near in self.neighbours.get(ts, []):
                total += self.alpha * float(rewards.get(near, 0.0))
                weight += self.alpha
            # REWARD_SCALE turns the discounted sum the critic must predict
            # into a discounted average - see its definition for why the
            # critic could not otherwise arrive.
            shaped[ts] = REWARD_SCALE * total / weight
        return shaped

    # -- acting ----------------------------------------------------------
    def act(self, observations: dict, deterministic: bool = False) -> dict:
        """
        Choose a phase for every junction present in `observations`.

        THE ORDER OF THESE THREE STEPS IS THE ALGORITHM
        -----------------------------------------------
        Every agent's input is built FIRST, from the policies of the previous
        timestep; then every agent decides; only then are the policies
        replaced. The first version updated `_last_policy[ts]` inside the
        decision loop, which broke two things at once:

          * **Order dependence.** The second agent processed saw the first
            agent's *current* policy while the third's was still one step old,
            so the corridor's behaviour depended on dictionary iteration order.
            Neighbour policy sharing in Chu et al. uses the previous step's
            policies for all neighbours - it has to, because in a real
            deployment the agents decide simultaneously and cannot see a
            decision that has not been made yet.
          * **A gradient computed against the wrong state.** The trainer built
            its stored input from all-old policies, but the action came from a
            partially-updated one. Storing s while the action came from s'
            means the policy gradient is evaluated at a state that did not
            produce the action - the update is not wrong in an obvious,
            crashing way, it is just quietly not the gradient of the thing
            being maximised.

        `self.last_inputs` therefore holds exactly the vectors these actions
        were computed from, and the trainer stores those rather than rebuilding
        them, so the two can no longer disagree.
        """
        inputs = {ts: self.build_input(ts, observations)
                  for ts in observations if ts in self.nets}

        actions, values, log_probs, policies = {}, {}, {}, {}
        with torch.no_grad():
            for ts, x in inputs.items():
                logits, value = self.nets[ts](
                    torch.from_numpy(x).float().unsqueeze(0))
                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    action = int(torch.argmax(probs, dim=-1).item())
                else:
                    action = int(torch.multinomial(probs, 1).item())
                actions[ts] = action
                values[ts] = float(value.item())
                log_probs[ts] = float(torch.log(probs[0, action] + 1e-8).item())
                policies[ts] = probs.squeeze(0).numpy().astype(np.float32)

        # Replaced only once every agent has decided.
        self._last_policy.update(policies)
        self.last_inputs = inputs
        self._pending = {"values": values, "log_probs": log_probs}
        return actions

    # -- learning --------------------------------------------------------
    def update(self, batch: dict) -> dict:
        """
        One A2C update per agent from a rollout of n-step returns.

        `batch[ts]` holds lists of inputs, actions, shaped rewards and the
        bootstrap value of the state the rollout ended in.
        """
        losses = {}
        for ts, data in batch.items():
            if not data["inputs"]:
                continue
            inputs = torch.from_numpy(np.asarray(data["inputs"],
                                                 dtype=np.float32))
            actions = torch.from_numpy(np.asarray(data["actions"],
                                                  dtype=np.int64))

            # Discounted n-step returns, bootstrapped from the final value.
            returns = []
            running = data["bootstrap"]
            for reward in reversed(data["rewards"]):
                running = reward + GAMMA * running
                returns.append(running)
            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float32)

            logits, values = self.nets[ts](inputs)
            advantage = returns - values
            log_probs = F.log_softmax(logits, dim=-1)
            chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # NORMALISED advantage for the policy gradient, and a SMOOTH-L1
            # value loss. Both are standard; together they are what stops the
            # critic starving the actor under a shared gradient clip.
            #
            # Measured on the first version, which used the raw advantage and
            # a squared value loss: with returns around -76, the value loss
            # came to 2895 against a policy loss of 105. The two heads share
            # one clip_grad_norm_(0.5), so a total gradient norm of 97 was
            # scaled by 0.005 and the policy head received 0.036 - half a
            # percent of its natural step. The actor was effectively frozen
            # while the critic crawled, MA2C never learned, and mean queue
            # drifted from 118 to 362 vehicles instead of descending.
            #
            # Normalising the advantage makes the policy gradient independent
            # of reward scale, which is what PPO does by default. Smooth-L1
            # bounds the critic's gradient per element instead of letting it
            # grow with the square of the error, which is why SB3's DQN uses it
            # for the same reason. Neither is fitted to this problem's numbers.
            normalised = advantage.detach()
            if normalised.numel() > 1:
                normalised = ((normalised - normalised.mean())
                              / (normalised.std() + 1e-8))

            policy_loss = -(chosen * normalised).mean()
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
            actor_loss = policy_loss - ENTROPY_COEF * entropy
            # A squared loss, because the critic now has its own gradient
            # budget and can afford a step proportional to its error. That is
            # what lets it travel from 0 to around -40 within a training run
            # instead of needing twenty times more updates than exist.
            critic_loss = VALUE_COEF * F.mse_loss(values, returns)

            # Each network is clipped against its own norm, so the critic's
            # scale cannot consume the actor's step or vice versa.
            self.optimisers[ts].zero_grad()
            (actor_loss + critic_loss).backward()
            nn.utils.clip_grad_norm_(self.nets[ts].actor.parameters(),
                                     MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(self.nets[ts].critic.parameters(),
                                     MAX_GRAD_NORM)
            self.optimisers[ts].step()
            losses[ts] = float((actor_loss + critic_loss).item())
        return losses

    def reset_policies(self) -> None:
        """Forget neighbour policies at an episode boundary."""
        for ts in self.ts_ids:
            self._last_policy[ts] = np.full(N_ACTIONS, 1.0 / N_ACTIONS,
                                            dtype=np.float32)
        self.last_inputs = {}

    # -- persistence -----------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "ts_ids": self.ts_ids,
            "alpha": self.alpha,
            "input_dim": agent_input_dim(),
            "nets": {ts: net.state_dict() for ts, net in self.nets.items()},
        }, path)

    @classmethod
    def load(cls, path: Path, ts_ids: list = None):
        """
        Rebuild a saved learner.

        `ts_ids` overrides the saved ids, which is how a corridor3 policy is
        evaluated on a network with different junction names: the agents are
        matched positionally along the corridor.  Where the target has more
        junctions than the policy was trained with, the saved agents are reused
        cyclically - an approximation, flagged wherever a transfer number is
        reported.
        """
        blob = torch.load(path, map_location="cpu", weights_only=False)
        saved_ids = list(blob["ts_ids"])
        model = cls(saved_ids, alpha=blob["alpha"])
        for ts, state in blob["nets"].items():
            model.nets[ts].load_state_dict(state)
        if ts_ids is not None and list(ts_ids) != saved_ids:
            model = model._remap(list(ts_ids), saved_ids)
        return model

    def _remap(self, new_ids: list, saved_ids: list):
        """Reassign trained agents to a different set of junction ids, in order."""
        remapped = MA2C(new_ids, alpha=self.alpha)
        for i, ts in enumerate(new_ids):
            source = saved_ids[i % len(saved_ids)]
            remapped.nets[ts].load_state_dict(self.nets[source].state_dict())
        return remapped


class MA2CController:
    """
    Drives a trained MA2C through the corridor's shared episode runner.

    It needs the whole observation dict rather than one junction's vector,
    because an agent's input includes its neighbours' observations - which is
    why MA2C cannot reuse `IndependentPolicyController`.
    """

    uses_env_control = True

    def __init__(self, model: MA2C, name: str = "ma2c", env=None):
        self.model = model
        self.name = name
        self.env = env

    def on_episode_start(self, env) -> None:
        self.env = env
        self.model.reset_policies()

    def act(self, observations: dict) -> dict:
        return self.model.act(observations, deterministic=True)


def ma2c_factory(model: MA2C, name: str = "ma2c"):
    return lambda env: MA2CController(model, name, env)
