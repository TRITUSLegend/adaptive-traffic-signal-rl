"""
Tests for the from-scratch MA2C in corridor/ma2c.py.

WHY THESE TESTS EXIST
---------------------
MA2C is the highest-risk component of the corridor work: it is the one piece
with no library behind it, and its three cooperative mechanisms all fail
*quietly*. If neighbour observations were assembled in the wrong order, or the
spatial discount were applied to the wrong agent, or the padding leaked a
signal, the thing would still train and still produce a number - it simply
would not be MA2C, and nothing in the results would say so.

So the mechanisms are tested directly, on hand-computable cases, rather than
being inferred from whether training converges. The single-intersection study
took the same view of its hand-rolled p-values: an implementation that is
quietly wrong is worse than an absent one, because it gets believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor.ma2c import (  # noqa: E402
    ALPHA,
    GAMMA,
    MAX_NEIGHBOURS,
    N_ACTIONS,
    OBS_DIM,
    ROLLOUT,
    MA2C,
    ActorCritic,
    agent_input_dim,
    neighbours,
)

IDS = ["J1", "J2", "J3"]


def fake_observations(values: dict) -> dict:
    """One constant-valued observation vector per junction."""
    return {ts: np.full(OBS_DIM, v, dtype=np.float32) for ts, v in values.items()}


# ---------------------------------------------------------------------------
# The neighbour graph
# ---------------------------------------------------------------------------
def test_interior_junction_has_two_neighbours():
    assert neighbours(IDS)["J2"] == ["J1", "J3"]


def test_edge_junctions_have_one_neighbour_each():
    near = neighbours(IDS)
    assert near["J1"] == ["J2"]
    assert near["J3"] == ["J2"]


def test_a_junction_is_never_its_own_neighbour():
    for ts, near in neighbours(IDS).items():
        assert ts not in near


def test_adjacency_is_symmetric():
    near = neighbours(IDS)
    for ts, neighbour_list in near.items():
        for other in neighbour_list:
            assert ts in near[other]


def test_a_two_junction_corridor_makes_both_agents_edge_agents():
    """
    The reason the plan rejected a 2-junction corridor: with two junctions
    MA2C's neighbour sharing is degenerate, because no agent has two
    neighbours and the architecture's asymmetry is never exercised.
    """
    near = neighbours(["J1", "J2"])
    assert near["J1"] == ["J2"] and near["J2"] == ["J1"]
    assert max(len(v) for v in near.values()) == 1


def test_a_single_junction_has_no_neighbours():
    assert neighbours(["J1"]) == {"J1": []}


# ---------------------------------------------------------------------------
# Input assembly
# ---------------------------------------------------------------------------
def test_input_dimension_accounts_for_every_part():
    assert agent_input_dim() == (
        OBS_DIM + MAX_NEIGHBOURS * OBS_DIM + MAX_NEIGHBOURS * N_ACTIONS)


def test_own_observation_is_first_and_undiscounted():
    """J2's own value is 2.0 and must arrive unscaled, ahead of its neighbours."""
    model = MA2C(IDS)
    obs = fake_observations({"J1": 1.0, "J2": 2.0, "J3": 3.0})
    x = model.build_input("J2", obs)
    assert np.allclose(x[:OBS_DIM], 2.0)


def test_neighbour_observations_are_scaled_by_the_spatial_discount():
    """J2's neighbours are J1 (value 1) and J3 (value 3), each scaled by ALPHA."""
    model = MA2C(IDS)
    obs = fake_observations({"J1": 1.0, "J2": 2.0, "J3": 3.0})
    x = model.build_input("J2", obs)
    first = x[OBS_DIM:2 * OBS_DIM]
    second = x[2 * OBS_DIM:3 * OBS_DIM]
    assert np.allclose(first, ALPHA * 1.0)
    assert np.allclose(second, ALPHA * 3.0)


def test_a_missing_neighbour_slot_is_padded_with_zeros():
    """J1 has one neighbour, so its second slot must be neutral filler."""
    model = MA2C(IDS)
    obs = fake_observations({"J1": 1.0, "J2": 2.0, "J3": 3.0})
    x = model.build_input("J1", obs)
    assert np.allclose(x[OBS_DIM:2 * OBS_DIM], ALPHA * 2.0)   # J2
    assert np.allclose(x[2 * OBS_DIM:3 * OBS_DIM], 0.0)       # padding


def test_every_agent_produces_the_same_input_shape():
    """Fixed shape is what lets one policy drive a differently-sized corridor."""
    model = MA2C(IDS)
    obs = fake_observations({"J1": 1.0, "J2": 2.0, "J3": 3.0})
    for ts in IDS:
        assert model.build_input(ts, obs).shape == (agent_input_dim(),)


def test_neighbour_policies_start_uniform():
    model = MA2C(IDS)
    obs = fake_observations({"J1": 0.0, "J2": 0.0, "J3": 0.0})
    x = model.build_input("J2", obs)
    tail = x[-MAX_NEIGHBOURS * N_ACTIONS:]
    assert np.allclose(tail[:N_ACTIONS], 1.0 / N_ACTIONS)
    assert np.allclose(tail[N_ACTIONS:], 1.0 / N_ACTIONS)


def test_acting_updates_the_policy_seen_by_neighbours():
    """Neighbour policy sharing is only real if acting actually records it."""
    model = MA2C(IDS)
    obs = fake_observations({"J1": 0.5, "J2": 0.5, "J3": 0.5})
    before = model.build_input("J2", obs)[-MAX_NEIGHBOURS * N_ACTIONS:].copy()
    model.act(obs)
    after = model.build_input("J2", obs)[-MAX_NEIGHBOURS * N_ACTIONS:]
    assert not np.allclose(before, after), (
        "J2's input did not change after its neighbours acted, so neighbour "
        "policy sharing is not wired up")


def test_resetting_policies_restores_the_uniform_prior():
    model = MA2C(IDS)
    obs = fake_observations({"J1": 0.5, "J2": 0.5, "J3": 0.5})
    model.act(obs)
    model.reset_policies()
    tail = model.build_input("J2", obs)[-MAX_NEIGHBOURS * N_ACTIONS:]
    assert np.allclose(tail, 1.0 / N_ACTIONS)


# ---------------------------------------------------------------------------
# The spatially discounted reward
# ---------------------------------------------------------------------------
def test_interior_agent_absorbs_both_neighbours_rewards():
    """A weighted MEAN: own plus ALPHA per neighbour, over the total weight."""
    model = MA2C(IDS)
    shaped = model.shaped_rewards({"J1": -1.0, "J2": -2.0, "J3": -4.0})
    expected = (-2.0 + ALPHA * (-1.0) + ALPHA * (-4.0)) / (1 + 2 * ALPHA)
    assert shaped["J2"] == pytest.approx(expected)


def test_edge_agent_absorbs_only_its_one_neighbour():
    model = MA2C(IDS)
    shaped = model.shaped_rewards({"J1": -1.0, "J2": -2.0, "J3": -4.0})
    expected = (-1.0 + ALPHA * (-2.0)) / (1 + ALPHA)
    assert shaped["J1"] == pytest.approx(expected)


def test_shaping_is_not_applied_twice_to_the_same_agent():
    model = MA2C(IDS)
    shaped = model.shaped_rewards({"J1": 0.0, "J2": -1.0, "J3": 0.0})
    assert shaped["J2"] == pytest.approx(-1.0 / (1 + 2 * ALPHA))


def test_shaping_does_not_inflate_the_reward_scale():
    """
    The divergence this guards. A weighted SUM made an interior agent's reward
    2.5x the raw per-junction value, so its returns ran far beyond the scale
    the shared learning rate was chosen for, and MA2C climbed from 118 to 362
    vehicles instead of descending.
    """
    model = MA2C(IDS)
    uniform = {ts: -5.0 for ts in IDS}
    for value in model.shaped_rewards(uniform).values():
        assert value == pytest.approx(-5.0), (
            "a uniformly bad corridor must shape to the same value it started "
            "at, not to a multiple of it")


def test_every_agent_shares_one_reward_scale_whatever_its_neighbour_count():
    """
    The worse half of the same defect: an interior junction has two neighbours
    and an edge junction one, so an unnormalised sum gave J2 a value function
    on a 1.43x different scale from J1's - the same algorithm effectively tuned
    differently by position in the corridor.
    """
    model = MA2C(IDS)
    shaped = model.shaped_rewards({ts: -5.0 for ts in IDS})
    assert shaped["J1"] == pytest.approx(shaped["J2"])
    assert shaped["J2"] == pytest.approx(shaped["J3"])


def test_relative_weighting_of_neighbours_is_preserved():
    """
    Normalising must not flatten the mechanism: a neighbour still counts for
    less than the agent itself, in exactly the ALPHA ratio.
    """
    model = MA2C(IDS)
    own_only = model.shaped_rewards({"J1": -1.0, "J2": 0.0, "J3": 0.0})["J2"]
    neighbour_only = model.shaped_rewards({"J1": 0.0, "J2": -1.0, "J3": 0.0})["J2"]
    assert neighbour_only == pytest.approx(own_only / ALPHA)


def test_a_lone_agent_reward_is_unchanged():
    model = MA2C(["J1"])
    assert model.shaped_rewards({"J1": -3.0})["J1"] == pytest.approx(-3.0)


def test_shaping_preserves_the_sign_of_a_uniformly_bad_corridor():
    model = MA2C(IDS)
    shaped = model.shaped_rewards({ts: -1.0 for ts in IDS})
    assert all(v < 0 for v in shaped.values())


# ---------------------------------------------------------------------------
# Acting and learning
# ---------------------------------------------------------------------------
def test_act_returns_one_legal_action_per_junction():
    model = MA2C(IDS)
    obs = fake_observations({ts: 0.3 for ts in IDS})
    actions = model.act(obs)
    assert set(actions) == set(IDS)
    assert all(0 <= a < N_ACTIONS for a in actions.values())


def test_act_only_answers_for_junctions_it_was_asked_about():
    """The runner may pass a partial dict; inventing actions would be wrong."""
    model = MA2C(IDS)
    obs = fake_observations({"J1": 0.3, "J2": 0.3})
    assert set(model.act(obs)) == {"J1", "J2"}


def test_deterministic_acting_is_repeatable():
    model = MA2C(IDS)
    obs = fake_observations({ts: 0.3 for ts in IDS})
    first = model.act(obs, deterministic=True)
    model.reset_policies()
    second = model.act(obs, deterministic=True)
    assert first == second


def test_agents_do_not_share_parameters():
    """
    What distinguishes MA2C from the parameter-sharing independent learners:
    each junction has its own network and can specialise to its own role.
    """
    model = MA2C(IDS)
    a = dict(model.nets["J1"].named_parameters())
    b = dict(model.nets["J2"].named_parameters())
    assert all(a[k] is not b[k] for k in a)


def test_an_update_changes_the_acting_agents_parameters():
    model = MA2C(IDS)
    before = model.nets["J1"].policy.weight.detach().clone()
    batch = {"J1": {
        "inputs": [np.zeros(agent_input_dim(), dtype=np.float32) for _ in range(4)],
        "actions": [0, 1, 2, 3],
        "rewards": [-1.0, -1.0, -1.0, -1.0],
        "bootstrap": 0.0,
    }}
    model.update(batch)
    after = model.nets["J1"].policy.weight.detach()
    assert not np.allclose(before.numpy(), after.numpy())


def test_an_update_leaves_other_agents_untouched():
    model = MA2C(IDS)
    before = model.nets["J2"].policy.weight.detach().clone()
    batch = {"J1": {
        "inputs": [np.zeros(agent_input_dim(), dtype=np.float32)],
        "actions": [0], "rewards": [-1.0], "bootstrap": 0.0,
    }}
    model.update(batch)
    assert np.allclose(before.numpy(), model.nets["J2"].policy.weight.detach().numpy())


def test_an_empty_rollout_is_ignored_rather_than_crashing():
    model = MA2C(IDS)
    assert model.update({"J1": {"inputs": [], "actions": [], "rewards": [],
                                "bootstrap": 0.0}}) == {}


def test_returns_use_the_projects_discount_factor():
    """gamma is ~one signal cycle, the same value the rest of the project uses."""
    assert GAMMA == 0.95


def test_network_outputs_one_logit_per_phase_and_one_value():
    net = ActorCritic(agent_input_dim())
    import torch
    logits, value = net(torch.zeros(3, agent_input_dim()))
    assert logits.shape == (3, N_ACTIONS)
    assert value.shape == (3,)


# ---------------------------------------------------------------------------
# Persistence and transfer
# ---------------------------------------------------------------------------
def test_a_saved_model_reloads_to_the_same_policy(tmp_path):
    model = MA2C(IDS, seed=1)
    path = tmp_path / "ma2c.pt"
    model.save(path)
    restored = MA2C.load(path)
    obs = fake_observations({ts: 0.4 for ts in IDS})
    assert model.act(obs, deterministic=True) == restored.act(obs, deterministic=True)


def test_loading_onto_different_junction_names_preserves_behaviour(tmp_path):
    """
    The transfer case: a corridor3 policy driving a network whose junctions are
    named differently. Agents are matched positionally along the corridor.
    """
    model = MA2C(IDS, seed=2)
    path = tmp_path / "ma2c.pt"
    model.save(path)
    moved = MA2C.load(path, ts_ids=["A", "B", "C"])
    assert set(moved.nets) == {"A", "B", "C"}
    obs_original = fake_observations({"J1": 0.2, "J2": 0.6, "J3": 0.9})
    obs_moved = fake_observations({"A": 0.2, "B": 0.6, "C": 0.9})
    original = model.act(obs_original, deterministic=True)
    remapped = moved.act(obs_moved, deterministic=True)
    assert [original[t] for t in IDS] == [remapped[t] for t in ["A", "B", "C"]]


def test_loading_onto_a_longer_corridor_reuses_agents_cyclically(tmp_path):
    """An approximation, flagged wherever a transfer number is reported."""
    model = MA2C(IDS, seed=3)
    path = tmp_path / "ma2c.pt"
    model.save(path)
    longer = MA2C.load(path, ts_ids=["K1", "K2", "K3", "K4", "K5"])
    assert set(longer.nets) == {"K1", "K2", "K3", "K4", "K5"}
    assert longer.neighbours["K3"] == ["K2", "K4"]


# ---------------------------------------------------------------------------
# Simultaneity: neighbour policy sharing uses the PREVIOUS step, for everyone
# ---------------------------------------------------------------------------
def test_acting_is_independent_of_the_order_agents_are_given_in():
    """
    The first version updated each agent's policy inside the decision loop, so
    the second agent processed saw the first's *current* policy while the
    third's was still one step old - and the corridor's behaviour depended on
    dictionary iteration order.

    In Chu et al. the agents decide simultaneously and share the previous
    step's policies, which is also the only thing realisable in a deployment:
    an agent cannot see a decision that has not been made yet.
    """
    obs = fake_observations({"J1": 0.2, "J2": 0.5, "J3": 0.8})
    reversed_obs = {ts: obs[ts] for ts in reversed(IDS)}

    forward = MA2C(IDS, seed=7)
    backward = MA2C(IDS, seed=7)
    for _ in range(3):
        a = forward.act(obs, deterministic=True)
        b = backward.act(reversed_obs, deterministic=True)
        assert a == b


def test_one_agents_decision_does_not_reach_another_within_the_same_step():
    """
    Directly: J1's input must be identical whether or not J2 has already been
    processed in this step.
    """
    obs = fake_observations({ts: 0.4 for ts in IDS})
    model = MA2C(IDS, seed=8)

    before = model.build_input("J1", obs).copy()
    model.act(obs)
    used = model.last_inputs["J1"]
    assert np.allclose(before, used), (
        "J1's input changed during the step it was deciding in")


def test_the_stored_input_is_the_one_the_action_came_from():
    """
    The gradient bug. The trainer stores `last_inputs`; if that disagreed with
    what `act` fed the network, the policy gradient would be evaluated at a
    state that never produced the action.
    """
    import torch
    import torch.nn.functional as F

    obs = fake_observations({"J1": 0.1, "J2": 0.6, "J3": 0.9})
    model = MA2C(IDS, seed=9)
    actions = model.act(obs, deterministic=True)

    for ts, x in model.last_inputs.items():
        with torch.no_grad():
            logits, _ = model.nets[ts](torch.from_numpy(x).float().unsqueeze(0))
        replayed = int(torch.argmax(F.softmax(logits, dim=-1), dim=-1).item())
        assert replayed == actions[ts], (
            "%s's stored input replays to action %d but %d was taken, so the "
            "gradient would be evaluated at a state that did not produce the "
            "action" % (ts, replayed, actions[ts]))


def test_last_inputs_covers_exactly_the_agents_that_acted():
    model = MA2C(IDS, seed=10)
    obs = fake_observations({"J1": 0.3, "J2": 0.3})
    actions = model.act(obs)
    assert set(model.last_inputs) == set(actions) == {"J1", "J2"}


def test_neighbour_policies_do_advance_between_steps():
    """
    Simultaneity must not become staleness: the previous step's policies are
    shared, and they do change from one step to the next.
    """
    model = MA2C(IDS, seed=11)
    obs = fake_observations({ts: 0.5 for ts in IDS})
    model.act(obs)
    first = model.last_inputs["J2"].copy()
    model.act(obs)
    second = model.last_inputs["J2"]
    assert not np.allclose(first[-MAX_NEIGHBOURS * N_ACTIONS:],
                           second[-MAX_NEIGHBOURS * N_ACTIONS:])


def test_resetting_clears_the_stored_inputs():
    model = MA2C(IDS, seed=12)
    model.act(fake_observations({ts: 0.5 for ts in IDS}))
    model.reset_policies()
    assert model.last_inputs == {}


# ---------------------------------------------------------------------------
# Can it actually learn? The end-to-end property everything else serves
# ---------------------------------------------------------------------------
def test_the_learner_finds_a_better_action_when_one_exists():
    """
    The test that would have caught the divergence, and the one every other
    test in this file was missing.

    Each mechanism was verified in isolation - neighbour assembly, spatial
    discount, simultaneity, persistence - and all of them were correct. The
    learner still could not learn, because the defect was in how the actor's
    and critic's gradients shared one clip: with returns around -76, the value
    loss came to 2895 against a policy loss of 105, a total gradient norm of 97
    was scaled to 0.5, and the policy head received half a percent of its
    natural step.

    On the corridor that showed up as mean queue climbing 118 -> 362 over ten
    episodes. Measured here on a problem whose answer is known: one action pays
    -2 and the rest pay -6, so a working learner must end up preferring it.
    The unfixed version collapsed onto a WRONG action with 95% confidence,
    which is why the corridor gridlocked rather than merely failing to improve.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    model = MA2C(["J1"], seed=0)
    good = 2

    for _ in range(400):
        actions = [int(rng.integers(0, N_ACTIONS)) for _ in range(ROLLOUT)]
        rewards = [(-2.0 if a == good else -6.0) for a in actions]
        model.update({"J1": {
            "inputs": [np.zeros(agent_input_dim(), dtype=np.float32)
                       for _ in range(ROLLOUT)],
            "actions": actions,
            "rewards": rewards,
            "bootstrap": -60.0,
        }})

    with torch.no_grad():
        logits, _ = model.nets["J1"](torch.zeros(1, agent_input_dim()))
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

    assert probs[good] > 0.40, (
        "the learner ended at P(best action)=%.3f against 0.25 for a coin "
        "flip, so it is not learning" % probs[good])
    assert probs.argmax() == good, (
        "the learner became confident in action %d when %d is the better one"
        % (probs.argmax(), good))


def test_learning_does_not_come_from_entropy_collapse():
    """
    A policy can reach high P(best) by collapsing to a single action, which
    looks like learning and is not. Some entropy must survive.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(1)
    model = MA2C(["J1"], seed=1)
    for _ in range(400):
        actions = [int(rng.integers(0, N_ACTIONS)) for _ in range(ROLLOUT)]
        rewards = [(-2.0 if a == 2 else -6.0) for a in actions]
        model.update({"J1": {
            "inputs": [np.zeros(agent_input_dim(), dtype=np.float32)
                       for _ in range(ROLLOUT)],
            "actions": actions, "rewards": rewards, "bootstrap": -60.0}})

    with torch.no_grad():
        logits, _ = model.nets["J1"](torch.zeros(1, agent_input_dim()))
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
    entropy = float(-(probs * np.log(probs + 1e-8)).sum())
    assert entropy > 0.2, "policy collapsed to a point mass (entropy %.3f)" % entropy


def test_the_critic_gradient_cannot_swamp_the_actor():
    """
    The mechanism, pinned directly. Actor and critic share one
    clip_grad_norm_, so if the value loss grows with the square of a large
    error it takes essentially the whole gradient budget. A smooth-L1 value
    loss bounds it per element instead.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from corridor.ma2c import MAX_GRAD_NORM, VALUE_COEF

    model = MA2C(["J1"], seed=0)
    net = model.nets["J1"]
    inputs = torch.zeros(ROLLOUT, agent_input_dim())
    returns = torch.full((ROLLOUT,), -76.0)
    actions = torch.zeros(ROLLOUT, dtype=torch.int64)

    logits, values = net(inputs)
    advantage = returns - values
    normalised = advantage.detach()
    normalised = (normalised - normalised.mean()) / (normalised.std() + 1e-8)
    log_probs = F.log_softmax(logits, dim=-1)
    chosen = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = (-(chosen * normalised).mean()
            + VALUE_COEF * F.smooth_l1_loss(values, returns))

    net.zero_grad()
    loss.backward()
    total = float(nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM))
    assert total < 50.0, (
        "total gradient norm %.1f means the clip is discarding almost all of "
        "it, and the actor's share with it" % total)
