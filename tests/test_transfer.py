"""
Tests for corridor/transfer.py - zero-shot transfer onto an unseen network.

WHY THESE TESTS EXIST
---------------------
The projection that lets a 21-value policy drive `cologne3`'s 14- and
17-value junctions is the single most dangerous piece of the robustness study,
because it cannot fail loudly. If the observation were assembled in the wrong
order, or the phase one-hot were shifted, the policy would still emit legal
actions and the corridor would still produce a number. That number would be
reported as "degradation under topology transfer" when part of it was a
scrambled input.

So the projection is pinned on hand-computable vectors, and the identity case -
a junction that already matches - is asserted to be exactly a no-op, because
that is what makes the *clean* transfer results uncontaminated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corridor.transfer import (  # noqa: E402
    TRAINED_LANES,
    TRAINED_OBS,
    TRAINED_PHASES,
    adapt_observation,
    clamp_action,
    cologne3_junction_shapes,
    describe_adaptation,
)


def observation(n_phases: int, n_lanes: int, active_phase: int = 0,
                density: float = 0.3, queue: float = 0.6) -> np.ndarray:
    """A junction observation in sumo-rl's layout, with distinctive parts."""
    phases = [1.0 if i == active_phase else 0.0 for i in range(n_phases)]
    return np.array(phases + [1.0] + [density] * n_lanes + [queue] * n_lanes,
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# The identity case - what keeps clean transfer clean
# ---------------------------------------------------------------------------
def test_a_matching_junction_is_left_exactly_unchanged():
    """
    corridor2 and corridor5 junctions already have 4 phases and 8 lanes, so
    the projection must be a no-op there. If it were not, the clean transfer
    results would silently include an adaptation cost and stop being clean.
    """
    obs = observation(TRAINED_PHASES, TRAINED_LANES, active_phase=2)
    out = adapt_observation(obs, TRAINED_PHASES, TRAINED_LANES)
    assert np.array_equal(out, obs)


def test_output_is_always_the_trained_width():
    for n_phases, n_lanes in [(3, 5), (4, 6), (4, 8), (5, 12), (2, 3)]:
        out = adapt_observation(observation(n_phases, n_lanes),
                                n_phases, n_lanes)
        assert out.shape == (TRAINED_OBS,)


def test_the_trained_width_is_what_the_policy_expects():
    assert TRAINED_OBS == TRAINED_PHASES + 1 + 2 * TRAINED_LANES == 21


# ---------------------------------------------------------------------------
# Padding a smaller junction
# ---------------------------------------------------------------------------
def test_a_smaller_junction_keeps_its_parts_in_the_right_slots():
    """cologne3's 360082: 3 phases, 5 lanes -> a 14-value observation."""
    obs = observation(3, 5, active_phase=1, density=0.2, queue=0.7)
    out = adapt_observation(obs, 3, 5)

    assert np.allclose(out[:TRAINED_PHASES], [0.0, 1.0, 0.0, 0.0])
    assert out[TRAINED_PHASES] == 1.0                       # min-green flag
    densities = out[TRAINED_PHASES + 1:TRAINED_PHASES + 1 + TRAINED_LANES]
    queues = out[TRAINED_PHASES + 1 + TRAINED_LANES:]
    assert np.allclose(densities[:5], 0.2)
    assert np.allclose(densities[5:], 0.0)                  # padding
    assert np.allclose(queues[:5], 0.7)
    assert np.allclose(queues[5:], 0.0)


def test_padding_never_leaks_density_into_the_queue_block():
    """
    The failure that would be invisible: densities spilling into the queue
    slots would give the policy a coherent-looking but wrong input.
    """
    obs = observation(3, 5, density=0.9, queue=0.1)
    out = adapt_observation(obs, 3, 5)
    queues = out[TRAINED_PHASES + 1 + TRAINED_LANES:]
    assert not np.any(np.isclose(queues, 0.9))


def test_the_min_green_flag_is_never_confused_with_a_phase():
    """It sits between the one-hot and the densities and must stay there."""
    for n_phases in (2, 3, 4):
        obs = observation(n_phases, 8)
        obs[n_phases] = 0.0                     # min green NOT yet elapsed
        out = adapt_observation(obs, n_phases, 8)
        assert out[TRAINED_PHASES] == 0.0


def test_an_inactive_phase_stays_inactive_after_padding():
    obs = observation(3, 5, active_phase=2)
    out = adapt_observation(obs, 3, 5)
    assert out[3] == 0.0                        # the phase that does not exist
    assert out[2] == 1.0


# ---------------------------------------------------------------------------
# Truncating a larger junction
# ---------------------------------------------------------------------------
def test_a_larger_junction_is_truncated_to_what_the_policy_can_see():
    obs = observation(4, 12, density=0.4, queue=0.8)
    out = adapt_observation(obs, 4, 12)
    densities = out[TRAINED_PHASES + 1:TRAINED_PHASES + 1 + TRAINED_LANES]
    assert len(densities) == TRAINED_LANES
    assert np.allclose(densities, 0.4)


def test_truncation_loses_lanes_rather_than_shifting_them():
    """
    Lane 0 must still be lane 0. A shift would be the worst possible failure:
    every value plausible, every one attached to the wrong approach.
    """
    n_lanes = 12
    obs = observation(4, n_lanes)
    densities_in = np.arange(n_lanes, dtype=np.float32) / 100.0
    obs[5:5 + n_lanes] = densities_in
    out = adapt_observation(obs, 4, n_lanes)
    densities_out = out[TRAINED_PHASES + 1:TRAINED_PHASES + 1 + TRAINED_LANES]
    assert np.allclose(densities_out, densities_in[:TRAINED_LANES])


# ---------------------------------------------------------------------------
# Action clamping
# ---------------------------------------------------------------------------
def test_an_action_a_junction_cannot_serve_is_clamped_into_range():
    assert clamp_action(3, 3) == 2
    assert clamp_action(7, 4) == 3


def test_a_legal_action_passes_through_untouched():
    for action in range(4):
        assert clamp_action(action, 4) == action


def test_clamping_never_returns_a_negative_phase():
    assert clamp_action(-1, 4) == 0


def test_clamping_prefers_the_adjacent_phase_over_wrapping():
    """
    Wrapping would turn "serve the last phase" into "serve the first", which is
    a different instruction rather than a nearby one.
    """
    assert clamp_action(3, 3) == 2      # not 0


# ---------------------------------------------------------------------------
# cologne3's actual shape, as reported
# ---------------------------------------------------------------------------
def test_cologne3_shapes_match_what_the_documentation_claims():
    """
    The report states 3/4/4 green phases and 5/6/8 lanes. Read it from the
    network so the claim cannot go stale.
    """
    if not (ROOT / ".venv" / "Lib" / "site-packages" / "sumo_rl" / "nets"
            / "RESCO" / "cologne3" / "cologne3.net.xml").exists():
        pytest.skip("cologne3 not installed")
    shapes = cologne3_junction_shapes()
    assert len(shapes) == 3
    assert sorted(s["green_phases"] for s in shapes.values()) == [3, 4, 4]
    assert sorted(s["lanes"] for s in shapes.values()) == [5, 6, 8]


def test_at_least_one_cologne3_junction_genuinely_needs_adapting():
    """If they all matched, the projection would be dead code and the report's
    caveat about it would be misleading."""
    if not (ROOT / ".venv" / "Lib" / "site-packages" / "sumo_rl" / "nets"
            / "RESCO" / "cologne3" / "cologne3.net.xml").exists():
        pytest.skip("cologne3 not installed")
    shapes = cologne3_junction_shapes()
    mismatched = [s for s in shapes.values()
                  if s["green_phases"] != TRAINED_PHASES
                  or s["lanes"] != TRAINED_LANES]
    assert mismatched, "no cologne3 junction needs adaptation"


def test_the_adaptation_description_names_every_junction():
    if not (ROOT / ".venv" / "Lib" / "site-packages" / "sumo_rl" / "nets"
            / "RESCO" / "cologne3" / "cologne3.net.xml").exists():
        pytest.skip("cologne3 not installed")
    text = describe_adaptation()
    for ts_id in cologne3_junction_shapes():
        assert ts_id in text


def test_the_bundled_cologne3_network_is_never_modified():
    """
    The real-data provenance depends on the dependency's file being untouched;
    the actuated variant is written into our own tree instead.
    """
    from corridor import paths
    from corridor.transfer import COLOGNE3_NET, cologne3_actuated_net

    if not COLOGNE3_NET.exists():
        pytest.skip("cologne3 not installed")
    before = COLOGNE3_NET.read_bytes()
    out = cologne3_actuated_net()
    assert COLOGNE3_NET.read_bytes() == before
    assert paths.NETWORKS_DIR in out.parents
