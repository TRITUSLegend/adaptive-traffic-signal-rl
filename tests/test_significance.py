"""
Tests for the paired significance tests (eval/significance.py).

WHY THESE TESTS EXIST
---------------------
This module implements Student's t distribution and an exact Wilcoxon
signed-rank test from scratch, so that the project does not gain a SciPy
dependency for eighty lines of statistics. Hand-rolled statistics is exactly
the kind of code that returns a plausible-looking wrong number, and a wrong
p-value would be worse than no p-value at all - it would put a false claim of
significance into the report.

So the implementations are checked against values that can be verified
independently: closed forms the t distribution must satisfy exactly, published
reference p-values, and - for Wilcoxon - the enumeration of a small case that
can be counted by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.significance import (  # noqa: E402
    exact_wilcoxon,
    paired_differences,
    paired_t_test,
    regularised_incomplete_beta,
    t_sf_two_sided,
)


# ---------------------------------------------------------------------------
# The incomplete beta function
# ---------------------------------------------------------------------------
def test_incomplete_beta_endpoints():
    assert regularised_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert regularised_incomplete_beta(2.0, 3.0, 1.0) == 1.0


def test_incomplete_beta_is_symmetric_at_the_half():
    """I_0.5(a, a) = 0.5 for any a, by symmetry of the beta density."""
    for a in (0.5, 1.0, 2.5, 10.0):
        assert regularised_incomplete_beta(a, a, 0.5) == pytest.approx(0.5)


def test_incomplete_beta_matches_a_closed_form():
    """For b = 1, I_x(a, 1) = x**a exactly."""
    for a in (1.0, 2.0, 3.5):
        for x in (0.2, 0.5, 0.9):
            assert regularised_incomplete_beta(a, 1.0, x) == pytest.approx(x ** a)


def test_incomplete_beta_is_monotone_in_x():
    values = [regularised_incomplete_beta(2.0, 3.0, x)
              for x in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert values == sorted(values)


# ---------------------------------------------------------------------------
# The t distribution
# ---------------------------------------------------------------------------
def test_t_of_zero_is_certain():
    assert t_sf_two_sided(0.0, 4) == 1.0


def test_t_is_symmetric():
    assert t_sf_two_sided(2.3, 7) == pytest.approx(t_sf_two_sided(-2.3, 7))


def test_t_with_one_degree_of_freedom_matches_the_cauchy_closed_form():
    """
    At df=1 the t distribution is Cauchy, whose two-sided tail has the closed
    form 1 - (2/pi) * arctan(|t|).
    """
    import math
    for t in (0.5, 1.0, 2.0, 5.0):
        expected = 1.0 - (2.0 / math.pi) * math.atan(abs(t))
        assert t_sf_two_sided(t, 1) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("t,df,expected", [
    # Published critical values: these t are the 0.05 two-sided points.
    (2.776445, 4, 0.05),
    (2.228139, 10, 0.05),
    (2.570582, 5, 0.05),
    (4.604095, 4, 0.01),
    (3.169273, 10, 0.01),
])
def test_t_reproduces_published_critical_values(t, df, expected):
    assert t_sf_two_sided(t, df) == pytest.approx(expected, abs=1e-5)


def test_larger_t_means_smaller_p():
    ps = [t_sf_two_sided(t, 4) for t in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert ps == sorted(ps, reverse=True)


# ---------------------------------------------------------------------------
# The paired t-test
# ---------------------------------------------------------------------------
def test_a_constant_difference_is_perfectly_significant():
    """No variance and a non-zero mean: the difference is certain."""
    result = paired_t_test([-3.0] * 5)
    assert result["p"] == 0.0
    assert result["mean"] == -3.0


def test_all_zero_differences_are_not_significant():
    result = paired_t_test([0.0] * 5)
    assert result["p"] == 1.0


def test_paired_t_matches_a_hand_computed_case():
    """
    differences = [1, 2, 3, 4, 5]: mean 3, sd sqrt(2.5), n 5.
    t = 3 / (sqrt(2.5)/sqrt(5)) = 3 / 0.70710678 = 4.2426407 on 4 df.
    """
    result = paired_t_test([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["mean"] == pytest.approx(3.0)
    assert result["t"] == pytest.approx(4.2426407, rel=1e-6)
    # Cross-checked against Simpson integration of the t density: 0.0132355993.
    assert result["p"] == pytest.approx(0.0132356, abs=1e-6)
    assert result["dz"] == pytest.approx(3.0 / (2.5 ** 0.5))


def test_sign_of_the_mean_difference_is_preserved():
    assert paired_t_test([-1.0, -2.0, -3.0])["mean"] < 0
    assert paired_t_test([1.0, 2.0, 3.0])["mean"] > 0


def test_too_few_observations_gives_nan_rather_than_a_number():
    result = paired_t_test([2.0])
    assert result["n"] == 1
    assert result["p"] != result["p"]  # NaN


# ---------------------------------------------------------------------------
# The exact Wilcoxon test
# ---------------------------------------------------------------------------
def test_wilcoxon_floor_at_five_observations():
    """
    THE POINT THE ADDENDUM MAKES. All five differences of the same sign is the
    most extreme outcome available, and it still only reaches 2/32 = 0.0625.
    """
    result = exact_wilcoxon([-1.0, -2.0, -3.0, -4.0, -5.0])
    assert result["n"] == 5
    assert result["p"] == pytest.approx(2 / 32)
    assert result["floor"] == pytest.approx(2 / 32)
    assert result["p"] > 0.05  # cannot be significant at this n


def test_wilcoxon_cannot_beat_its_floor_for_any_input():
    """No arrangement of five differences produces a p below the floor."""
    import itertools
    for values in itertools.product((-3.0, -1.0, 1.0, 4.0), repeat=5):
        assert exact_wilcoxon(list(values))["p"] >= 2 / 32 - 1e-12


def test_wilcoxon_is_symmetric_under_negation():
    a = [1.0, -2.0, 3.0, 4.0, -5.0]
    b = [-x for x in a]
    assert exact_wilcoxon(a)["p"] == pytest.approx(exact_wilcoxon(b)["p"])


def test_wilcoxon_by_hand_on_three_observations():
    """
    differences [1, 2, 3]: ranks 1, 2, 3, all positive, so W+ = 6 and the null
    mean is 3. Of the 8 sign assignments, exactly two (all-positive and
    all-negative) are at least as far from 3 as 6 is, so p = 2/8.
    """
    result = exact_wilcoxon([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["w"] == pytest.approx(6.0)
    assert result["p"] == pytest.approx(2 / 8)


def test_wilcoxon_drops_zero_differences():
    """A tie carries no information about direction, so it leaves the sample."""
    assert exact_wilcoxon([0.0, 1.0, 2.0, 3.0])["n"] == 3
    assert exact_wilcoxon([0.0, 0.0])["n"] == 0
    assert exact_wilcoxon([0.0, 0.0])["p"] == 1.0


def test_wilcoxon_handles_tied_magnitudes_with_average_ranks():
    result = exact_wilcoxon([2.0, 2.0, -2.0])
    assert result["n"] == 3
    # All three magnitudes tie, so each takes the average rank 2.0.
    assert result["w"] == pytest.approx(4.0)


def test_a_balanced_sample_is_not_significant():
    result = exact_wilcoxon([1.0, -1.0, 2.0, -2.0, 0.5])
    assert result["p"] > 0.5


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def test_pairing_matches_on_seed_and_skips_unmatched_ones():
    raw = [
        {"controller": "dqn", "scenario": "peak", "seed": "100",
         "mean_waiting_time_s": "39.0"},
        {"controller": "dqn", "scenario": "peak", "seed": "101",
         "mean_waiting_time_s": "40.0"},
        {"controller": "dqn", "scenario": "peak", "seed": "102",
         "mean_waiting_time_s": "41.0"},   # no partner
        {"controller": "greedy-queue", "scenario": "peak", "seed": "100",
         "mean_waiting_time_s": "52.0"},
        {"controller": "greedy-queue", "scenario": "peak", "seed": "101",
         "mean_waiting_time_s": "54.0"},
        {"controller": "greedy-queue", "scenario": "balanced", "seed": "100",
         "mean_waiting_time_s": "18.0"},   # different scenario
    ]
    diffs = paired_differences(raw, "dqn", "greedy-queue", "peak",
                               "mean_waiting_time_s")
    assert diffs == [pytest.approx(-13.0), pytest.approx(-14.0)]


def test_pairing_is_ordered_by_seed_not_by_file_order():
    raw = [
        {"controller": "a", "scenario": "s", "seed": "101",
         "mean_waiting_time_s": "5.0"},
        {"controller": "a", "scenario": "s", "seed": "100",
         "mean_waiting_time_s": "1.0"},
        {"controller": "b", "scenario": "s", "seed": "100",
         "mean_waiting_time_s": "0.0"},
        {"controller": "b", "scenario": "s", "seed": "101",
         "mean_waiting_time_s": "0.0"},
    ]
    assert paired_differences(raw, "a", "b", "s", "mean_waiting_time_s") == [1.0, 5.0]
