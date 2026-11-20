"""Bootstrap intervals, paired tests, McNemar, Holm and Cohen's kappa.

Reference values are computed by hand or from closed forms in each test.
"""

import math

import pytest

from aire.evaluator.stats import (
    bootstrap_mean_ci,
    cohens_kappa,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap,
)


def test_bootstrap_interval_is_seeded_and_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 50
    lo, hi = bootstrap_mean_ci(values, seed=0)
    assert lo is not None and hi is not None
    assert lo < 0.5 < hi
    assert (lo, hi) == bootstrap_mean_ci(values, seed=0)
    # Close to the normal approximation 0.5 +- 1.96 * 0.5 / 10 = [0.402, 0.598].
    assert lo == pytest.approx(0.402, abs=0.03)
    assert hi == pytest.approx(0.598, abs=0.03)


def test_a_constant_sample_has_a_degenerate_interval() -> None:
    assert bootstrap_mean_ci([1.0] * 20) == (1.0, 1.0)
    assert bootstrap_mean_ci([]) == (None, None)


def test_paired_bootstrap_detects_a_consistent_drop() -> None:
    base = [1.0] * 40
    cand = [1.0] * 10 + [0.0] * 30
    result = paired_bootstrap(base, cand, seed=0)
    assert result.delta == pytest.approx(-0.75)
    assert result.ci_high < 0
    assert result.p_value < 0.001
    assert result.n == 40


def test_paired_bootstrap_of_identical_runs_is_exactly_zero() -> None:
    values = [0.2, 0.9, 0.4, 0.7] * 5
    result = paired_bootstrap(values, values)
    assert (result.delta, result.ci_low, result.ci_high, result.p_value) == (0.0, 0.0, 0.0, 1.0)


def test_paired_bootstrap_interval_excludes_zero_exactly_when_p_is_small() -> None:
    base = [0.5] * 30
    cand = [0.5 + 0.01 * ((i % 7) - 2) for i in range(30)]
    r = paired_bootstrap(base, cand, seed=3)
    assert (r.ci_low > 0 or r.ci_high < 0) == (r.p_value < 0.05)


def test_paired_bootstrap_needs_equal_lengths() -> None:
    with pytest.raises(ValueError, match="same number"):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_mcnemar_exact_matches_the_binomial_tail() -> None:
    # 6 discordant pairs, all one way: two-sided p = 2 * 0.5**6.
    assert mcnemar_exact(6, 0) == pytest.approx(2 * 0.5**6)
    # Balanced discordance is no evidence at all.
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(0, 0) == 1.0
    # 2 vs 8: 2 * P(X <= 2), X ~ Bin(10, 0.5) = 2 * 56 / 1024.
    assert mcnemar_exact(2, 8) == pytest.approx(2 * 56 / 1024)


def test_holm_adjustment_is_monotone_and_capped() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03, 0.5])
    # Sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> 0.09 (monotone), 0.5*1=0.5.
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert holm_adjust([0.9, 0.9]) == [1.0, 1.0]
    assert holm_adjust([]) == []


def test_cohens_kappa_against_a_hand_computed_table() -> None:
    # 20 items: both yes 9, both no 6, a-yes/b-no 3, a-no/b-yes 2.
    a = ["y"] * 9 + ["n"] * 6 + ["y"] * 3 + ["n"] * 2
    b = ["y"] * 9 + ["n"] * 6 + ["n"] * 3 + ["y"] * 2
    po = 15 / 20
    pe = (12 / 20) * (11 / 20) + (8 / 20) * (9 / 20)
    assert cohens_kappa(a, b) == pytest.approx((po - pe) / (1 - pe))


def test_kappa_edge_cases() -> None:
    assert cohens_kappa(["y", "n"], ["y", "n"]) == 1.0
    assert cohens_kappa(["y", "y"], ["y", "y"]) == 1.0  # perfect agreement on one label
    assert math.isnan(cohens_kappa([], []))
    with pytest.raises(ValueError):
        cohens_kappa(["y"], ["y", "n"])


def test_every_mean_in_the_results_gets_a_seeded_interval() -> None:
    from aire.evaluator.engine import EvaluationEngine
    from aire.runner.dataset import EvalCase
    from aire.schema import Span, Trace

    cases = [
        EvalCase(case_id=f"c{i}", input="q", expected_keywords=["30 days"], must_cite=False)
        for i in range(12)
    ]
    traces = [
        Trace(
            run_id="r",
            case_id=f"c{i}",
            system_config={},
            final_answer="Refunds take 30 days." if i % 3 else "No idea.",
            spans=[Span("generation", "x", 0.0, 10.0 + i)],
        )
        for i in range(12)
    ]
    first = EvaluationEngine().evaluate_run(traces, cases)["aggregates"]
    second = EvaluationEngine().evaluate_run(traces, cases)["aggregates"]
    means = [k.removeprefix("mean_") for k in first if k.startswith("mean_")]
    assert means
    for metric in means:
        lo, hi = first[f"ci95_{metric}"]
        assert lo <= first[f"mean_{metric}"] <= hi
    assert first["ci95_keyword_recall"] == second["ci95_keyword_recall"]
    assert first["ci95_keyword_recall"][0] < first["ci95_keyword_recall"][1]
