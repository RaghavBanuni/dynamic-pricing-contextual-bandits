"""Tests. Linear algebra identities, then policy behaviour, then the failure modes of OPE.

The load-bearing tests:

* ``test_sherman_morrison_agrees_with_a_full_inverse`` -- the rank-1 update is used on every round, and it
  drifts silently. If it drifts, LinUCB's confidence widths are wrong and nothing else in the repository means
  anything.
* ``test_ips_is_close_to_the_truth_on_a_uniform_log`` -- the estimator, validated against a value only a
  simulator can supply.
* ``test_ips_returns_zero_on_a_greedy_log`` -- the failure that matters in practice, pinned so it cannot be
  mistaken for a small number.
* ``test_effective_sample_size_collapses_with_one_dominant_weight`` -- the diagnostic that tells you which of
  the two situations above you are in.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricing.environment import (  # noqa: E402
    Context,
    PricingEnvironment,
    sigmoid,
    uniform_logging_policy,
)
from pricing.evaluation import (  # noqa: E402
    RewardModel,
    capped_ips,
    direct_method,
    doubly_robust,
    effective_sample_size,
    evaluate_all,
    importance_weights,
    ips,
    run_online,
    snips,
    true_policy_value,
)
from pricing.policies import (  # noqa: E402
    EpsilonGreedy,
    FixedPricePolicy,
    LinearArm,
    LinearThompson,
    LinUCB,
    OraclePolicy,
    UniformPolicy,
    cholesky,
    dot,
    identity,
    invert,
    matrix_vector,
    sherman_morrison,
)


class TestEnvironment:
    def test_sigmoid_does_not_overflow(self):
        assert sigmoid(-800.0) == pytest.approx(0.0)
        assert sigmoid(800.0) == pytest.approx(1.0)

    def test_purchase_probability_falls_with_price(self):
        environment = PricingEnvironment()
        context = Context("regular", False, 0.3)
        probabilities = [environment.purchase_probability(context, p) for p in environment.prices]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_the_revenue_optimum_is_interior(self):
        """If the optimum sat at an endpoint the problem would be trivial and the demo dishonest."""
        environment = PricingEnvironment()
        context = Context("regular", False, 0.3)
        best = environment.optimal_price(context)
        assert environment.prices[0] < best < environment.prices[-1]

    def test_price_sensitive_segments_get_lower_prices(self):
        environment = PricingEnvironment()
        bargain = environment.optimal_price(Context("bargain", False, 0.3))
        premium = environment.optimal_price(Context("premium", False, 0.3))
        assert bargain < premium  # this is the contextual signal a bandit has to find

    def test_regret_is_zero_at_the_optimum_and_positive_elsewhere(self):
        environment = PricingEnvironment()
        context = Context("premium", False, 0.3)
        best = environment.optimal_price(context)
        assert environment.regret(context, best) == pytest.approx(0.0)
        for price in environment.prices:
            if price != best:
                assert environment.regret(context, price) > 0.0

    def test_a_unit_cost_moves_the_optimum_upwards(self):
        """Margin optimisation prices higher than revenue optimisation, always."""
        context = Context("regular", False, 0.3)
        revenue = PricingEnvironment(unit_cost=0.0).optimal_price(context)
        margin = PricingEnvironment(unit_cost=7.0).optimal_price(context)
        assert margin >= revenue

    def test_an_unavailable_price_is_rejected(self):
        with pytest.raises(ValueError, match="not in the action set"):
            PricingEnvironment().pull(Context("regular", False, 0.3), 9.99)

    def test_the_logging_policy_records_a_usable_propensity(self):
        environment = PricingEnvironment()
        log = uniform_logging_policy(environment, rounds=200, seed=0)
        assert all(record["propensity"] == pytest.approx(1 / len(environment.prices)) for record in log)


class TestLinearAlgebra:
    def matrix(self):
        return [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]]

    def test_inverse_times_matrix_is_the_identity(self):
        matrix = self.matrix()
        inverse = invert(matrix)
        product = [
            [sum(matrix[row][k] * inverse[k][column] for k in range(3)) for column in range(3)]
            for row in range(3)
        ]
        for row in range(3):
            for column in range(3):
                assert product[row][column] == pytest.approx(1.0 if row == column else 0.0, abs=1e-9)

    def test_a_singular_matrix_raises(self):
        with pytest.raises(ValueError, match="singular"):
            invert([[1.0, 2.0], [2.0, 4.0]])

    def test_sherman_morrison_agrees_with_a_full_inverse(self):
        """Applied on every round; drifts silently if wrong. Twenty updates is enough to expose drift."""
        size = 4
        A = identity(size, 2.0)
        A_inverse = identity(size, 0.5)
        for index in range(20):
            vector = [math.sin(index + position) + 0.5 * position for position in range(size)]
            for row in range(size):
                for column in range(size):
                    A[row][column] += vector[row] * vector[column]
            A_inverse = sherman_morrison(A_inverse, vector)
        exact = invert(A)
        for row in range(size):
            for column in range(size):
                assert A_inverse[row][column] == pytest.approx(exact[row][column], abs=1e-8)

    def test_cholesky_reproduces_the_matrix(self):
        matrix = self.matrix()
        lower = cholesky(matrix)
        product = [
            [sum(lower[row][k] * lower[column][k] for k in range(3)) for column in range(3)]
            for row in range(3)
        ]
        for row in range(3):
            for column in range(3):
                assert product[row][column] == pytest.approx(matrix[row][column], abs=1e-9)

    def test_cholesky_refuses_a_non_positive_definite_matrix(self):
        with pytest.raises(ValueError, match="non-positive pivot"):
            cholesky([[1.0, 2.0], [2.0, 1.0]])


class TestLinearArm:
    def test_ridge_recovers_a_linear_reward(self):
        """With a small ridge and clean data, theta should be the generating vector."""
        arm = LinearArm(dimension=3, ridge=1e-8)
        truth = [1.5, -2.0, 0.75]
        for first in range(4):
            for second in range(4):
                features = [float(first), float(second), 1.0]
                arm.update(features, dot(truth, features))
        for estimated, actual in zip(arm.theta, truth):
            assert estimated == pytest.approx(actual, abs=1e-4)

    def test_the_confidence_width_shrinks_with_evidence(self):
        arm = LinearArm(dimension=3, ridge=1.0)
        features = [1.0, 0.5, 1.0]
        before = arm.width(features)
        for _ in range(50):
            arm.update(features, 1.0)
        assert arm.width(features) < before / 3.0

    def test_zero_ridge_is_rejected(self):
        with pytest.raises(ValueError, match="ridge must be positive"):
            LinearArm(dimension=2, ridge=0.0)


class TestPolicies:
    def environment(self):
        return PricingEnvironment(seed=0)

    @pytest.mark.parametrize(
        "factory",
        [
            lambda env: UniformPolicy(env),
            lambda env: FixedPricePolicy(env, 10.0),
            lambda env: EpsilonGreedy(env, epsilon=0.2),
            lambda env: LinUCB(env, alpha=1.0),
            lambda env: LinearThompson(env, samples=40),
            lambda env: OraclePolicy(env),
        ],
    )
    def test_action_probabilities_are_a_distribution(self, factory):
        policy = factory(self.environment())
        probabilities = policy.probabilities(Context("regular", False, 0.3))
        assert len(probabilities) == len(policy.prices)
        assert all(value >= 0.0 for value in probabilities)
        assert sum(probabilities) == pytest.approx(1.0)

    def test_epsilon_greedy_keeps_every_arm_reachable(self):
        policy = EpsilonGreedy(self.environment(), epsilon=0.2)
        probabilities = policy.probabilities(Context("regular", False, 0.3))
        assert min(probabilities) >= 0.2 / len(policy.prices) - 1e-12

    def test_linucb_is_deterministic_and_says_so(self):
        policy = LinUCB(self.environment(), alpha=1.0)
        probabilities = policy.probabilities(Context("regular", False, 0.3))
        assert policy.deterministic is True
        assert sorted(probabilities)[-1] == pytest.approx(1.0)

    def test_thompson_sampling_is_stochastic(self):
        """Stochasticity is what makes it loggable, so it is worth asserting rather than assuming."""
        policy = LinearThompson(self.environment(), variance=1.0, seed=0)
        context = Context("regular", False, 0.3)
        actions = {policy.select(context) for _ in range(40)}
        assert len(actions) > 1

    def test_thompson_propensities_are_floored_above_zero(self):
        policy = LinearThompson(self.environment(), variance=0.1, samples=30, seed=0)
        assert min(policy.probabilities(Context("bargain", False, 0.3))) > 0.0

    def test_the_oracle_picks_the_optimal_price(self):
        environment = self.environment()
        policy = OraclePolicy(environment)
        for segment in ("bargain", "regular", "premium"):
            context = Context(segment, False, 0.3)
            assert policy.select(context) == environment.optimal_price(context)

    def test_a_fixed_price_outside_the_ladder_is_rejected(self):
        with pytest.raises(ValueError, match="must be in the action set"):
            FixedPricePolicy(self.environment(), 11.11)


class TestOnlineLearning:
    def test_the_oracle_has_no_regret(self):
        environment = PricingEnvironment(seed=1)
        result = run_online(OraclePolicy(environment), environment, rounds=300)
        assert result.total_regret == pytest.approx(0.0, abs=1e-9)

    def test_learning_policies_beat_uniform_exploration(self):
        rounds = 1500
        regrets = {}
        for name, factory in (
            ("uniform", lambda env: UniformPolicy(env, seed=2)),
            ("linucb", lambda env: LinUCB(env, alpha=0.6, seed=2)),
            ("thompson", lambda env: LinearThompson(env, variance=0.4, samples=1, seed=2)),
        ):
            environment = PricingEnvironment(seed=9)
            regrets[name] = run_online(factory(environment), environment, rounds=rounds).total_regret
        assert regrets["linucb"] < regrets["uniform"]
        assert regrets["thompson"] < regrets["uniform"]

    def test_regret_accumulates_more_slowly_over_time(self):
        """Sublinear regret in the only form worth asserting on one run: the second half costs less."""
        environment = PricingEnvironment(seed=4)
        result = run_online(LinUCB(environment, alpha=0.6, seed=3), environment, rounds=2000)
        first_half = result.regret_at(999)
        second_half = result.total_regret - first_half
        assert second_half < first_half

    def test_the_curve_is_monotone_and_matches_its_length(self):
        environment = PricingEnvironment(seed=5)
        result = run_online(EpsilonGreedy(environment, seed=3), environment, rounds=200)
        assert len(result.cumulative_regret) == 200
        assert result.cumulative_regret == sorted(result.cumulative_regret)

    def test_the_log_records_the_propensity_of_the_action_taken(self):
        environment = PricingEnvironment(seed=6)
        policy = EpsilonGreedy(environment, epsilon=0.3, seed=4)
        _, log = run_online(policy, environment, rounds=100, log=True)
        assert len(log) == 100
        assert all(0.0 < record["propensity"] <= 1.0 for record in log)


class TestImportanceWeights:
    def test_a_point_mass_target_on_a_uniform_log_gives_weight_k_or_zero(self):
        environment = PricingEnvironment(seed=0)
        log = uniform_logging_policy(environment, rounds=300, seed=0)
        target = FixedPricePolicy(environment, 12.0)
        weights, _ = importance_weights(log, target, environment.prices)
        assert set(round(weight, 6) for weight in weights) <= {0.0, float(len(environment.prices))}

    def test_the_weights_average_to_about_one(self):
        """A necessary condition: importance weights are a change of measure, not a scaling factor."""
        environment = PricingEnvironment(seed=0)
        log = uniform_logging_policy(environment, rounds=4000, seed=0)
        weights, _ = importance_weights(log, OraclePolicy(environment), environment.prices)
        assert sum(weights) / len(weights) == pytest.approx(1.0, abs=0.1)

    def test_effective_sample_size_equals_n_for_uniform_weights(self):
        assert effective_sample_size([1.0] * 500) == pytest.approx(500.0)

    def test_effective_sample_size_collapses_with_one_dominant_weight(self):
        weights = [0.001] * 999 + [1000.0]
        assert effective_sample_size(weights) < 2.0

    def test_effective_sample_size_of_nothing_is_zero(self):
        assert effective_sample_size([0.0, 0.0]) == 0.0


class TestOffPolicyEvaluation:
    def setup_method(self):
        self.environment = PricingEnvironment(seed=3)
        self.log = uniform_logging_policy(self.environment, rounds=6000, seed=3)
        self.contexts = [record["context"] for record in self.log]

    def test_ips_is_close_to_the_truth_on_a_uniform_log(self):
        target = FixedPricePolicy(self.environment, 12.0)
        truth = true_policy_value(target, self.environment, self.contexts)
        estimate = ips(self.log, target, self.environment.prices)
        assert estimate.value == pytest.approx(truth, rel=0.15)
        assert estimate.support_violations == 0

    def test_every_estimator_lands_near_the_truth_when_support_is_complete(self):
        target = OraclePolicy(self.environment)
        truth = true_policy_value(target, self.environment, self.contexts)
        for result in evaluate_all(self.log, target, self.environment.prices):
            assert result.value == pytest.approx(truth, rel=0.25), result.estimator

    def test_snips_cannot_exceed_the_largest_logged_reward(self):
        target = FixedPricePolicy(self.environment, self.environment.prices[-1])
        largest = max(record["reward"] for record in self.log)
        assert snips(self.log, target, self.environment.prices).value <= largest + 1e-9

    def test_capping_reduces_the_maximum_weight(self):
        target = OraclePolicy(self.environment)
        uncapped = ips(self.log, target, self.environment.prices)
        capped = capped_ips(self.log, target, self.environment.prices, cap=2.0)
        assert capped.max_weight <= 2.0 <= uncapped.max_weight

    def test_doubly_robust_reduces_to_the_model_when_weights_are_capped_to_zero(self):
        """A structural check on the decomposition: with no importance term, DR is the direct method."""
        target = FixedPricePolicy(self.environment, 12.0)
        model = RewardModel(self.environment.prices, len(self.contexts[0].features)).fit(self.log)
        model_only = direct_method(self.log, target, self.environment.prices, model)
        no_correction = doubly_robust(self.log, target, self.environment.prices, model, cap=0.0)
        assert no_correction.value == pytest.approx(model_only.value, abs=1e-9)

    def test_ips_returns_zero_on_a_greedy_log(self):
        """The failure that matters: an estimate of nothing, presented as a number."""
        environment = PricingEnvironment(seed=5)
        _, greedy_log = run_online(
            FixedPricePolicy(environment, 10.0), environment, rounds=800, log=True
        )
        target = FixedPricePolicy(environment, 18.0)
        estimate = ips(greedy_log, target, environment.prices)
        assert estimate.value == pytest.approx(0.0)
        assert estimate.effective_sample_size == pytest.approx(0.0)
        # and the truth is emphatically not zero
        contexts = [record["context"] for record in greedy_log]
        assert true_policy_value(target, environment, contexts) > 1.0

    def test_a_narrow_log_gives_a_low_effective_sample_size(self):
        """Between full support and none: the ESS is what makes the difference visible."""
        environment = PricingEnvironment(seed=7)
        policy = EpsilonGreedy(environment, epsilon=0.05, seed=7)
        _, narrow_log = run_online(policy, environment, rounds=3000, log=True)
        target = OraclePolicy(environment)
        narrow = ips(narrow_log, target, environment.prices)
        wide = ips(self.log, target, self.environment.prices)
        assert narrow.ess_share < wide.ess_share

    def test_the_reward_model_reports_its_coverage(self):
        model = RewardModel(self.environment.prices, len(self.contexts[0].features)).fit(self.log)
        coverage = model.coverage()
        assert sum(coverage.values()) == len(self.log)
        assert all(count > 0 for count in coverage.values())  # uniform logging covers everything

    def test_true_policy_value_prefers_the_oracle(self):
        oracle = true_policy_value(OraclePolicy(self.environment), self.environment, self.contexts)
        flat = true_policy_value(
            FixedPricePolicy(self.environment, 8.0), self.environment, self.contexts
        )
        assert oracle > flat
