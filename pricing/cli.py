"""Demos.

    python -m pricing.cli demand      the revenue curve: why this is not a regression problem
    python -m pricing.cli regret       LinUCB / Thompson / eps-greedy / flat price, against the optimum
    python -m pricing.cli ope          four estimators against the true policy value
    python -m pricing.cli support      what a greedy logging policy does to off-policy evaluation
    python -m pricing.cli margin       pricing on revenue while paying a unit cost
    python -m pricing.cli all
"""

from __future__ import annotations

import sys

from .environment import Context, PricingEnvironment, uniform_logging_policy
from .evaluation import (
    OPEResult,
    evaluate_all,
    ips,
    run_online,
    snips,
    true_policy_value,
    weight_percentiles,
)
from .policies import (
    EpsilonGreedy,
    FixedPricePolicy,
    LinearThompson,
    LinUCB,
    OraclePolicy,
    UniformPolicy,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def demo_demand() -> None:
    rule("THE REVENUE CURVE -- an interior optimum that moves with the context")
    environment = PricingEnvironment()
    print("segment     best price   P(buy)   expected revenue")
    for segment, price, probability, revenue in environment.optimum_table():
        print(f"{segment:<11} {price:>10.2f}   {probability:>6.1%}   {revenue:>16.3f}")

    print("\nprice ladder for the 'regular' segment:")
    print("price    P(buy)    expected revenue")
    for price, probability, revenue in environment.revenue_curve(
        Context("regular", False, 0.3)
    ):
        bar = "#" * int(revenue * 12)
        print(f"{price:>5.2f}   {probability:>6.1%}   {revenue:>16.3f}  {bar}")
    print(
        "\nRevenue rises then falls. A model fitted to a narrow observed price band learns the\n"
        "left-hand slope and recommends raising the price without limit -- which is how pricing\n"
        "models built on observational data lose money."
    )


def demo_regret() -> None:
    rule("REGRET AGAINST A KNOWN OPTIMUM -- 3,000 rounds")
    rounds = 3000
    print(f"{'policy':<18} {'total regret':>13} {'regret @500':>12} {'revenue vs oracle':>19}")
    for factory in (
        lambda env: UniformPolicy(env, seed=1),
        lambda env: FixedPricePolicy(env, 10.0, seed=1),
        lambda env: FixedPricePolicy(env, 15.0, seed=1),
        lambda env: EpsilonGreedy(env, epsilon=0.1, seed=1),
        lambda env: LinUCB(env, alpha=0.6, seed=1),
        lambda env: LinearThompson(env, variance=0.4, seed=1),
        lambda env: OraclePolicy(env, seed=1),
    ):
        environment = PricingEnvironment(seed=7)  # same stream of customers for every policy
        policy = factory(environment)
        result = run_online(policy, environment, rounds=rounds)
        print(
            f"{policy.name:<18} {result.total_regret:>13.1f} {result.regret_at(500):>12.1f} "
            f"{result.revenue_share:>18.1%}"
        )
    print(
        "\nThe flat prices are the incumbent, and a good flat price is a real competitor -- a bandit\n"
        "that cannot beat it is not worth its operational cost. Uniform exploration shows what\n"
        "learning nothing costs; the oracle shows the ceiling, which noise makes unreachable."
    )


def demo_ope() -> None:
    rule("OFF-POLICY EVALUATION -- four estimators against the truth")
    environment = PricingEnvironment(seed=3)
    log = uniform_logging_policy(environment, rounds=4000, seed=3)
    contexts = [record["context"] for record in log]

    for label, policy in (
        ("a good flat price", FixedPricePolicy(environment, 12.0, seed=2)),
        ("the oracle", OraclePolicy(environment, seed=2)),
        ("a trained LinUCB", None),
    ):
        if policy is None:
            trained_env = PricingEnvironment(seed=11)
            policy = LinUCB(trained_env, alpha=0.6, seed=2)
            run_online(policy, trained_env, rounds=2000)
            policy.environment = environment

        truth = true_policy_value(policy, environment, contexts)
        print(f"\ntarget: {label}   (true value {truth:.3f} per impression)")
        for result in evaluate_all(log, policy, environment.prices):
            print("  " + result.line(truth))
    print(
        "\nThe log is uniform exploration, so support is complete and every estimator is roughly\n"
        "right. That is the easy case, and it is the case almost no production log satisfies."
    )


def demo_support() -> None:
    rule("SUPPORT -- what a greedy logging policy does to off-policy evaluation")
    environment = PricingEnvironment(seed=5)

    # a greedy logging policy: it logs one price, so the log says nothing about any other
    greedy = FixedPricePolicy(environment, 10.0, seed=5)
    _, greedy_log = run_online(greedy, environment, rounds=4000, log=True)

    explorer = LinearThompson(PricingEnvironment(seed=6), variance=0.4, seed=6)
    _, mixed_log = run_online(explorer, PricingEnvironment(seed=6), rounds=4000, log=True)

    target = FixedPricePolicy(environment, 15.0, seed=0)
    contexts = [record["context"] for record in greedy_log]
    truth = true_policy_value(target, environment, contexts)
    print(f"target policy: flat 15.00, true value {truth:.3f} per impression\n")

    for label, log in (("greedy log (flat 10)", greedy_log), ("Thompson log", mixed_log)):
        estimate = ips(log, target, environment.prices)
        normalised = snips(log, target, environment.prices)
        print(f"{label}:")
        print("  " + estimate.line(truth))
        print("  " + normalised.line(truth))
        weights = [
            record["propensity"] for record in log
        ]
        print(f"  logged propensities: min {min(weights):.4f}, max {max(weights):.4f}")

    print(
        "\nThe greedy log gives every record zero target probability, so IPS returns 0.000 -- an\n"
        "estimate of nothing, reported as a number. No amount of extra data fixes it: the log\n"
        "contains no information about a price that was never shown. This is why the logging policy\n"
        "must be stochastic, and why that decision has to be made before the logs are collected."
    )


def demo_margin() -> None:
    rule("REVENUE VS MARGIN -- optimising the wrong objective, priced out")
    revenue_env = PricingEnvironment(unit_cost=0.0, seed=4)
    margin_env = PricingEnvironment(unit_cost=7.0, seed=4)

    print("segment     best on revenue   best on margin")
    for segment in ("bargain", "regular", "premium"):
        context = Context(segment, False, 0.3)
        print(
            f"{segment:<11} {revenue_env.optimal_price(context):>15.2f} "
            f"{margin_env.optimal_price(context):>16.2f}"
        )

    contexts = [margin_env.sample_context() for _ in range(4000)]
    revenue_optimal = FixedPricePolicy(margin_env, revenue_env.optimal_price(Context("regular", False, 0.3)), seed=0)
    margin_optimal = FixedPricePolicy(margin_env, margin_env.optimal_price(Context("regular", False, 0.3)), seed=0)
    lost = true_policy_value(margin_optimal, margin_env, contexts) - true_policy_value(
        revenue_optimal, margin_env, contexts
    )
    print(
        f"\nMargin left on the table by pricing for revenue: {lost:.3f} per impression "
        f"({lost / max(true_policy_value(margin_optimal, margin_env, contexts), 1e-9):.1%} of the optimum)"
    )
    print(
        "With a unit cost, the revenue-optimal price is too low: every marginal sale carries a cost\n"
        "the objective never accounted for. The bandit machinery is indifferent to this -- it will\n"
        "optimise whatever reward you hand it, including the wrong one, and quickly."
    )


DEMOS = {
    "demand": demo_demand,
    "regret": demo_regret,
    "ope": demo_ope,
    "support": demo_support,
    "margin": demo_margin,
}


def main(argv: "list[str] | None" = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    choice = arguments[0] if arguments else "all"
    if choice == "all":
        for demo in DEMOS.values():
            demo()
        return 0
    if choice not in DEMOS:
        print(f"unknown demo {choice!r}\navailable: {', '.join(DEMOS)}, all")
        return 2
    DEMOS[choice]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
