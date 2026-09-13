"""Running policies online, and evaluating them offline -- the second being where most projects go wrong.

**Online** is easy to measure here because the environment knows the optimum: cumulative expected regret,
computed on expectations rather than realised rewards.

**Offline** is the real problem. You have a log from the policy that was running and you want the value of a
policy that was not. Four estimators, in the order anyone should reach for them:

* **Direct method (DM).** Fit a reward model, evaluate the new policy against the model. Zero variance, and
  biased by exactly however wrong the model is -- including on the actions the log barely covers, which are
  precisely the actions a new policy tends to prefer. That is not a coincidence: a new policy is interesting
  because it disagrees with the old one.
* **IPS.** Reweight logged rewards by ``pi_new(a|x) / pi_log(a|x)``. Unbiased when the log has full support,
  and its variance is unbounded when propensities get small. A single record with propensity 0.01 carries a
  weight of 100.
* **Capped / self-normalised IPS.** Clip weights, or divide by their sum. Both trade a little bias for a lot of
  variance; SNIPS additionally cannot exceed the largest logged reward, which removes the most embarrassing
  failure mode -- an estimate above the maximum attainable value.
* **Doubly robust.** Model the reward *and* correct it with importance weights on the residual. Consistent if
  **either** the model or the propensities are right, and lower variance than IPS because the weights multiply
  a residual instead of the whole reward.

And the diagnostic that decides whether any of them mean anything: **effective sample size**.

    ESS = (sum w)^2 / sum w^2

Ten thousand logged records with an ESS of 40 contain about as much information about the new policy as forty
records. Reporting a confidence interval from ``n = 10,000`` in that situation is not a small error. Every
estimator here returns its ESS and its weight tail alongside its point estimate, because the estimate without
them is not interpretable -- and a **support violation** (the new policy choosing an action the log gives zero
probability) makes IPS structurally biased no matter how much data there is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .environment import Context, PricingEnvironment
from .policies import LinearArm, Policy, dot


# ---------------------------------------------------------------------------------------------
# online
# ---------------------------------------------------------------------------------------------


@dataclass
class OnlineResult:
    name: str
    cumulative_regret: "list[float]"
    cumulative_reward: float
    optimal_reward: float
    chosen: "dict[float, int]"

    @property
    def total_regret(self) -> float:
        return self.cumulative_regret[-1] if self.cumulative_regret else 0.0

    @property
    def revenue_share(self) -> float:
        """Realised reward as a share of what the oracle would have earned in expectation."""
        return self.cumulative_reward / self.optimal_reward if self.optimal_reward else 0.0

    def regret_at(self, round_index: int) -> float:
        return self.cumulative_regret[min(round_index, len(self.cumulative_regret) - 1)]


def run_online(
    policy: Policy, environment: PricingEnvironment, rounds: int = 2000, log: bool = False
):
    """Run a policy and accumulate expected regret. Returns ``OnlineResult`` (and the log, if asked).

    The interaction log records the propensity of the action taken, which is the only thing that makes the log
    reusable later. Recording it costs nothing at the time and is impossible to reconstruct afterwards -- the
    single most valuable line in a production bandit deployment.
    """
    regret = 0.0
    curve: list[float] = []
    total = 0.0
    optimal_total = 0.0
    chosen: dict[float, int] = dict.fromkeys(environment.prices, 0)
    records: list[dict] = []

    for _ in range(rounds):
        context = environment.sample_context()
        probabilities = policy.probabilities(context)
        price = policy.select(context)
        index = environment.prices.index(price)
        reward, bought = environment.pull(context, price)
        policy.update(context, price, reward)

        regret += environment.regret(context, price)
        curve.append(regret)
        total += reward
        optimal_total += environment.optimal_reward(context)
        chosen[price] += 1
        if log:
            records.append(
                {
                    "context": context,
                    "price": price,
                    "reward": reward,
                    "bought": bought,
                    "propensity": probabilities[index],
                }
            )

    result = OnlineResult(policy.name, curve, total, optimal_total, chosen)
    return (result, records) if log else result


def true_policy_value(
    policy: Policy, environment: PricingEnvironment, contexts: "list[Context]"
) -> float:
    """The ground truth an OPE estimate is checked against: expected reward under the true model.

    Available only because this is a simulator. In production there is no such column, which is why OPE
    diagnostics matter more than OPE point estimates.
    """
    total = 0.0
    for context in contexts:
        probabilities = policy.probabilities(context)
        total += sum(
            probability * environment.expected_reward(context, price)
            for probability, price in zip(probabilities, environment.prices)
        )
    return total / len(contexts) if contexts else 0.0


# ---------------------------------------------------------------------------------------------
# off-policy evaluation
# ---------------------------------------------------------------------------------------------


@dataclass
class OPEResult:
    """An estimate is not a number: it is a number plus the reasons it might be wrong."""

    estimator: str
    value: float
    effective_sample_size: float
    max_weight: float
    support_violations: int
    records: int

    @property
    def ess_share(self) -> float:
        return self.effective_sample_size / self.records if self.records else 0.0

    def line(self, truth: float | None = None) -> str:
        error = "" if truth is None else f"   error {self.value - truth:+.3f}"
        return (
            f"{self.estimator:<18} {self.value:>8.3f}   ESS {self.effective_sample_size:>8.1f} "
            f"({self.ess_share:>5.1%})   max w {self.max_weight:>7.1f}   "
            f"violations {self.support_violations:>5}{error}"
        )


def importance_weights(
    log: "list[dict]", target: Policy, prices: "tuple[float, ...]"
) -> "tuple[list[float], int]":
    """Per-record weights ``pi_new(a|x) / pi_log(a|x)``, and the count of support violations.

    A support violation is a record where the target policy would take an action the logging policy could never
    have taken. IPS silently assigns those a weight of zero -- the estimator does not fail, it just answers a
    different question, one about the part of the target policy the log happens to cover.
    """
    weights: list[float] = []
    violations = 0
    for record in log:
        probabilities = target.probabilities(record["context"])
        index = prices.index(record["price"])
        logged = record["propensity"]
        if logged <= 0.0:
            violations += 1
            weights.append(0.0)
            continue
        weights.append(probabilities[index] / logged)
        # the reverse direction: an action the target wants that the log cannot supply
        for position, probability in enumerate(probabilities):
            if probability > 0.0 and record.get("logging_support") is not None:
                if not record["logging_support"][position]:
                    violations += 1
                    break
    return weights, violations


def effective_sample_size(weights: "list[float]") -> float:
    """``(sum w)^2 / sum w^2``. Equal to n for uniform weights, and far below it otherwise."""
    total = sum(weights)
    squares = sum(weight**2 for weight in weights)
    return (total**2 / squares) if squares > 0 else 0.0


def ips(log: "list[dict]", target: Policy, prices: "tuple[float, ...]") -> OPEResult:
    weights, violations = importance_weights(log, target, prices)
    value = sum(weight * record["reward"] for weight, record in zip(weights, log)) / len(log)
    return OPEResult(
        "IPS", value, effective_sample_size(weights), max(weights, default=0.0), violations, len(log)
    )


def capped_ips(
    log: "list[dict]", target: Policy, prices: "tuple[float, ...]", cap: float = 10.0
) -> OPEResult:
    """Clip weights at ``cap``: bounded variance, and a downward bias that grows with the clipping."""
    weights, violations = importance_weights(log, target, prices)
    clipped = [min(weight, cap) for weight in weights]
    value = sum(weight * record["reward"] for weight, record in zip(clipped, log)) / len(log)
    return OPEResult(
        f"capped IPS (c={cap:g})",
        value,
        effective_sample_size(clipped),
        max(clipped, default=0.0),
        violations,
        len(log),
    )


def snips(log: "list[dict]", target: Policy, prices: "tuple[float, ...]") -> OPEResult:
    """Self-normalised IPS: divide by the weight sum instead of by n.

    Slightly biased, much lower variance, and -- unlike IPS -- it cannot return a value above the largest
    logged reward, which is the failure mode that gets a deck laughed out of a review.
    """
    weights, violations = importance_weights(log, target, prices)
    total_weight = sum(weights)
    value = (
        sum(weight * record["reward"] for weight, record in zip(weights, log)) / total_weight
        if total_weight > 0
        else 0.0
    )
    return OPEResult(
        "SNIPS", value, effective_sample_size(weights), max(weights, default=0.0), violations, len(log)
    )


class RewardModel:
    """Ridge regression per price fitted on the log: the reward model DM and DR both need.

    Its weakness is structural rather than a matter of tuning. On the actions the log barely covers, the model
    is extrapolating -- and those are exactly the actions a new policy prefers, because a policy that agreed
    with the log would not be worth evaluating.
    """

    def __init__(self, prices: "tuple[float, ...]", dimension: int, ridge: float = 1.0) -> None:
        self.prices = prices
        self.arms = {price: LinearArm(dimension, ridge) for price in prices}

    def fit(self, log: "list[dict]") -> "RewardModel":
        for record in log:
            self.arms[record["price"]].update(record["context"].features, record["reward"])
        return self

    def predict(self, context: Context, price: float) -> float:
        return dot(self.arms[price].theta, context.features)

    def coverage(self) -> "dict[float, int]":
        return {price: arm.pulls for price, arm in self.arms.items()}


def direct_method(
    log: "list[dict]", target: Policy, prices: "tuple[float, ...]", model: RewardModel | None = None
) -> OPEResult:
    dimension = len(log[0]["context"].features)
    model = model or RewardModel(prices, dimension).fit(log)
    value = 0.0
    for record in log:
        probabilities = target.probabilities(record["context"])
        value += sum(
            probability * model.predict(record["context"], price)
            for probability, price in zip(probabilities, prices)
        )
    return OPEResult("direct method", value / len(log), float(len(log)), 1.0, 0, len(log))


def doubly_robust(
    log: "list[dict]",
    target: Policy,
    prices: "tuple[float, ...]",
    model: RewardModel | None = None,
    cap: float | None = None,
) -> OPEResult:
    """``DM + IPS on the residual``: consistent if either the model or the propensities are right.

        V_DR = mean over records of [ sum_a pi(a|x) q(x,a) + w * (r - q(x,a_logged)) ]

    The importance weight multiplies a *residual* rather than a reward, so when the model is decent the
    weighted term is small and the variance collapses. When the model is useless, DR degrades to IPS rather
    than to nonsense -- which is the entire reason to prefer it.
    """
    dimension = len(log[0]["context"].features)
    model = model or RewardModel(prices, dimension).fit(log)
    weights, violations = importance_weights(log, target, prices)
    if cap is not None:
        weights = [min(weight, cap) for weight in weights]

    value = 0.0
    for weight, record in zip(weights, log):
        context = record["context"]
        probabilities = target.probabilities(context)
        baseline = sum(
            probability * model.predict(context, price)
            for probability, price in zip(probabilities, prices)
        )
        residual = record["reward"] - model.predict(context, record["price"])
        value += baseline + weight * residual
    name = "doubly robust" if cap is None else f"DR (cap {cap:g})"
    return OPEResult(
        name,
        value / len(log),
        effective_sample_size(weights),
        max(weights, default=0.0),
        violations,
        len(log),
    )


def evaluate_all(
    log: "list[dict]", target: Policy, prices: "tuple[float, ...]", cap: float = 10.0
) -> "list[OPEResult]":
    """Every estimator on the same log, which is the only way to read them.

    Agreement between DM and DR is weak evidence that the model is adequate; disagreement between IPS and SNIPS
    means the weights are heavy-tailed and the point estimates should not be trusted at all.
    """
    dimension = len(log[0]["context"].features)
    model = RewardModel(prices, dimension).fit(log)
    return [
        direct_method(log, target, prices, model),
        ips(log, target, prices),
        capped_ips(log, target, prices, cap),
        snips(log, target, prices),
        doubly_robust(log, target, prices, model),
    ]


def weight_percentiles(weights: "list[float]") -> "dict[str, float]":
    """The weight tail, which is where an OPE estimate actually comes from."""
    if not weights:
        return {}
    ordered = sorted(weights)

    def quantile(level: float) -> float:
        position = min(int(level * len(ordered)), len(ordered) - 1)
        return ordered[position]

    return {
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "max": ordered[-1],
        "share above 10": sum(1 for weight in weights if weight > 10.0) / len(weights),
    }


def confidence_interval(values: "list[float]", level: float = 0.95) -> "tuple[float, float]":
    """A normal interval on per-record contributions -- correct only when the ESS is large.

    Included with that caveat attached because the mistake it enables is so common: computing an interval from
    n when the ESS is a fraction of n produces a bound that is far too narrow, and the resulting decision is
    made with unwarranted confidence rather than with none.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    half = 1.959964 * math.sqrt(variance / len(values)) if level == 0.95 else 2.575829 * math.sqrt(
        variance / len(values)
    )
    return (mean - half, mean + half)
