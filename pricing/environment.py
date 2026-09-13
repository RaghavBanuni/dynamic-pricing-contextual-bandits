"""A pricing environment whose optimum is known, so regret can be measured rather than described.

Customers arrive with a context (segment, whether it is a weekend, a normalised basket size). For a price ``p``
the purchase probability is logistic in price, with an intercept and a slope that depend on the context:

    P(buy | x, p) = sigmoid(a(x) - b(x) * p)      b(x) > 0
    reward         = p  if bought else 0          (revenue, not profit -- see below)

Two properties make this the right toy problem for bandits rather than for regression:

**The reward is non-monotone in the action.** Revenue is ``p * P(buy | x, p)``: raise the price and margin per
sale rises while conversion falls. The optimum is interior, and it moves with the context -- a
price-insensitive segment should be charged more. A policy that learns "higher price = more revenue" from a
narrow price range will walk straight off the cliff, which is the characteristic failure of pricing models
fitted to observational data.

**Only the chosen price is observed.** A customer shown 12.00 who declined tells you nothing directly about
9.00. That is the bandit feedback structure, and it is why off-policy evaluation is hard.

``optimal_price`` and ``expected_revenue`` are computed by grid search over the true model, so every regret
number in this repository is measured against a known optimum instead of against another algorithm.

Revenue rather than profit is deliberate: with a unit cost the optimum shifts and the objective becomes
``(p - c) * P(buy)``. ``unit_cost`` is supported for exactly that reason -- pricing on revenue while paying a
per-unit cost maximises the wrong thing, and it is the most common framing error in a pricing project.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

SEGMENTS = ("bargain", "regular", "premium")


def sigmoid(value: float) -> float:
    """Numerically safe logistic. The naive form overflows for value < -700."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


@dataclass(frozen=True)
class Context:
    """One arriving customer. ``features`` is what a policy sees; ``segment`` is latent truth."""

    segment: str
    weekend: bool
    basket: float

    @property
    def features(self) -> "list[float]":
        """One-hot segment, weekend flag, basket size, and an intercept.

        The segment is included as an observable feature here, which is generous: in production the segment is
        inferred and noisy, and a bandit that appears to work on clean segment labels can fail entirely on
        estimated ones.
        """
        one_hot = [1.0 if self.segment == name else 0.0 for name in SEGMENTS]
        return [*one_hot, 1.0 if self.weekend else 0.0, self.basket, 1.0]


class PricingEnvironment:
    """Logistic demand with context-dependent price sensitivity and a known optimum.

    Parameters
    ----------
    prices:
        The discrete action set. Discrete because that is what pricing systems actually deploy -- price
        ladders, not continuous optimisation -- and because it makes the regret decomposition exact.
    unit_cost:
        Per-sale cost. Zero means the objective is revenue; anything else makes it margin.
    noise:
        Standard deviation of a mean-zero shock added to the demand intercept each round, representing
        everything the features do not capture. It is what stops any policy from reaching zero regret.
    """

    def __init__(
        self,
        prices: "tuple[float, ...]" = (6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0),
        unit_cost: float = 0.0,
        noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not prices or any(price <= 0 for price in prices):
            raise ValueError("prices must be positive")
        if unit_cost < 0:
            raise ValueError("unit_cost cannot be negative")
        self.prices = tuple(sorted(prices))
        self.unit_cost = unit_cost
        self.noise = noise
        self.rng = random.Random(seed)

        # true demand parameters per segment: (intercept, price slope)
        self.demand = {
            "bargain": (2.6, 0.34),   # very price sensitive: optimum near the bottom of the ladder
            "regular": (3.4, 0.26),
            "premium": (4.6, 0.17),   # tolerant: optimum near the top
        }

    # -- truth --------------------------------------------------------------------------------

    def purchase_probability(self, context: Context, price: float) -> float:
        intercept, slope = self.demand[context.segment]
        # weekends bring slightly more willingness to buy; a large basket makes an extra item easier to add
        adjusted = intercept + (0.25 if context.weekend else 0.0) + 0.4 * context.basket
        return sigmoid(adjusted - slope * price)

    def expected_reward(self, context: Context, price: float) -> float:
        """Expected margin (or revenue when ``unit_cost`` is zero) for one impression."""
        return (price - self.unit_cost) * self.purchase_probability(context, price)

    def optimal_price(self, context: Context) -> float:
        return max(self.prices, key=lambda price: self.expected_reward(context, price))

    def optimal_reward(self, context: Context) -> float:
        return max(self.expected_reward(context, price) for price in self.prices)

    def regret(self, context: Context, price: float) -> float:
        """Expected regret of one decision -- computed on expectations, never on realised rewards.

        Realised regret is dominated by purchase noise: a policy can pick the right price and see no sale.
        Averaging expected regret is what makes 2,000-round comparisons readable at all.
        """
        return self.optimal_reward(context) - self.expected_reward(context, price)

    # -- sampling -----------------------------------------------------------------------------

    def sample_context(self) -> Context:
        return Context(
            segment=self.rng.choices(SEGMENTS, weights=[0.45, 0.4, 0.15])[0],
            weekend=self.rng.random() < 2.0 / 7.0,
            basket=round(self.rng.betavariate(2.0, 5.0), 3),
        )

    def pull(self, context: Context, price: float) -> "tuple[float, bool]":
        """Realise one interaction: ``(reward, bought)``.

        The noise shock is applied to the demand intercept, not to the reward, because demand shocks are what
        actually happen -- a competitor promotion moves conversion, not the money received per sale.
        """
        if price not in self.prices:
            raise ValueError(f"price {price} is not in the action set")
        probability = self.purchase_probability(context, price)
        if self.noise:
            intercept, slope = self.demand[context.segment]
            shock = self.rng.gauss(0.0, self.noise)
            probability = sigmoid(
                intercept + shock + (0.25 if context.weekend else 0.0) + 0.4 * context.basket - slope * price
            )
        bought = self.rng.random() < probability
        return ((price - self.unit_cost) if bought else 0.0), bought

    # -- reporting ----------------------------------------------------------------------------

    def optimum_table(self) -> "list[tuple[str, float, float, float]]":
        """Best price per segment at an average basket -- the answer a policy has to find."""
        rows = []
        for segment in SEGMENTS:
            context = Context(segment=segment, weekend=False, basket=0.3)
            best = self.optimal_price(context)
            rows.append(
                (
                    segment,
                    best,
                    self.purchase_probability(context, best),
                    self.expected_reward(context, best),
                )
            )
        return rows

    def revenue_curve(self, context: Context) -> "list[tuple[float, float, float]]":
        """``(price, purchase probability, expected reward)`` across the ladder.

        Printing this is the fastest way to see that the objective is not monotone, and therefore that
        "test a higher price" is not a strategy.
        """
        return [
            (price, self.purchase_probability(context, price), self.expected_reward(context, price))
            for price in self.prices
        ]


def uniform_logging_policy(environment: PricingEnvironment, rounds: int, seed: int = 0):
    """Explore-only logging: uniform over the ladder, with the propensity recorded.

    This exists because off-policy evaluation is impossible without logged propensities, and it is the cheapest
    honest way to obtain them. It is also expensive in revenue -- which is the real reason production logs are
    usually unusable for OPE: nobody wants to pay for uniform exploration, so the logging policy is greedy, and
    a greedy policy has zero probability on most actions.
    """
    rng = random.Random(seed)
    propensity = 1.0 / len(environment.prices)
    log = []
    for _ in range(rounds):
        context = environment.sample_context()
        price = rng.choice(environment.prices)
        reward, bought = environment.pull(context, price)
        log.append(
            {
                "context": context,
                "price": price,
                "reward": reward,
                "bought": bought,
                "propensity": propensity,
            }
        )
    return log
