"""Policies, and the linear algebra they need, written out.

Every policy here is *disjoint linear*: one ridge regression per price, mapping context features to expected
reward. Disjoint means arms share no parameters, so each price must be learned separately -- pessimistic, and
the honest default, since a shared model that assumes reward is linear in price gets the non-monotone revenue
curve wrong in exactly the direction that costs money.

Three exploration strategies:

* **Epsilon-greedy.** Explore uniformly a fixed fraction of the time. Simple, and wasteful: it keeps sampling
  prices it already knows are bad, forever, at a constant rate.
* **LinUCB** (Li et al., 2010). Score = predicted reward + ``alpha * sqrt(x' A^-1 x)``, the ridge prediction's
  standard error. Optimism directs exploration to prices whose value is *uncertain* rather than to random
  ones, and the bonus shrinks as evidence accumulates. ``alpha`` is the confidence width, and the theory's
  value is usually far too conservative in practice -- which is worth knowing before quoting a regret bound.
* **Linear Thompson sampling** (Agrawal & Goyal, 2013). Sample coefficients from the posterior
  ``N(theta_hat, v^2 A^-1)`` and act greedily on the sample. Exploration is proportional to posterior
  uncertainty and, unlike UCB, the policy is *stochastic* -- so its action probabilities exist and can be
  logged, which is what makes it usable as a logging policy for later off-policy evaluation.

Numerics that matter:

* ``A`` is maintained as ``lambda I + sum x x'`` and its inverse updated by the **Sherman-Morrison** rank-1
  formula, so no matrix is ever inverted in the hot loop. A full Gauss-Jordan inverse is kept for tests, and
  the two are asserted to agree -- Sherman-Morrison drifts if applied to a matrix that was not updated
  consistently, and the drift is silent.
* Thompson sampling needs a correlated Gaussian sample, which needs a **Cholesky factor** of ``A^-1``. Ridge
  keeps ``A`` positive definite for ``lambda > 0``, so the factorisation exists; the implementation still
  raises rather than fudging a negative pivot, because a failed Cholesky means the state is corrupt.
"""

from __future__ import annotations

import math
import random

from .environment import Context, PricingEnvironment

# ---------------------------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------------------------


def identity(size: int, scale: float = 1.0) -> "list[list[float]]":
    return [[scale if row == column else 0.0 for column in range(size)] for row in range(size)]


def matrix_vector(matrix: "list[list[float]]", vector: "list[float]") -> "list[float]":
    return [sum(value * vector[index] for index, value in enumerate(row)) for row in matrix]


def dot(left: "list[float]", right: "list[float]") -> float:
    return sum(a * b for a, b in zip(left, right))


def invert(matrix: "list[list[float]]") -> "list[list[float]]":
    """Gauss-Jordan with partial pivoting. Used in tests and setup, never in the hot loop."""
    size = len(matrix)
    augmented = [list(row) + identity(size)[index] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("matrix is singular: ridge lambda should have prevented this")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(augmented[row], augmented[column])
                ]
    return [row[size:] for row in augmented]


def sherman_morrison(inverse: "list[list[float]]", vector: "list[float]") -> "list[list[float]]":
    """Update ``A^-1`` after ``A += x x'`` in O(d^2) instead of O(d^3).

        (A + x x')^-1 = A^-1 - (A^-1 x)(x' A^-1) / (1 + x' A^-1 x)

    The denominator is ``1 + x' A^-1 x``, which is at least 1 for positive definite ``A``, so the update is
    numerically safe here -- unlike the general rank-1 case, where it explodes near ``1 + x' A^-1 x = 0``.
    """
    left = matrix_vector(inverse, vector)
    denominator = 1.0 + dot(vector, left)
    if denominator <= 0.0:
        raise ValueError("Sherman-Morrison denominator is not positive: A is no longer positive definite")
    return [
        [inverse[row][column] - left[row] * left[column] / denominator for column in range(len(vector))]
        for row in range(len(vector))
    ]


def cholesky(matrix: "list[list[float]]") -> "list[list[float]]":
    """Lower-triangular ``L`` with ``L L' = matrix``. Raises on a non-positive pivot."""
    size = len(matrix)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for column in range(row + 1):
            total = matrix[row][column] - sum(
                lower[row][index] * lower[column][index] for index in range(column)
            )
            if row == column:
                if total <= 0.0:
                    raise ValueError(
                        "Cholesky hit a non-positive pivot: the covariance is not positive definite, "
                        "which means the accumulated state is corrupt rather than merely ill-conditioned"
                    )
                lower[row][column] = math.sqrt(total)
            else:
                lower[row][column] = total / lower[column][column]
    return lower


def gaussian_sample(
    mean: "list[float]", covariance: "list[list[float]]", rng: random.Random
) -> "list[float]":
    """Draw from ``N(mean, covariance)`` as ``mean + L z``, which is why the Cholesky factor is needed."""
    lower = cholesky(covariance)
    standard = [rng.gauss(0.0, 1.0) for _ in mean]
    return [
        mean[row] + sum(lower[row][column] * standard[column] for column in range(row + 1))
        for row in range(len(mean))
    ]


# ---------------------------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------------------------


class LinearArm:
    """Ridge regression state for one price: ``A``, ``A^-1``, ``b``, and the coefficients."""

    def __init__(self, dimension: int, ridge: float = 1.0) -> None:
        if ridge <= 0:
            raise ValueError("ridge must be positive: with lambda = 0 the first update is singular")
        self.dimension = dimension
        self.ridge = ridge
        self.A = identity(dimension, ridge)
        self.A_inverse = identity(dimension, 1.0 / ridge)
        self.b = [0.0] * dimension
        self.pulls = 0

    def update(self, features: "list[float]", reward: float) -> None:
        for row in range(self.dimension):
            for column in range(self.dimension):
                self.A[row][column] += features[row] * features[column]
        self.A_inverse = sherman_morrison(self.A_inverse, features)
        self.b = [value + reward * features[index] for index, value in enumerate(self.b)]
        self.pulls += 1

    @property
    def theta(self) -> "list[float]":
        return matrix_vector(self.A_inverse, self.b)

    def predict(self, features: "list[float]") -> float:
        return dot(self.theta, features)

    def width(self, features: "list[float]") -> float:
        """``sqrt(x' A^-1 x)`` -- the ridge prediction's standard error, up to the noise scale."""
        return math.sqrt(max(dot(features, matrix_vector(self.A_inverse, features)), 0.0))


class Policy:
    """Base class: ``select`` acts, ``probabilities`` exposes the action distribution.

    ``probabilities`` is not decoration. A policy that cannot state its action distribution cannot be logged
    for off-policy evaluation, and cannot be evaluated later by anyone else either. Deterministic policies
    return a point mass, which is exactly the case where OPE breaks -- and it is better for that to be visible
    in the interface than discovered afterwards.
    """

    name = "policy"

    def __init__(self, environment: PricingEnvironment, seed: int = 0) -> None:
        self.environment = environment
        self.prices = environment.prices
        self.rng = random.Random(seed)

    def probabilities(self, context: Context) -> "list[float]":
        raise NotImplementedError

    def select(self, context: Context) -> float:
        weights = self.probabilities(context)
        return self.rng.choices(self.prices, weights=weights)[0]

    def update(self, context: Context, price: float, reward: float) -> None:
        pass

    @property
    def deterministic(self) -> bool:
        return False


class UniformPolicy(Policy):
    """Pure exploration. The reference for how much regret exploration costs."""

    name = "uniform"

    def probabilities(self, context: Context) -> "list[float]":
        return [1.0 / len(self.prices)] * len(self.prices)


class FixedPricePolicy(Policy):
    """One price for everyone -- the incumbent that any pricing project is measured against.

    Beating a good flat price is harder than it sounds, and a bandit that cannot is not worth its operational
    cost. Including it prevents the comparison from being run only against straw men.
    """

    def __init__(self, environment: PricingEnvironment, price: float, seed: int = 0) -> None:
        super().__init__(environment, seed)
        if price not in self.prices:
            raise ValueError("fixed price must be in the action set")
        self.price = price
        self.name = f"fixed@{price:g}"

    def probabilities(self, context: Context) -> "list[float]":
        return [1.0 if price == self.price else 0.0 for price in self.prices]

    @property
    def deterministic(self) -> bool:
        return True


class LinearPolicy(Policy):
    """Shared machinery for the three learning policies: one ``LinearArm`` per price."""

    def __init__(self, environment: PricingEnvironment, ridge: float = 1.0, seed: int = 0) -> None:
        super().__init__(environment, seed)
        dimension = len(Context("regular", False, 0.0).features)
        self.arms = {price: LinearArm(dimension, ridge) for price in self.prices}

    def update(self, context: Context, price: float, reward: float) -> None:
        self.arms[price].update(context.features, reward)

    def estimates(self, context: Context) -> "dict[float, float]":
        features = context.features
        return {price: arm.predict(features) for price, arm in self.arms.items()}


class EpsilonGreedy(LinearPolicy):
    """Greedy on the ridge estimate, uniform with probability ``epsilon``.

    The waste is structural: exploration never stops and never concentrates. After ten thousand rounds it is
    still sampling the price it has known to be terrible since round fifty, at the same rate.
    """

    def __init__(
        self, environment: PricingEnvironment, epsilon: float = 0.1, ridge: float = 1.0, seed: int = 0
    ) -> None:
        super().__init__(environment, ridge, seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        self.epsilon = epsilon
        self.name = f"eps-greedy({epsilon:g})"

    def probabilities(self, context: Context) -> "list[float]":
        estimates = self.estimates(context)
        best = max(estimates, key=lambda price: (estimates[price], price))
        share = self.epsilon / len(self.prices)
        return [share + (1.0 - self.epsilon if price == best else 0.0) for price in self.prices]


class LinUCB(LinearPolicy):
    """Optimism in the face of uncertainty: ``prediction + alpha * width``.

    Deterministic given its state, so ``probabilities`` is a point mass -- and therefore LinUCB is a poor
    logging policy however good it is at earning revenue. Choosing what to deploy and what to log from are two
    different decisions, and conflating them is why so many teams have logs they cannot evaluate against.
    """

    def __init__(
        self, environment: PricingEnvironment, alpha: float = 1.0, ridge: float = 1.0, seed: int = 0
    ) -> None:
        super().__init__(environment, ridge, seed)
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        self.alpha = alpha
        self.name = f"linucb({alpha:g})"

    def scores(self, context: Context) -> "dict[float, float]":
        features = context.features
        return {
            price: arm.predict(features) + self.alpha * arm.width(features)
            for price, arm in self.arms.items()
        }

    def probabilities(self, context: Context) -> "list[float]":
        scores = self.scores(context)
        best = max(scores, key=lambda price: (scores[price], price))
        return [1.0 if price == best else 0.0 for price in self.prices]

    @property
    def deterministic(self) -> bool:
        return True


class LinearThompson(LinearPolicy):
    """Posterior sampling: draw ``theta ~ N(theta_hat, v^2 A^-1)`` per arm, then act greedily.

    Stochastic, so its action probabilities exist -- though they have no closed form, which is the practical
    catch. ``probabilities`` estimates them by Monte Carlo, and the cost of that estimate is the honest price
    of using Thompson sampling as a logging policy: propensities are approximate, and an approximate
    propensity in the denominator of an importance weight is a bias nobody reports.
    """

    def __init__(
        self,
        environment: PricingEnvironment,
        variance: float = 1.0,
        ridge: float = 1.0,
        samples: int = 200,
        seed: int = 0,
    ) -> None:
        super().__init__(environment, ridge, seed)
        if variance <= 0:
            raise ValueError("variance must be positive")
        self.variance = variance
        self.samples = samples
        self.name = f"lin-ts({variance:g})"

    def _sample_scores(self, context: Context) -> "dict[float, float]":
        features = context.features
        scores = {}
        for price, arm in self.arms.items():
            covariance = [
                [value * self.variance**2 for value in row] for row in arm.A_inverse
            ]
            theta = gaussian_sample(arm.theta, covariance, self.rng)
            scores[price] = dot(theta, features)
        return scores

    def select(self, context: Context) -> float:
        scores = self._sample_scores(context)
        return max(scores, key=lambda price: (scores[price], price))

    def probabilities(self, context: Context) -> "list[float]":
        """Monte Carlo estimate of the action distribution, floored away from zero.

        The floor is not cosmetic: a propensity of exactly zero from a finite sample would make an importance
        weight infinite for an action the policy can in fact take. Flooring biases the estimate slightly and
        keeps it finite, which is the correct trade -- but it is a choice, and it is made here explicitly
        rather than by accident.
        """
        counts = dict.fromkeys(self.prices, 0)
        for _ in range(self.samples):
            scores = self._sample_scores(context)
            counts[max(scores, key=lambda price: (scores[price], price))] += 1
        floor = 1.0 / (2.0 * self.samples)
        raw = [max(counts[price] / self.samples, floor) for price in self.prices]
        total = sum(raw)
        return [value / total for value in raw]


class OraclePolicy(Policy):
    """Knows the true demand model. The upper bound, and not achievable -- that is the point of showing it."""

    name = "oracle"

    def probabilities(self, context: Context) -> "list[float]":
        best = self.environment.optimal_price(context)
        return [1.0 if price == best else 0.0 for price in self.prices]

    @property
    def deterministic(self) -> bool:
        return True
