# Dynamic Pricing With Contextual Bandits

LinUCB, linear Thompson sampling and epsilon-greedy written from scratch (ridge per arm, Sherman-Morrison
updates, Cholesky posterior sampling), measured as **regret against a known optimum**, plus off-policy
evaluation — IPS, capped, self-normalised, direct method, doubly robust — with the diagnostics that decide
whether any of those estimates mean anything. **Pure Python, standard library only.**

## Why pricing is a bandit problem and not a regression

Revenue is `p * P(buy | x, p)`. Raise the price, margin per sale rises and conversion falls, so the optimum is
**interior** — and it moves with the context:

```
segment     best price   P(buy)   expected revenue
bargain           8.00    56.4%              4.51
regular          12.00    52.0%              6.24
premium          18.00    73.3%             13.19
```

Two consequences, both expensive:

1. **A model fitted to observational data learns the wrong slope.** If historical prices only ever sat between
   8 and 10, the fitted relationship says "higher price, more revenue" — because over that range it is true.
   Extrapolate and you walk off the cliff on the right-hand side of the curve. `python -m pricing.cli demand`
   prints the whole ladder so the shape is visible.
2. **Only the chosen price is observed.** A customer shown 12.00 who declined tells you nothing directly about
   9.00. That is bandit feedback, and it is why offline evaluation is the hard part of this project rather than
   an afterthought.

The environment knows the true demand model, so every number below is measured against the actual optimum
rather than against another algorithm.

```bash
python -m pricing.cli demand    # the revenue curve
python -m pricing.cli regret    # 3,000 rounds, regret vs a known optimum
python -m pricing.cli ope       # four estimators against the true policy value
python -m pricing.cli support   # what a greedy logging policy does to OPE
python -m pricing.cli margin    # pricing on revenue while paying a unit cost
```

## Regret

```
policy              total regret   regret @500   revenue vs oracle
uniform                   3891.2         648.4               71.4%
fixed@10                  1204.7         201.1               88.9%
fixed@15                  1876.3         312.7               84.1%
eps-greedy(0.1)            622.4         173.9               94.6%
linucb(0.6)                311.8         142.2               97.2%
lin-ts(0.4)                288.5         131.6               97.4%
oracle                       0.0           0.0              100.0%
```

> Illustrative, seed-dependent. The tests assert the orderings that matter: the learners beat uniform
> exploration, and LinUCB's regret in the second half of a run is smaller than in the first.

The **flat prices are the incumbent**, and that comparison is the one to take seriously. A good flat price
captures most of the available value; a bandit has to beat it by enough to justify the exploration cost, the
propensity logging, the monitoring and the on-call burden. Comparing a bandit only against uniform exploration
is a straw man.

`alpha` in LinUCB is the confidence width. The theoretically justified value is usually far too conservative to
be useful, which is worth knowing before quoting a regret bound at anyone.

## Off-policy evaluation, and the two ways it fails

You have logs from the policy that was running and you want the value of one that was not.

| estimator | bias | variance | fails when |
|---|---|---|---|
| direct method | model bias | ~zero | the model extrapolates onto rarely-logged actions |
| IPS | unbiased with full support | unbounded | propensities are small |
| capped IPS | downward | bounded | the cap does the work, not the data |
| SNIPS | slight | much lower | still needs support |
| doubly robust | consistent if *either* model or propensities are right | between DM and IPS | both are wrong |

The direct method's weakness is structural, not a tuning problem: it extrapolates exactly on the actions the
log barely covers, and those are the actions a new policy prefers — a policy that agreed with the log would not
be worth evaluating.

**Failure one: no support.** A greedy logging policy shows one price. Evaluating any other policy against that
log gives an importance weight of zero on every record:

```
target policy: flat 15.00, true value 6.83 per impression
greedy log:  IPS 0.000   ESS 0.0 (0.0%)   violations 4000
Thompson log: IPS 6.71   ESS 812.4 (20.3%)   violations 0
```

IPS returns `0.000` — an estimate of nothing, presented as a number. No volume of additional data repairs it:
the log contains no information about a price that was never shown. **The logging policy must be stochastic,
and that decision has to be made before the logs are collected**, which is why `probabilities()` is part of the
`Policy` interface here and why LinUCB — deterministic, and excellent at earning revenue — makes a poor logging
policy. Deploying and logging are two different decisions.

**Failure two: weak support.** With propensities that are small but non-zero, IPS is unbiased and useless:

```
ESS = (sum w)^2 / sum w^2
```

Ten thousand records with an ESS of 40 hold about as much information about the new policy as forty records.
Every estimator here returns its ESS, maximum weight and support-violation count alongside the point estimate,
because the estimate alone is not interpretable — and `confidence_interval` carries an explicit warning, since
an interval computed from `n` while the ESS is a fraction of `n` produces unwarranted confidence rather than
none.

Reading them together: DM agreeing with DR is weak evidence the model is adequate; IPS disagreeing with SNIPS
means the weights are heavy-tailed and no point estimate should be trusted.

## Implementation notes

- `A^-1` is maintained by **Sherman-Morrison** rank-1 updates, never re-inverted in the loop. The tests compare
  it against a full Gauss-Jordan inverse after twenty updates, because the rank-1 update drifts *silently* —
  and if it drifts, every confidence width is wrong while everything still runs.
- Thompson sampling needs a correlated Gaussian draw, so a **Cholesky** factor of `A^-1`. It raises on a
  non-positive pivot rather than clamping, because a failed factorisation means the accumulated state is
  corrupt, not merely awkward.
- Arms are **disjoint**: no parameter sharing across prices. A shared model linear in price would get the
  non-monotone revenue curve wrong in the direction that costs money.
- Thompson propensities have no closed form; they are estimated by Monte Carlo and floored above zero, which
  biases them slightly and keeps importance weights finite. A deliberate trade, made explicitly.

## Revenue is probably the wrong objective

`python -m pricing.cli margin` prices the same demand with a unit cost. The margin-optimal price is strictly
higher than the revenue-optimal one, because every marginal sale carries a cost the revenue objective never
counted. The bandit machinery is entirely indifferent: it optimises whatever reward you hand it, including the
wrong one, and it does so efficiently.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Matrix identities, Sherman-Morrison against a full inverse, Cholesky reconstruction and its refusal on a
non-PD matrix, ridge recovering a known linear reward, action distributions summing to one for every policy,
the oracle having exactly zero regret, learners beating uniform exploration, importance weights averaging to
one, ESS collapsing under a dominant weight, IPS matching the truth on a uniform log, and IPS returning exactly
zero on a greedy log while the true value is not zero.

## Limits

- **Stationary demand.** No trend, seasonality or competitor response. Real pricing is non-stationary, and a
  bandit that has converged is precisely the one that will not notice a regime change — sliding windows or
  discounted statistics are the first thing to add.
- **No strategic customers.** Real buyers wait for sales, share prices, and punish perceived unfairness. Price
  discrimination across observable segments also carries legal and reputational constraints this model does not
  represent.
- **Linear reward model.** Reward is assumed linear in features per arm. The true model here is logistic, so
  the model is *misspecified on purpose* — which is realistic, and one reason the learners never reach zero
  regret.
- **Segments are given, not inferred.** Production features are estimated and noisy; a bandit that works on
  clean segment labels can fail on predicted ones.
- **No delayed or partial feedback.** Purchases are immediate. Returns, chargebacks and subscription churn
  arrive weeks later, and a bandit that treats a sale as final over-prices systematically.
- **Discrete price ladder.** Continuous pricing needs a different treatment; the ladder is what pricing systems
  deploy anyway, and it makes the regret decomposition exact.

## References

- Li, Chu, Langford & Schapire (2010), *A contextual-bandit approach to personalized news article recommendation* — LinUCB.
- Agrawal & Goyal (2013), *Thompson sampling for contextual bandits with linear payoffs*.
- Chapelle & Li (2011), *An empirical evaluation of Thompson sampling*.
- Dudík, Langford & Li (2011), *Doubly robust policy evaluation and learning*.
- Swaminathan & Joachims (2015), *The self-normalized estimator for counterfactual learning*.
- Bottou et al. (2013), *Counterfactual reasoning and learning systems* — propensity logging in production.
- Lattimore & Szepesvári (2020), *Bandit Algorithms* — the regret analysis in full.

MIT licensed.
