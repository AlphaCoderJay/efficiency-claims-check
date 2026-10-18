# T001: Early gains fade, so judge efficiency at the end of the budget

**By:** DotrixAI. **Evidence level:** CHECKED, in [ECC-001](../results/ECC-001-dotrixai/) (0.73M parameters, 3 seeds, CPU).

## The technique

When comparing the training cost of two recipes, choose the target quality near the end of the baseline's budget, and report the whole cost-to-quality curve. Do not judge a method by an early point.

## Why

Many methods help most early in training. If you stop early, or pick an early target, the method looks far cheaper than it is.

## Evidence

In ECC-001, nGPT needed 0.21 of the baseline's FLOPs to reach the quality the baseline had at update 250, which is 4.9x cheaper. At the baseline's final quality (update 1,200) it needed 0.68, or 1.47x cheaper. A run judged at a quarter of the budget would have overstated the gain more than threefold.

## Limits

This is one check at one small scale. How strongly gains fade depends on the method, the scale and the budget. The rule is still cheap insurance: it costs nothing to report the full curve.
