# Feasibility of Using Regression for CSF Screening Inference

## Assessment

Using a plain regression model to infer CSF importance from your CI² distribution is feasible in a narrow sense, but it is not the best-aligned primary objective for screening.

The main reason is objective mismatch. Your screening task is effectively a head-retrieval problem, not a uniform value-estimation problem:

- CI² spans roughly 15 to 20 orders of magnitude.
- The physically relevant mass is extremely concentrated in a tiny head.
- Missing a few top CSFs hurts much more than making large errors on the long tail.

A standard regression loss treats all samples more evenly. That means the model spends too much capacity fitting the vast tail of near-zero CSFs, even though those are mostly irrelevant for screening.

## What The Current Code Does

The repository already reflects this tradeoff:

- The classifier path converts CI² into binary importance labels using `CI² >= cutoff_value` in `src/graspkit/ml_module/ml_initializer.py` and trains a multi-label classifier in `src/graspkit/ml_module/ml_trainer.py`.
- The regression path predicts `log10(CI²)` directly in `src/graspkit/ml_module/ml_regression_trainer.py` using a Huber-loss MLP in `src/graspkit/ml_module/ml_regression_model.py`.
- At inference, regression ranks CSFs by `max(predicted log10(CI²))` across levels in `src/graspkit/ml_module/ml_regression_trainer.py`.

That ranking rule is directionally reasonable for screening, but the training loss is still value-regression, not head-focused ranking.

## Why Plain Regression Is Risky Here

### 1. Heavy-tail compression is still not enough

Even after `log10`, the target range is still huge. The current code also clips at `1e-15`, which collapses everything below that into one floor value; your stated tail extends to about `1e-20`, so the bottom 5 orders are discarded in training.

### 2. Absolute error is not your real metric

For screening, what matters is:

- recall of top-k / top-mass CSFs
- how much cumulative CI² mass is recovered
- whether the model preserves ordering near the head

MAE/RMSE on all CSFs can look fine while top-CSF retrieval is poor.

### 3. The head is too important

If top 5 already carry about 50% of the mass, then small ranking mistakes among those few CSFs dominate downstream quality. Regression on the whole distribution does not naturally prioritize that region.

### 4. Multi-output averaging can dilute rare signals

The regression model predicts all levels jointly. If a CSF is critical for one level but negligible for others, shared hidden layers plus average loss can weaken that rare but important pattern.

## Feasibility Conclusion

Regression is feasible as an auxiliary ranking signal, but I would not use plain regression as the sole screening model unless you evaluate it with retrieval-style metrics and redesign the loss toward the head.

In practical terms:

- Good use: reranking candidates after a coarse filter.
- Weak use: direct end-to-end replacement for importance classification.
- Best use: two-stage pipeline.

## Recommended Modeling Strategy

I would recommend one of these, in order:

### 1. Two-stage model

- Stage 1: classify whether a CSF is potentially important.
- Stage 2: regress or rank only within that candidate set.

### 2. Ranking/ordinal model instead of plain regression

- Predict quantile/bin of `log10(CI²)`, or
- use pairwise/listwise ranking loss focused on top-ranked CSFs.

### 3. Redefine labels around screening utility

Instead of a fixed CI² threshold, define positives by:

- top-k per level, or
- minimum set covering 90% / 95% cumulative CI² mass.

That aligns the model with your actual screening objective better than raw CI² regression.

## What To Measure

If you test regression, do not judge it by MAE/RMSE first. Use:

- Recall@k for top CSFs per level
- cumulative CI² mass captured in top-k predictions
- rank correlation restricted to head region
- false-negative rate among top 5 / top 30 / top 100
- downstream effect on retained energy accuracy after screening

If those are good, regression is viable. If only global MAE is good, it probably is not.

## Bottom Line

For your CI² distribution, the task is fundamentally extreme-imbalance head retrieval. Plain regression is possible, but not naturally suited as the main inference model. A classifier or ranking-first approach is better aligned, and regression is most useful as a second-stage scorer.

If needed, the next practical step would be a code-level design review and a concrete replacement objective for this repository, such as:

1. top-mass label generation
2. two-stage classifier + regressor
3. pairwise ranking loss on `log10(CI²)`
