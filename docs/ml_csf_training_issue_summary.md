# ML CSF Training Issue Summary

Last updated: 2026-03-11

## Background

Current workflow:

- CSF CI coefficients come from the GRASP atomic-structure program, mainly the RCI stage.
- The ML pipeline uses descriptors plus labels derived from CI coefficients to predict which CSFs should be included in later calculation rounds.
- In the current runs, the presampled configuration space is less than 1% of the full CSF space.

Observed training log symptoms:

- Round 4 cumulative training samples: `244437`
- Positive labels: `226315`
- Average positive ratio: `0.4629`
- Diagnostics:
  - `AUC: 0.5114`
  - `F1: 0.6329`
  - `Accuracy: 0.3844`
  - `Precision: 0.4629`
  - `Recall: 1.0000`
- Final epochs report zero gradients for several parameters.

## Main Finding

The current failure mode is not simply "too few samples".

The more direct issue is:

- the labeled training set is built only from the already sampled subspace;
- the positive-label definition is broad enough that positives occupy nearly half of the labeled set;
- the classifier therefore collapses toward almost-all-positive prediction;
- once this happens, it becomes difficult to discover genuinely new important CSFs.

## Important Clarification About `cutoff_value`

`cutoff_value` is applied to the **squared CI coefficient**:

- label rule: `CI^2 >= cutoff_value`
- not: `|CI| >= cutoff_value`

For example:

- if `cutoff_value = 1e-9`, then the equivalent amplitude threshold is
  `|CI| >= sqrt(1e-9) ~= 3.16e-5`

So this threshold must be interpreted in the `CI^2` scale, not the raw `CI` scale.

## Why The Current Training Degenerates

### 1. Training labels come only from the accumulated sampled CSFs

Current code path:

- `/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/ml_moduleml_initializer.py`
- `/Users/yiqin/Documents/PythonProjects/graspkit`
- function: `generate_train_csfs_descriptors`

Current behavior:

- load only accumulated historical sampled CSFs;
- mark labels from `accumulated_ci_squared >= cutoff_value`;
- train only on this accumulated sampled subset.

Consequence:

- the model does not see enough background negatives from the unsampled full CSF pool;
- the labeled set no longer reflects the real imbalance of the full search space.

### 2. Positive ratio is too high for a rare-CSF discovery task

Current log already warns:

- positive ratio `0.4629`

And the metric pattern strongly suggests near-all-positive prediction:

- precision equals positive ratio (`0.4629`);
- recall is `1.0000`.

This is consistent with a classifier that predicts almost everything as positive.

### 3. Fixed 0.5 decision threshold is unsuitable here

Current code path:

- `graspkit\ml_module\ml_trainer.py`
- function: `predict_model`

Current behavior:

- compute probabilities on unsampled CSFs;
- use `max(probability across levels) > 0.5` as the importance decision rule.

Consequence:

- once calibration drifts, this threshold becomes meaningless;
- if the model is over-positive, almost everything is predicted important;
- if the model becomes over-conservative, almost nothing is selected.

For this task, ranking and top-N selection are more robust than a fixed threshold.

### 4. Binary labels discard too much information from the CI data

Physical target:

- identify CSFs that make important contributions to target levels and energy separations.

Current classifier target:

- only learn whether `CI^2` crosses a hard threshold.

Consequence:

- a lot of ordering information in the real `CI^2` values is lost;
- weak, medium, and strong contributors get compressed into coarse labels.

This makes learning especially fragile when the sampled space is tiny.

## Small-Positive-Sample Concern

There is a valid concern that if `cutoff_value` is increased too much:

- positive samples may become extremely rare;
- a pure binary classifier may then fail because the positive class is too small.

This concern is real.

However, the conclusion is **not** that the threshold should stay very loose.

Instead:

- the task should be treated as a **ranking / retrieval** problem, not a standard balanced classification problem;
- a small number of reliable positive samples can still be useful if the model is optimized to rank promising CSFs near the top;
- this is why regression or score-based selection is likely a better fit than fixed-threshold binary classification.

## Files Reviewed

- `ml_CSFs_selection_scripts/ml_csf_choosing/train.py`
- `ml_CSFs_selection_scripts/ml_csf_choosing/train_regression.py`
- `/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/ml_moduleml_initializer.py`
- `/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/ml_moduleml_trainer.py`
- `/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/ml_moduleneural_network.py`

## Recommended Direction

### Option A: Keep classification, but fix the data and selection strategy

Recommended changes:

1. Add background negative samples from the unsampled CSF pool.
2. Optionally add hard negatives:
   - CSFs predicted as important in previous rounds but later shown by RCI to have low `CI^2`.
3. Tighten positive-label definition carefully in `CI^2` space:
   - use quantiles, or
   - use per-level top-k / top-q%, or
   - increase `cutoff_value` moderately only after distribution analysis.
4. Replace fixed `0.5` inference threshold with score ranking:
   - directly select top-N unsampled CSFs each round.

Expected benefit:

- keeps current pipeline structure;
- should reduce all-positive collapse;
- improves exploration of new CSFs.

### Option B: Move to regression on CI magnitude

Recommended target:

- predict `log10(CI^2 + eps)` or another monotonic transformed CI importance score.

Why this is likely better:

- preserves more physical information than hard labels;
- works better when truly important CSFs are rare;
- naturally supports rank-based top-N sampling.

Relevant existing file:

- `ml_CSFs_selection_scripts/ml_csf_choosing/train_regression.py`

This path already exists and is likely the better long-term direction.

## Concrete Next Steps For Code Optimization

Suggested implementation order:

1. Add diagnostic logging for `CI^2` distribution on target levels:
   - min / max / mean
   - quantiles such as 50%, 90%, 95%, 99%, 99.9%
   - positive ratio under several candidate `cutoff_value` values
2. Change inference from fixed `0.5` to top-N score selection.
3. Inject unsampled background negatives into classifier training.
4. Compare classifier vs regression on the same historical rounds.
5. If regression clearly ranks useful CSFs better, switch the iterative workflow to regression-first.

## Suggested Validation Metrics

Do not rely only on in-sample classification metrics such as:

- AUC
- F1
- Accuracy

For this task, more meaningful checks are:

- among top-N ML-selected CSFs in round `k+1`, how many later exceed the target `CI^2` threshold;
- how much the selected CSFs improve target-level energy spacing;
- retention of truly important CSFs across rounds;
- enrichment over random sampling.

## Resume Notes For Next Session

If continuing this optimization later, start with:

1. inspect `generate_train_csfs_descriptors` in `/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/ml_moduleml_initializer.py`;
2. inspect `predict_model` in `/Users/yiqin/Documents/PythonProjects/graspkit/src/graspkit/ml_moduleml_trainer.py`;
3. add `CI^2` distribution diagnostics before changing `cutoff_value`;
4. implement top-N sampling before tuning the neural network itself;
5. evaluate whether `train_regression.py` should become the primary path.
