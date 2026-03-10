# Iterative Full-Fit Training Mode

Date: 2026-03-10

## What changed

The iterative classification pipeline now trains on all cumulatively labeled samples in each loop.

Previous behavior:
- split labeled data into train/test with `train_test_split(..., test_size=0.2)`
- split the train portion again into a validation set with `test_size=0.1`
- disable early stopping with `early_stopping_patience=999999`
- still report train/test metrics and save separate train/test parquet files

Current behavior:
- use all cumulatively labeled samples as `X_train` / `y_train`
- do not create validation or holdout test splits in iterative mode
- keep fixed-epoch training (`max_epochs=150`)
- evaluate only on the labeled set as in-sample diagnostics
- save one labeled-results parquet file instead of separate train/test result files
- skip ROC/PR plotting in iterative mode because there is no holdout set

## Why

In the iterative GRASP workflow, the earliest loops may contain labeled samples that are orders of magnitude fewer than the total search space. In that regime:
- carving out validation or test subsets removes scarce signal from training
- random holdout loss is not a reliable proxy for the real objective: discovering important CSFs in later loops
- early stopping based on same-loop validation data can bias the model toward reproducing the current calculated configurations instead of ranking unseen important ones

## Scope of the code changes

Updated files:
- `src/graspkit/ml_module/ml_trainer.py`
- `src/graspkit/ml_module/ml_results_analyzer.py`
- `src/graspkit/ml_module/ml_types.py`
- `tests/test_iterative_training_mode.py`

Main API changes:

```python
model, X_labeled, y_labeled = train_model(...)
evaluation_results, prediction_outputs = evaluate_model(
    model,
    X_labeled,
    y_labeled,
    config,
    logger,
)
```

Result schema changes:
- `test_metrics` and `train_metrics` are removed from iterative mode outputs
- `labeled_metrics` is the primary metric block
- `y_prediction_labeled` and `y_probability_labeled` replace train/test-specific fields
- `metadata.metric_scope` is set to `"in_sample"`

Saved artifacts:
- labeled diagnostics parquet: `*_labeled_results.parquet`
- training CSV columns now record `labeled_*` metrics

## Interpretation guidance

`labeled_metrics` should be treated as diagnostics only.

They answer:
- has the model fit the currently known labeled data?

They do not answer:
- will the next loop discover important CSFs efficiently?
- is the model generalizing well across loops?

For model quality in production iterative runs, prefer workflow-level metrics such as:
- top-k hit rate on newly discovered important CSFs
- recall of newly verified important CSFs in the next loop
- reduction in required calculations relative to non-ML selection

## Compatibility note

Tooling or scripts that previously expected:
- `X_train, X_test, y_train, y_test` from `train_model`
- `test_metrics` / `train_metrics` in `evaluation_results`
- `*_test_results.parquet` and `*_train_results.parquet`

must be updated to the labeled-only iterative interface.
