# Version 1 results (superseded)

The first model, evaluated once on the test set on 2026-10-18. Its `payment_ratio_1`
feature divided the September payment by the September bill; the payment actually
settles the previous (August) statement. After that bug was found and fixed, the model
was retrained and the test set was used a second time (see `../test_metrics.json`).
The fix was adopted regardless of its effect on any score.

Version 1 test results: PR-AUC 0.556, ROC-AUC 0.776, recall 74.7% at threshold 0.16,
cost per client 0.564 (no model: 1.106).

Note: `train_summary.json` here is from a retrain at 14:56 UTC that produced identical
predictions to the evaluated model trained at 14:49 UTC (same platform, deterministic);
model files were not hashed in version 1.
