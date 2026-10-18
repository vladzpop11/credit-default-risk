# Cross-validation results (training split, stratified 5-fold)

Mean ± standard deviation over the 5 folds. F1, recall and precision use the default 0.5 threshold; the decision threshold is chosen separately, from the cost of errors.

| Model | PR-AUC | ROC-AUC | F1@0.5 | Recall@0.5 | Precision@0.5 | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (prior) | 0.221 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1 |
| logistic regression | 0.515 ± 0.012 | 0.764 ± 0.007 | 0.419 ± 0.020 | 0.310 ± 0.019 | 0.646 ± 0.021 | 1 |
| random forest | 0.555 ± 0.011 | 0.782 ± 0.007 | 0.478 ± 0.017 | 0.370 ± 0.017 | 0.674 ± 0.012 | 34 |
| xgboost | 0.560 ± 0.012 | 0.784 ± 0.005 | 0.482 ± 0.015 | 0.376 ± 0.017 | 0.674 ± 0.012 | 6 |
| logistic regression, without SEX/MARRIAGE/AGE | 0.513 ± 0.012 | 0.763 ± 0.008 | 0.417 ± 0.020 | 0.308 ± 0.018 | 0.648 ± 0.024 | 1 |
| xgboost, without SEX/MARRIAGE/AGE | 0.560 ± 0.012 | 0.783 ± 0.005 | 0.474 ± 0.017 | 0.366 ± 0.018 | 0.672 ± 0.016 | 5 |
