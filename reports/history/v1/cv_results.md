# Cross-validation results (training split, stratified 5-fold)

Mean ± standard deviation over the 5 folds. F1, recall and precision use the default 0.5 threshold; the decision threshold is chosen separately, from the cost of errors.

| Model | PR-AUC | ROC-AUC | F1@0.5 | Recall@0.5 | Precision@0.5 | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (prior) | 0.221 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1 |
| logistic regression | 0.515 ± 0.012 | 0.765 ± 0.007 | 0.420 ± 0.019 | 0.311 ± 0.018 | 0.647 ± 0.021 | 1 |
| random forest | 0.556 ± 0.009 | 0.782 ± 0.008 | 0.477 ± 0.019 | 0.370 ± 0.021 | 0.675 ± 0.013 | 34 |
| xgboost | 0.561 ± 0.012 | 0.784 ± 0.005 | 0.479 ± 0.019 | 0.373 ± 0.019 | 0.670 ± 0.015 | 6 |
| logistic regression, without SEX/MARRIAGE/AGE | 0.513 ± 0.012 | 0.764 ± 0.007 | 0.418 ± 0.019 | 0.308 ± 0.018 | 0.648 ± 0.021 | 1 |
| xgboost, without SEX/MARRIAGE/AGE | 0.560 ± 0.011 | 0.783 ± 0.005 | 0.478 ± 0.014 | 0.370 ± 0.015 | 0.676 ± 0.014 | 5 |
