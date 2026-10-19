# Cross-validation results (training split, stratified 5-fold)

Mean ± standard deviation (ddof=1) over the 5 folds. All models use fixed, untuned settings. F1, recall and precision use the default 0.5 threshold; the decision threshold is chosen separately, from the cost of errors.

| Model | PR-AUC | ROC-AUC | F1@0.5 | Recall@0.5 | Precision@0.5 | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (prior) | 0.221 ± 0.000 | 0.500 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 1 |
| logistic regression | 0.515 ± 0.013 | 0.764 ± 0.008 | 0.419 ± 0.023 | 0.310 ± 0.021 | 0.646 ± 0.023 | 1 |
| random forest | 0.555 ± 0.012 | 0.782 ± 0.008 | 0.478 ± 0.019 | 0.370 ± 0.019 | 0.674 ± 0.014 | 40 |
| xgboost | 0.560 ± 0.013 | 0.784 ± 0.006 | 0.482 ± 0.017 | 0.376 ± 0.019 | 0.674 ± 0.014 | 6 |
| logistic regression, one-hot PAY codes | 0.551 ± 0.020 | 0.781 ± 0.009 | 0.474 ± 0.020 | 0.364 ± 0.021 | 0.682 ± 0.019 | 4 |
| logistic regression, one-hot PAY, without SEX/MARRIAGE/AGE | 0.549 ± 0.020 | 0.780 ± 0.009 | 0.470 ± 0.022 | 0.360 ± 0.020 | 0.680 ± 0.024 | 4 |
| xgboost, without SEX/MARRIAGE/AGE | 0.560 ± 0.013 | 0.783 ± 0.006 | 0.474 ± 0.019 | 0.366 ± 0.021 | 0.672 ± 0.018 | 5 |

## Paired PR-AUC differences against the chosen model (xgboost, without SEX/MARRIAGE/AGE)

Fold-wise difference (chosen minus other), its standard deviation, and the p-value of a paired t-test with the Nadeau-Bengio correction for overlapping CV training sets. With 5 folds the test has little power: p > 0.05 means the models cannot be distinguished, not that they are equal.

| Other model | Mean difference | SD | p-value |
| --- | --- | --- | --- |
| baseline (prior) | +0.3386 | 0.0130 | 0.000 |
| logistic regression | +0.0447 | 0.0100 | 0.003 |
| random forest | +0.0044 | 0.0046 | 0.233 |
| xgboost | -0.0002 | 0.0039 | 0.936 |
| logistic regression, one-hot PAY codes | +0.0091 | 0.0108 | 0.276 |
| logistic regression, one-hot PAY, without SEX/MARRIAGE/AGE | +0.0105 | 0.0116 | 0.250 |
