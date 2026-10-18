# Credit Card Default Risk

Predict whether a credit card client will miss next month's payment, choose the decision threshold from the cost of each kind of error, and serve the model as a containerised API.

**Result:** on a held-out test set of 6,000 clients, the model cuts the expected cost of errors by **49%** compared with approving everyone. It catches **75% of defaulters** (PR-AUC 0.556 against a 0.221 no-skill baseline), without using sex, marital status or age.

## Problem

A card issuer reviews its clients every month. For each client the model estimates the probability of defaulting on the next payment, and clients above a threshold are flagged for action (manual review, a reminder or a lower limit).

Missing a defaulter is assumed to cost **5 times** as much as flagging a client who would have paid. Success is measured as the expected cost per client against the policy of flagging no one. PR-AUC is the ranking metric, because only 22% of clients default and accuracy would be misleading.

## Data

[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009), UCI Machine Learning Repository, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data covers 30,000 clients of a Taiwanese bank, observed April to September 2005, with the target being October 2005. The features are the credit limit, demographics, and 6 months of repayment status, bill amounts and payments.

`make data` downloads the data and records the source, date and SHA-256 hash in `data/raw/SOURCE.md`.

## Approach

1. **Split first.** The data is split 80/20 into train and test, stratified at 22.12% defaults in both. The test set was used exactly once, at the end; `make evaluate` refuses to run a second time.
2. **Exploration on train only** ([reports/eda.md](reports/eda.md)). Recent repayment status dominates: clients late last month default at 50.4%, against 13.7% for everyone else. The categorical codes contain undocumented values, and the bill amounts are highly collinear and skewed.
3. **No-leakage pipeline.** Cleaning and engineered features (`months_late`, `max_months_late`, utilisation, payment ratio) are computed row by row. Imputation, scaling and one-hot encoding are fitted inside a scikit-learn `Pipeline` on each training fold only. The same `prepare()` function runs in training and in the API.
4. **Model selection** with stratified 5-fold cross-validation ([reports/cv_results.md](reports/cv_results.md)): a prior baseline, logistic regression, random forest and XGBoost. XGBoost was chosen before the test set was opened.
5. **Sensitive attributes removed.** Dropping sex, marital status and age cost 0.0015 PR-AUC in cross-validation, against a fold-to-fold standard deviation of 0.011. They are not model inputs, and the API does not accept them.
6. **Cost-based threshold** ([reports/error_analysis.md](reports/error_analysis.md)). The threshold 0.16 minimises expected cost on out-of-fold predictions. It matches the decision-theory optimum 1/(1+5) ≈ 0.167, because the model is well calibrated.

## Results

Cross-validation on the training split (5 folds, mean ± std):

| Model | PR-AUC | ROC-AUC |
| --- | --- | --- |
| Baseline (prior) | 0.221 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.515 ± 0.012 | 0.765 ± 0.007 |
| Random forest | 0.556 ± 0.009 | 0.782 ± 0.008 |
| XGBoost | 0.561 ± 0.012 | 0.784 ± 0.005 |
| **XGBoost without sex, marital status, age (chosen)** | **0.560 ± 0.011** | **0.783 ± 0.005** |

Final model, threshold 0.16 ([reports/test_metrics.json](reports/test_metrics.json)):

| | Out-of-fold (train) | Test (once) |
| --- | --- | --- |
| PR-AUC | 0.559 | 0.556 |
| ROC-AUC | 0.783 | 0.776 |
| Recall | 76.6% | 74.7% |
| Precision | 37.3% | 36.8% |
| Clients flagged | 45.5% | 44.9% |
| Cost per client (no model: 1.106) | 0.544 | 0.564 |

The test PR-AUC is within 0.003 of cross-validation, so there is no sign of overfitting to the validation folds.

**Serving:** CPU inference in Docker on a 2-core 2018 MacBook Air takes 34 ms on average and 52 ms at p95 per request, end to end over HTTP (50 requests).

## Architecture

```mermaid
flowchart LR
  A["UCI dataset"] -->|make data| B["data/raw + SOURCE.md<br/>(hash, date)"]
  B -->|make split| C["train 80% / test 20%<br/>(stratified, seed 42)"]
  C -->|make cv| D["5-fold CV<br/>4 models"]
  C -->|make train| E["OOF probabilities<br/>→ cost-based threshold"]
  E --> F["models/model.joblib<br/>pipeline + threshold + metadata"]
  F -->|make evaluate, once| G["reports/test_metrics.json"]
  F -->|make docker-build| H["Docker image<br/>FastAPI + uvicorn"]
  H --> I["GET /health<br/>POST /predict"]
```

| File | Role |
| --- | --- |
| `src/credit_default_risk/features.py` | Cleaning, engineered features and preprocessing pipeline, shared by training and serving |
| `src/credit_default_risk/train.py` | Out-of-fold threshold choice, final fit, model bundle with library versions |
| `src/credit_default_risk/evaluate.py` | One-time test evaluation, with a guard against reuse |
| `src/credit_default_risk/api.py` | FastAPI service with a strict Pydantic schema |
| `.github/workflows/ci.yml` | Lint, tests, retraining from the public source, image build |

## How to run

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make demo        # download, split, train, build the image and start the API
```

Then open http://localhost:8000/docs, or:

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "LIMIT_BAL": 50000, "EDUCATION": 2,
  "PAY_1": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
  "BILL_AMT1": 48000, "BILL_AMT2": 47000, "BILL_AMT3": 46000,
  "BILL_AMT4": 30000, "BILL_AMT5": 28000, "BILL_AMT6": 25000,
  "PAY_AMT1": 0, "PAY_AMT2": 1500, "PAY_AMT3": 1500,
  "PAY_AMT4": 1200, "PAY_AMT5": 1000, "PAY_AMT6": 1000}'
make docker-stop
```

Other targets: `make check` (Ruff + pytest), `make eda`, `make cv`, `make analysis`, `make serve` (API without Docker).

## Limitations

- **Clean-history defaulters are a blind spot.** 90% of the missed defaulters were never late in the previous 6 months. The data has no income, other debts or bureau score, so these clients look like good ones.
- **The 5:1 cost ratio is an assumption.** At this ratio 45% of clients are flagged, which only makes sense if the action is cheap (a reminder or a review, not a refusal). The cost curve is flat between about 0.12 and 0.25, so the threshold can be raised with little loss.
- **Indirect group differences remain.** Without using age, good clients aged 21–25 are still flagged 50% of the time, against 34% for ages 26–40, mostly through lower credit limits. This needs a review before any real use.
- **The data is old and from one market** (Taiwan, 2005, a period of rising card debt). The model would need revalidation on current data, and monitoring for drift.

## Next steps

- Calibrate the cost matrix with real loss and review costs, and consider a three-way decision (approve / review / act).
- Add a fairness constraint or per-group threshold analysis, and document it in a model card.
- Track experiments and register the model in MLflow, and monitor drift on incoming data (covered in the MLOps capstone project).
