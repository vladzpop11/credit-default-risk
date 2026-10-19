# Model card: credit card default risk

The structure follows [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2019). Every number here comes from a file in `reports/`, linked where it is used.

## Model details

- **What it is:** a gradient-boosted tree classifier (XGBoost 3.4.1) inside a scikit-learn `Pipeline`, plus a decision threshold of **0.16**.
- **Settings:** 400 trees, learning rate 0.05, depth 4, row and column subsampling 0.8, seed 42. They were fixed in advance; nothing was tuned.
- **Version:** package 0.1.0, trained 2026-10-18 on macOS x86_64. The evaluated file has SHA-256 `8e01702c…2639`, recorded in [reports/test_metrics.json](reports/test_metrics.json). `GET /health` reports `"evaluated_model": true` only for that file.
- **Author:** [vladzpop11](https://github.com/vladzpop11), as a portfolio project. It is not used by any lender.

## Intended use

- **For:** ranking a card issuer's existing clients by their risk of missing next month's payment, and flagging those above the threshold for a **cheap** action: a review, a reminder or a lower limit.
- **Not for:**
  - Deciding credit applications, or any action that is costly for the client. The threshold assumes a false alarm costs a fifth of a missed defaulter, which only holds for cheap actions.
  - Clients outside the training population: Taiwan, 2005, existing card holders. Use on current data would need revalidation.
  - Automatic decisions without human review. See [Fairness](#fairness).

## Inputs and output

- **Inputs (20):** credit limit, education, and six months of repayment status, bill amounts and payments. `PAY_1` is the UCI column `PAY_0` (September 2005).
- **Not inputs:** sex, marital status and age. The API rejects them, and a test checks that they have no effect on the score (see [Behavioural tests](#behavioural-tests)).
- **Output:** `default_probability` (0–1), and `default_prediction` (probability ≥ 0.16).

## Data

- **Source:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Yeh & Lien, 2009), UCI, CC BY 4.0. 30,000 clients, 22.1% default.
- **Split:** stratified 80/20 into 24,000 training and 6,000 test clients. Both files are pinned by SHA-256 in `configs/config.yaml`.
- **Training:** the 24,000 training clients. The threshold was chosen on their out-of-fold predictions.
- **Evaluation:** the 6,000 test clients, from the **same months** as the training clients. The test set was evaluated twice, both results kept; see the README's [Test-set use](README.md#test-set-use).

## Performance

On the 6,000 test clients, threshold 0.16, with 95% bootstrap intervals ([test_metrics.json](reports/test_metrics.json), [test_uncertainty.json](reports/test_uncertainty.json)):

| Metric | Test | Out-of-fold (train) |
| --- | --- | --- |
| PR-AUC (no-skill 0.221) | 0.554 (0.527–0.585) | 0.559 |
| ROC-AUC | 0.776 | 0.783 |
| Recall (defaulters caught) | 74.6% (72.5–76.9%) | 76.7% |
| Precision | 36.5% | 37.3% |
| Clients flagged | 45.2% | 45.5% |
| Error cost per client (false alarm = 1, miss = 5) | 0.568 | 0.543 |

**Against simple policies** ([error_analysis.md](reports/error_analysis.md)):

- **Flagging everyone** costs 0.779 per client. The model cuts that by 27% (23–31%) on test.
- **The rule "flag anyone ever late in 6 months"** costs 0.574 out-of-fold, against the model's 0.543. So most of the gain comes from a rule with no parameters; the model adds about 5% on top. The rule was not evaluated on the test set.
- **How much the model adds depends on the assumed 5:1 cost ratio:** 3% (at 4:1) to 27% (at 10:1) over the rule.

**Calibration** (out-of-fold): Brier score 0.134. Probabilities near the threshold are within about 1 percentage point of the observed rate. The extremes are mildly overconfident, by up to 1.6 points.

## Fairness

Sex, marital status and age are not inputs. Dropping them changed cross-validated PR-AUC by −0.0002 (corrected p = 0.94). The other features can still carry group differences. Audit on out-of-fold predictions for the training clients ([error_analysis.md](reports/error_analysis.md#fairness-audit)):

| Group | Default rate | Mean predicted | False positive rate |
| --- | --- | --- | --- |
| Female | 20.6% | 21.1% | 35.3% |
| Male | 24.4% | 23.4% | 38.8% |
| Age 21–25 | 26.9% | 27.1% | **49.6%** |
| Age 26–40 | 20.3% | 20.6% | 34.1% |
| Age 41–60 | 23.8% | **22.5%** | 36.6% |
| Age 61+ (214 clients) | 27.6% | 25.1% | 36.8% |

- **Young clients get more false alarms.** Good clients aged 21–25 are flagged 49.6% of the time, against 34.1% at 26–40. Their scores are calibrated, though. Most of the gap comes from lower credit limits: at the 26–40 limit mix, their false positive rate would be 30.4%. The model inherits the bank's earlier limit decisions.
- **Clients aged 41–60 are slightly under-predicted:** by 1.3 points, significant after Bonferroni correction for six groups. The model is lenient on them, not harsh. Men show the same direction, just short of significance.
- **Before any real use,** both findings would need a documented review, and the false positive rates should be monitored by group.

## Behavioural tests

[tests/test_model_behaviour.py](tests/test_model_behaviour.py) probes the trained model with edited copies of the training clients (never the test clients). `make test-model` runs it, and CI runs it on the model it retrains on Linux.

| Check | Required | macOS (evaluated model) | Linux retrain (CI, simulated in Docker) |
| --- | --- | --- | --- |
| Sex, marital status, age changed or removed | identical scores | identical | identical |
| One client at a time vs in a batch, shuffled | identical scores | identical | identical |
| Paid on time → 1 month late | risk up for ≥ 90% | 96.8% | 96.0% |
| 1 → 2 months late | risk up for ≥ 95% | 99.8% | 99.7% |
| Pays the previous bill in full | mean down; down for ≥ 65% | 75.5% | 75.3% |
| Stops paying (payment → 0) | mean up; up for ≥ 70% | 81.9% | 83.6% |
| 3 → 6 months late | risk not lower for ≥ 90% | **fails: lower for 96.6%** | **fails: lower for 87.4%** |
| Latency, one client, in-process | median < 250 ms | 24 ms | passes (time not recorded) |

- **The 3 → 6 months failure is a known flaw,** marked as an expected failure (`xfail(strict=True)`). Predicted risk peaks at three months late and falls for longer delays, because the training data has very few such clients. Monotonic constraints would fix it. When they do, the test starts passing and the strict marker fails the run, so the marker cannot be forgotten.
- **The payment effects are right on average but not for every client.** Paying the previous bill in full *raises* the score for 24% of clients who had been underpaying. Among clients who were not late, that share is 29%, and the average effect is small (−0.013). For clients with revolving credit (`PAY_1 = 0`), the average score falls from 0.147 when they pay nothing to 0.119 when they pay a quarter of the bill, and paying more barely moves it. The trees fit no smooth payment effect, which is one more reason for monotonic constraints.
- **Thresholds were set from the measured values,** with room for the Linux retrain. They are regression checks, not proof of the behaviour in general.

## Limitations

The full list is in the README's [Limitations](README.md#limitations). The ones that matter most for use:

- **Defaulters with a clean history are invisible.** 91% of the missed defaulters had no late payment in 6 months. The data has no income, other debts or bureau score.
- **The cost ratio is an assumption,** and it decides the threshold and how many clients are flagged.
- **Old, single-market data,** and a test set from the same months as training: nothing here shows performance on later months.
- **The model behaves oddly where data is thin:** long delays (above), "other" education, and credit limits below the training minimum.

## How to check this card

```bash
make verify          # models/model.joblib is the evaluated file (SHA-256)
make test-model      # behavioural tests above
make uncertainty     # re-scores the test set, stops unless it matches test_metrics.json exactly
```
