# Threshold choice and error analysis (training split, out-of-fold)

Every number here comes from out-of-fold predictions on the 24,000 training clients: each client was scored by a model that did not see it. The test split is not used. Full tables: `error_analysis_tables.md` (`make analysis`). Figures: `figures/cost_curve.png`, `figures/permutation_importance.png`.

## Decision threshold

**Cost assumption:** missing a defaulter (false negative) costs 5 times as much as wrongly flagging a good client (false positive). A flagged client could be sent for manual review or given a lower limit; a missed defaulter is a written-off balance. The 5:1 ratio is an assumption to be replaced with the lender's real numbers.

**Chosen threshold: 0.16.** It minimises the expected cost on a 0.01 grid. Decision theory gives the same answer: when probabilities are calibrated, the optimum is cost_FP / (cost_FP + cost_FN) = 1/6 ≈ 0.167. The model is well calibrated (mean predicted probability matches the observed default rate within 1 pp in every segment below).

| At threshold 0.16 | Value |
| --- | --- |
| Recall (defaulters flagged) | 76.6% |
| Precision (flagged clients who default) | 37.3% |
| Clients flagged | 45.5% |
| Cost per client | 0.544 |
| Cost per client with no model (approve everyone) | 1.106 |

The model **cuts the expected cost of errors by 51%** compared with approving everyone. The price is that almost half of clients are flagged. The cost curve is flat between about 0.12 and 0.25, so a lender who can only review fewer clients loses little by raising the threshold a bit.

## What drives the predictions

Permutation importance (drop in PR-AUC when a feature is shuffled, averaged over 5 validation folds):

1. `PAY_1` (most recent repayment status) dominates: 0.126.
2. The two engineered features come next: `months_late` (0.031) and `max_months_late` (0.017). They summarise the 6-month repayment history and earn their place.
3. Everything else is at most 0.012: last bill, payments, utilisation and the credit limit each add a little.

The model is essentially a refined reading of repayment history. That is sensible for credit risk, and it is also its main weakness (below).

## Where the model is wrong

1. **Defaulters with a clean history are the blind spot.** 1,242 of 5,309 defaulters (23%) are missed. 90.4% of those missed defaulters were never late in the last 6 months, although only 34.5% of all defaulters have a clean history. Among clients who were never late, the model catches only 38.6% of defaulters. These clients look identical to good clients in this data. Catching them would need new information (income changes, other debts, a bureau score), not a different algorithm.

2. **Anyone who was late is flagged, so most of the false alarms are there.** 87% of clients who were one month late are flagged, and 83% of the good clients among them, because a third of them do default. For clients two or more months late the model flags 99.8%. This is the cost assumption doing its job, but a lender would want a cheaper action than refusal for this group (a reminder, a lower limit).

3. **Low credit limits are flagged much more often.** In the lowest limit quintile 71% of clients are flagged and 64% of good clients are false alarms, against 20% in the highest quintile. The model follows the bank's own earlier risk judgement (the limit), which can entrench it.

## Fairness audit

Sex and age are not model inputs, but correlated features can still produce group differences.

| Group | Default rate | Flagged | False positive rate | Mean predicted |
| --- | --- | --- | --- | --- |
| Female | 20.6% | 43.4% | 35.1% | 21.1% |
| Male | 24.4% | 48.5% | 39.0% | 23.4% |
| Age 21–25 | 26.9% | 58.6% | 50.0% | 27.1% |
| Age 26–40 | 20.3% | 42.0% | 33.8% | 20.6% |

The model stays calibrated within each group. The flag-rate differences mostly follow the real difference in default rates. But good clients aged 21–25 are flagged half the time (50.0% false positive rate vs 33.8% for ages 26–40), mostly through their lower credit limits and shorter repayment records. This would need a documented review before any real use.
