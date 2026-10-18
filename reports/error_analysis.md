# Threshold choice and error analysis (training split, out-of-fold)

Every number here comes from out-of-fold predictions on the 24,000 training clients: each client was scored by a model that did not see it. The test split is not used. Full tables: `error_analysis_tables.md` (`make analysis`). Figures: `figures/cost_curve.png`, `figures/permutation_importance.png`.

## Decision threshold

**Cost assumption:** missing a defaulter (false negative) costs 5 times as much as wrongly flagging a good client (false positive). A flagged client could be sent for manual review or given a lower limit; a missed defaulter is a written-off balance. The 5:1 ratio is an assumption to be replaced with the lender's real numbers.

**Chosen threshold: 0.16.** It minimises the expected cost on a 0.01 grid. Decision theory gives the same answer: when probabilities are calibrated, the optimum is cost_FP / (cost_FP + cost_FN) = 1/6 ≈ 0.167. Overall, the model is well calibrated: the mean predicted probability is 0.220, against an observed default rate of 0.221.

| At threshold 0.16 | Value |
| --- | --- |
| Recall (defaulters flagged) | 76.7% |
| Precision (flagged clients who default) | 37.3% |
| Clients flagged | 45.5% |
| Cost per client | 0.543 |
| Cost per client with no model (approve everyone) | 1.106 |

The model **cuts the expected cost of errors by 51%** compared with approving everyone. The price is that almost half of clients are flagged. The cost curve is flat between about 0.12 and 0.25, so a lender who can only review fewer clients loses little by raising the threshold a bit.

## What drives the predictions

Permutation importance (drop in PR-AUC when a feature is shuffled, averaged over 5 validation folds):

1. `PAY_1` (most recent repayment status) dominates: 0.125.
2. The two engineered features come next: `months_late` (0.031) and `max_months_late` (0.017). They summarise the 6-month repayment history and earn their place.
3. Everything else is at most 0.014 (`BILL_AMT1`).

**Caveat:** permutation importance measures what a feature adds *given all the others*. Strongly related features share their credit and each looks unimportant. The six bill amounts correlate at 0.93–0.95 month to month, and `PAY_1` overlaps with the engineered lateness features. So a low score for an individual bill amount does not mean the bills carry no information, only that any one of them is replaceable.

The model is essentially a refined reading of repayment history. That is sensible for credit risk, and it is also its main weakness (below).

## Where the model is wrong

1. **Defaulters with a clean history are the blind spot.** 1,237 of 5,309 defaulters (23%) are missed. 90.6% of those missed defaulters were never late in the last 6 months, although only 34.5% of all defaulters have a clean history. Among clients who were never late, the model catches only 38.7% of defaulters. These clients look identical to good clients in this data. Catching them would need new information (income changes, other debts, a bureau score), not a different algorithm.

2. **Anyone who was late is flagged, so most of the false alarms are there.** 87% of clients who were one month late are flagged, and 83% of the good clients among them, because a third of them (34.4%) do default. For clients two or more months late the model flags 99.8%. This is the cost assumption doing its job, but a lender would want a cheaper action than refusal for this group (a reminder, a lower limit).

3. **Low credit limits are flagged much more often.** In the lowest limit quintile, 71% of clients are flagged and 63% of good clients are false alarms. In the highest quintile the figures are 25% and 20%. The model follows the bank's own earlier risk judgement (the limit), which can entrench it.

## Calibration by segment

In every segment defined by model inputs (repayment status, months late, credit-limit quintile, education), the mean predicted probability is within 0.8 percentage points of the observed default rate.

## Fairness audit

Sex and age are not model inputs, but correlated features can still produce group differences.

| Group | Clients | Default rate | Mean predicted | Flagged | False positive rate |
| --- | --- | --- | --- | --- | --- |
| Female | 14,514 | 20.6% | 21.1% | 43.6% | 35.3% |
| Male | 9,486 | 24.4% | 23.4% | 48.4% | 38.8% |
| Age 21–25 | 3,104 | 26.9% | 27.1% | 58.3% | 49.6% |
| Age 26–40 | 14,336 | 20.3% | 20.6% | 42.3% | 34.1% |
| Age 41–60 | 6,346 | 23.8% | 22.5% | 46.4% | 36.6% |
| Age 61+ | 214 | 27.6% | 25.1% | 49.5% | 36.8% |

- **Calibration differs a little by group.** The model slightly over-predicts for women (+0.5 pp) and under-predicts for men (−1.0 pp) and for ages 41–60 (−1.3 pp). With 6,346 clients aged 41–60 the sampling error is about ±0.5 pp, so that gap is probably real, though small. For ages 61+ the gap is −2.5 pp, but with only 214 clients its sampling error is about ±3.1 pp, so that gap cannot be distinguished from noise.
- **Flag rates mostly follow the real differences in default rates.**
- **Young clients bear more false alarms.** Good clients aged 21–25 are flagged half the time (49.6% false positive rate, against 34.1% for ages 26–40), mostly through their lower credit limits and shorter repayment records. This would need a documented review before any real use.
