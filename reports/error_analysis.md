# Threshold choice and error analysis (training split, out-of-fold)

Every number here comes from out-of-fold predictions on the 24,000 training clients: each client was scored by a model that did not see it. The test split is not used. Full tables: `error_analysis_tables.md` (`make analysis`). Figures: `figures/cost_curve.png`, `figures/permutation_importance.png`.

## Cost model

- A missed defaulter (false negative) is assumed to cost **5 units**, and wrongly flagging a client who would have paid (false positive) **1 unit**. Correct decisions cost nothing. In particular, flagging a real defaulter is assumed to prevent the loss at no cost.
- "Cost per client" is the average error cost per client, in units of one false alarm.
- The ratio is an assumption, chosen to represent a cheap intervention (a review, a reminder, a lower limit) against an expensive miss. A real lender would replace it with measured costs. The sensitivity table below shows how much the conclusions depend on it.

## Decision threshold

**Chosen threshold: 0.16**, the threshold that minimises the expected cost on a 0.01 grid.
- For this cost matrix the Bayes-optimal threshold for perfectly calibrated probabilities is 1/(1+5) ≈ 0.167, so the result is consistent with theory.
- Calibration near the threshold is good: predictions between 0.12 and 0.22 match the observed default rate within about 1 percentage point.
- **The exact value is partly luck.** The cost-optimal threshold within each of the 5 folds is 0.16, 0.20, 0.16, 0.16, 0.20.
- The cost is within 2% of its minimum for thresholds 0.15–0.20, and within 5% for 0.12–0.24.

## Against simple policies

| Policy | Clients flagged | Recall | Cost per client | Cut vs flagging everyone |
| --- | --- | --- | --- | --- |
| Approve everyone | 0% | 0% | 1.106 | −42% |
| Flag everyone | 100% | 100% | 0.779 | — |
| Rule: late last month (`PAY_1 ≥ 1`) | 22.9% | 52.2% | 0.643 | 17.5% |
| Rule: ever late in 6 months | 33.8% | 65.5% | 0.574 | 26.3% |
| **Model, threshold 0.16** | **45.5%** | **76.7%** | **0.543** | **30.2%** |

- Under the 5:1 assumption, flagging everyone is cheaper than approving everyone, so it is the fair trivial baseline.
- **A one-line rule with no fitted parameters already gets most of the way.** "Flag anyone who was late in the last 6 months" cuts the cost by 26.3%. The model adds a further 5.4% reduction over that rule (0.543 vs 0.574; 95% CI 3.1–7.5%, paired bootstrap, [rule_margin.json](rule_margin.json)), and catches 11 points more of the defaulters.
- **That margin, not the 30% against flagging everyone, is the value the model adds.**

## Sensitivity to the cost ratio

The threshold is re-chosen on the out-of-fold predictions for each ratio, with the false-alarm cost fixed at 1.

| Missed : false alarm | Threshold | Model cut vs best trivial policy | "Ever late" rule cut vs best trivial | Model cut vs the rule |
| --- | --- | --- | --- | --- |
| 2:1 | 0.32 | 31.3% | 22.0% | 11.9% |
| 3:1 | 0.27 | 39.6% | 36.5% | 4.9% |
| 4:1 | 0.21 | 38.1% | 36.1% | 3.2% |
| **5:1** | **0.16** | **30.2%** | **26.3%** | **5.4%** |
| 7:1 | 0.12 | 19.0% | 6.7% | 13.2% |
| 10:1 | 0.07 | 10.3% | −22.7% | 26.9% |

- The model beats the best trivial policy at every ratio.
- **Its advantage over the simple rule depends on the ratio:**
  - 3–5% for ratios 3–5;
  - 12% at 2:1;
  - 13–27% when misses are very expensive (7:1 and 10:1). At 10:1 the rule is worse than flagging everyone.

## Calibration

- **Overall:** the mean predicted probability is 0.220, against an observed default rate of 0.221.
- **Scores:** Brier score 0.134 (0.172 for a model that always predicts the average); expected calibration error 0.011.
- **By decile, the model is mildly overconfident:**
  - It under-predicts in the lowest deciles, by up to 1.5 percentage points: decile 1 predicts 3.2% against 4.5% observed.
  - It over-predicts in the top deciles, by up to 1.6 points.
- **No class weighting or resampling was used,** deliberately: the probabilities have to stay calibrated for the cost-based threshold to be meaningful.

## What drives the predictions

Permutation importance (drop in PR-AUC when a feature is shuffled, averaged over 5 validation folds):

1. `PAY_1` (most recent repayment status) dominates: 0.125.
2. The two engineered features come next: `months_late` (0.031) and `max_months_late` (0.017). They summarise the 6-month repayment history.
3. Everything else is at most 0.014 (`BILL_AMT1`).

**Caveats:**
- Permutation importance measures what a feature adds *given all the others*. Strongly related features share their credit: the six bill amounts correlate at 0.93–0.95 month to month, and `PAY_1` overlaps with the engineered lateness features. A low score for one bill amount means it is replaceable, not uninformative.
- `months_late` counts months with a *positive status code*. The codes are not used consistently across months (see `eda.md`), so it is not a true count of months of delay.

## Where the model is wrong

1. **Defaulters with a clean history are the blind spot.**
   - 1,237 of 5,309 defaulters (23%) are missed.
   - 90.6% of those had no positive repayment status in the last 6 months, although only 34.5% of all defaulters have a clean history.
   - Among clients who were never late, the model catches only 38.7% of defaulters.
   - These clients look identical to good clients in this data. Catching them would need new information (income changes, other debts, a bureau score), not a different algorithm.

2. **Anyone who was late is flagged, so most of the false alarms are there.**
   - 87% of clients who were one month late are flagged, including 83% of the good clients among them, because 34.4% of that group do default.
   - For clients two or more months late, the model flags 99.8%.
   - This is the cost assumption at work. A lender would want a cheaper action than refusal for this group.

3. **Low credit limits are flagged much more often.**
   - In the lowest limit quintile, 71% of clients are flagged and 63% of good clients are false alarms.
   - In the highest quintile, the figures are 25% and 20%.
   - The model follows the bank's own earlier risk judgement (the limit), which can entrench it.

## Fairness audit

Sex and age are not model inputs, but correlated features can still produce group differences.
- **The test:** for each group, does the mean prediction match the observed default rate? The standard error assumes the predictions are calibrated, so its variance comes from the predicted probabilities.
- **Correction:** six groups are audited, so significance is Bonferroni-corrected (|z| > 2.64).

| Group | Clients | Default rate | Mean predicted | Gap (pp) | z | Significant (corrected) | Flagged | False positive rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Female | 14,514 | 20.6% | 21.1% | +0.44 | +1.50 | no | 43.6% | 35.3% |
| Male | 9,486 | 24.4% | 23.4% | −0.97 | −2.59 | no (borderline) | 48.4% | 38.8% |
| Age 21–25 | 3,104 | 26.9% | 27.1% | +0.16 | +0.24 | no | 58.3% | 49.6% |
| Age 26–40 | 14,336 | 20.3% | 20.6% | +0.39 | +1.33 | no | 42.3% | 34.1% |
| Age 41–60 | 6,346 | 23.8% | 22.5% | −1.32 | −2.90 | **yes** | 46.4% | 36.6% |
| Age 61+ | 214 | 27.6% | 25.1% | −2.43 | −0.97 | no | 49.5% | 36.8% |

- **Clients aged 41–60 are slightly under-predicted.** The model gives them about 1.3 points less risk than they show, so it is *lenient* on this group, not harsh. Men show the same direction, just short of significance.
- **Young clients bear more false alarms.** Good clients aged 21–25 are wrongly flagged 49.6% of the time, against 34.1% for ages 26–40. They are not mis-calibrated: their mean prediction matches their default rate.
- **Credit limits explain all of that gap.** The median limit is NT$50k at ages 21–25 against NT$160k at 26–40. Reweighting the young good clients to the credit-limit mix of ages 26–40 lowers their false positive rate to 30.4%.
- The model does not use age, but it inherits the bank's lower limits for young clients, and it slightly understates risk for middle-aged clients. Both would need a documented review before any real use.
