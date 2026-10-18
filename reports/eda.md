# Exploratory analysis (training split only)

24,000 clients (80% of the data). The test split was not looked at. Figures: `reports/figures/`, produced by `make eda`.

## Conclusions

1. **The classes are imbalanced: 22.1% of clients default.** A model that predicts "no default" for everyone is 77.9% accurate, so accuracy is not a useful metric. PR-AUC is the primary metric; its no-skill baseline is 0.221.

2. **Recent repayment status is by far the strongest signal.** Clients who were at least one month late in the most recent month (`PAY_1 ≥ 1`, 22.9% of clients) default at 50.4%, against 13.7% for everyone else. At two months late the rate is 69.4%. Correlation with the target is 0.33 for `PAY_1` and fades steadily for older months (0.27 for `PAY_2` down to 0.19 for `PAY_6`). Recent behaviour matters most. See `default_by_pay1.png`.

3. **A higher credit limit means lower risk.** The default rate falls steadily from 32.2% in the lowest `LIMIT_BAL` quintile (NT$10k–50k) to 13.1% in the highest (NT$270k–1M). The median limit is NT$90k for defaulters and NT$150k for non-defaulters. The bank's own limit already encodes a risk judgement. See `default_by_limit.png`.

4. **The categorical codes do not match the dataset documentation.** `EDUCATION` has the undocumented values 0, 5 and 6 (289 rows, 1.2%). `MARRIAGE` has an undocumented 0 (47 rows). `PAY_*` uses −2 and 0, which the documentation does not define; they appear to mean "no balance" and "revolving credit, paid the minimum". The rare undocumented codes are merged into the existing "other" categories (`EDUCATION` → 4, `MARRIAGE` → 3). This is done in shared code so training and serving clean the data the same way.

5. **The monthly amounts are redundant and heavily skewed.** Bills in adjacent months correlate at 0.93–0.95 (0.80 between month 1 and month 6). Payments are very skewed (`PAY_AMT1` skewness 15.3). 6.5% of clients have a negative bill in at least one month (an overpayment). A payment settles the *previous* month's statement: `PAY_AMT1` equals `BILL_AMT2` exactly for 15.2% of clients with a positive bill, but equals `BILL_AMT1` for only 2.4%. Payment-to-bill features must pair each payment with the previous month's bill (found after the first evaluation; see `history/v1/`). The linear model gets a signed log transform and L2 regularisation; tree models are unaffected by either. See `correlations.png`.

6. **Sensitive attributes carry signal and need an explicit decision.** Men default at 24.4% and women at 20.6%. The youngest (21–25: 26.9%) and oldest (60+: 27.6%) age bands are riskier than 26–40 (20.2–20.3%). Using sex or marital status in a credit decision is restricted or prohibited in many jurisdictions. Cross-validation reports the chosen model both with and without `SEX`, `MARRIAGE` and `AGE`, so the trade-off is measured rather than assumed.

## Leakage check

All features describe April–September 2005 and the target is the October 2005 payment, so nothing comes from after the prediction date. `ID` is dropped. No single feature correlates suspiciously strongly with the target (the maximum is 0.33).
