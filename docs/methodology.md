## Method

### Data and warehouse
Home Credit Default Risk (Kaggle): ~307k labelled loan applications, the applicants' credit
bureau records, their previous Home Credit applications and ~13.6M instalment payment rows.
Raw CSVs are modelled with **dbt on DuckDB** in three layers:

| Layer | Models | Purpose |
|---|---|---|
| staging | `stg_applications`, `stg_bureau`, `stg_previous_applications`, `stg_installment_payments` | rename, type, clean (e.g. the `365243` "not employed" placeholder) |
| intermediate | `int_instalments`, `int_*_by_customer` | instalment grain with paid-by-due / paid-in-window amounts; customer aggregates |
| marts | `mart_customer_features`, `mart_collections_events` | model-ready tables |

26 dbt data tests run on every build: keys unique and not null, label values, a plausible
default rate, and two point-in-time checks on the collections table.

### 1. Application scorecard (probability of default)
* Target: `TARGET = 1`, the client had payment difficulties (~8% of applications).
* 70/30 stratified split. Application data has no calendar date, so an out-of-time split is not
  possible here.
* Optimal **WoE binning** per feature (optbinning), features kept if **IV ≥ 0.02**, then logistic
  regression on WoE values.
* Scaled to points: **600 points at 50:1 odds, 20 points to double the odds**. Higher score =
  lower risk. Six equal-sized grades A–F are cut on the training scores.
* Reported: AUC, Gini, KS, PSI train vs. test, default rate by grade, approval/default trade-off
  by cut-off.
* **Challenger**: LightGBM on the same features, explained with SHAP. The gap between the two
  shows how much power the transparent model gives up.

### 2. Collections: roll-rate model and prioritisation
* **Event (collections entry)** = an instalment with less than 95% of the amount paid **5 days
  after its due date (DPD 5)**. About half of all late instalments are paid within 3 days, so a
  short grace period keeps the queue to accounts a collector would actually work.
* **Label** `is_cured` = at least 95% paid within **30 days** after the due date; otherwise the
  account **rolls to DPD 30+**. Events whose 30-day window is not fully observed before the data
  cut-off are excluded.
* **Point-in-time features**: the customer's earlier instalments (count, delinquency rate),
  earlier entries whose 30-day outcome was already known on the entry day (cure rate), days since
  the last entry, same-loan history, amount and share still unpaid at DPD 5, loan term and progress,
  stable demographics. Application scores such as `EXT_SOURCE_*` are **excluded**: they are
  measured at the current application, after these events, and would leak future information.
* **Out-of-time validation**: trained on the oldest 70% of events, validated on the latest 30%.
* **Prioritisation test**: events are grouped into weekly cohorts (the queue a team sees), ranked
  within each cohort by four strategies, and the top X% is "worked". The metric is the share of
  the overdue amount that was **not** self-cured and lands in the worked part of the queue. It
  uses observed outcomes only, with no assumed treatment effect.

### 3. A/B test design
Ranking well does not prove that calling changes behaviour. The dashboard sizes the randomised
test needed to measure the lift: two-sided z-test on two proportions, alpha 0.05, power 0.80.

### Limitations
* Home Credit `DAYS_*` fields are relative to each customer's own application date, not a
  calendar. "Out-of-time" therefore means "closer to the customer's latest application", and
  weekly cohorts mix calendar dates. A production version would use real dates.
* The data has no record of collection actions, so the true causal effect of a call cannot be
  estimated here. That is exactly what the A/B test is for.
* The daily volume used to turn sample sizes into test duration is an assumption (2,000 new
  delinquencies per day), editable in the dashboard.
