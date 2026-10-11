-- One row per collections entry on a fixed-term instalment loan: an instalment still not (fully) paid N days after its
-- due date (the day it would land in an early-collections queue).
-- Label is_cured: paid within the cure window after the due date (otherwise it rolls to DPD 30+).
-- Every feature uses only information available on the entry day, so the model can be
-- validated out-of-time.
{% set window = var('cure_window_days') %}
{% set entry = var('collections_entry_dpd') %}

with inst as (
    select * from {{ ref('int_instalments') }}
),

history as (
    select
        *,
        -- earlier instalments of this customer: lateness at their due date is known by now
        count(*) over w_before                                          as prior_inst_cnt,
        sum(case when is_delinquent_at_due then 1 else 0 end) over w_before
                                                                        as prior_delinquent_cnt,
        max(case when is_collections_entry then due_day end) over w_before
                                                                        as last_entry_due_day,
        -- earlier collections entries whose cure window had already closed on today's entry day
        sum(case when is_collections_entry then 1 else 0 end) over w_known
                                                                        as prior_known_entry_cnt,
        sum(case when is_collections_entry and is_paid_in_window then 1 else 0 end) over w_known
                                                                        as prior_cured_cnt,
        -- same loan
        sum(case when is_delinquent_at_due then 1 else 0 end) over w_loan_before
                                                                        as loan_prior_delinquent_cnt,
        count(*) over w_loan_before                                     as loan_prior_inst_cnt
    from inst
    window
        w_before as (partition by customer_id order by due_day
                     range between unbounded preceding and 1 preceding),
        -- outcome of an entry due on d' is known on d' + window; today's entry day is due + entry
        w_known as (partition by customer_id order by due_day
                    range between unbounded preceding and {{ window - entry + 1 }} preceding),
        w_loan_before as (partition by prev_loan_id order by due_day
                          range between unbounded preceding and 1 preceding)
),

loans as (
    select prev_loan_id, contract_type as loan_contract_type, credit_amt as loan_credit_amt,
           term_months as loan_term_months
    from {{ ref('stg_previous_applications') }}
)

select
    h.instalment_id                                         as event_id,
    h.customer_id,
    h.prev_loan_id,
    h.due_day,
    h.instalment_number,
    h.due_amt,
    h.shortfall_at_entry_amt                                as shortfall_amt,
    h.shortfall_at_entry_amt / h.due_amt                    as shortfall_ratio,
    h.paid_by_entry_amt > 0                                 as is_partial_payment,
    coalesce(h.prior_inst_cnt, 0)                           as prior_inst_cnt,
    coalesce(h.prior_delinquent_cnt, 0)                     as prior_delinquent_cnt,
    coalesce(h.prior_delinquent_cnt, 0) / nullif(h.prior_inst_cnt, 0)
                                                            as prior_delinquent_rate,
    coalesce(h.prior_known_entry_cnt, 0)                    as prior_known_entry_cnt,
    coalesce(h.prior_cured_cnt, 0) / nullif(h.prior_known_entry_cnt, 0)
                                                            as prior_cure_rate,
    h.due_day - h.last_entry_due_day                        as days_since_last_entry,
    coalesce(h.loan_prior_delinquent_cnt, 0)                as loan_prior_delinquent_cnt,
    coalesce(h.loan_prior_inst_cnt, 0)                      as loan_prior_inst_cnt,
    l.loan_contract_type,
    l.loan_credit_amt,
    l.loan_term_months,
    h.instalment_number / nullif(l.loan_term_months, 0)     as loan_progress_ratio,
    -- stable customer attributes only: scores such as EXT_SOURCE are measured at the
    -- current application (after these events) and would leak future information
    a.gender,
    a.education_type,
    a.family_status,
    a.income_type,
    a.age_years + h.due_day / 365.25                        as age_at_due_years,
    -- label
    h.is_paid_in_window                                     as is_cured
from history as h
left join loans as l using (prev_loan_id)
left join {{ ref('stg_applications') }} as a using (customer_id)
where h.is_collections_entry
  -- the cure window must be fully observed before the data cut-off (day 0)
  and h.due_day <= -{{ window + 1 }}
  -- scope: fixed-term instalment loans. Revolving lines (card minimum payments, no term) follow a
  -- different collections process, and instalments past the contract term are restructured
  -- schedules; mixing them in lets a model "predict" the product type instead of the outcome.
  and l.loan_contract_type in ('Cash loans', 'Consumer loans')
  and h.instalment_number <= l.loan_term_months
