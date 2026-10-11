-- One row per scheduled instalment, with how much was paid by the due date, by the
-- collections entry day (due + entry DPD) and within the cure window.
{% set thr = var('paid_threshold') %}
{% set window = var('cure_window_days') %}
{% set entry = var('collections_entry_dpd') %}

with grouped as (
    select
        prev_loan_id,
        customer_id,
        instalment_number,
        instalment_version,
        due_day,
        max(due_amt)                                                         as due_amt,
        coalesce(sum(payment_amt), 0)                                        as paid_total_amt,
        coalesce(sum(payment_amt) filter (where payment_day <= due_day), 0)  as paid_by_due_amt,
        coalesce(sum(payment_amt) filter (where payment_day <= due_day + {{ entry }}), 0)
                                                                             as paid_by_entry_amt,
        coalesce(sum(payment_amt) filter (where payment_day <= due_day + {{ window }}), 0)
                                                                             as paid_in_window_amt,
        min(payment_day)                                                     as first_payment_day,
        max(payment_day)                                                     as last_payment_day
    from {{ ref('stg_installment_payments') }}
    where due_day is not null
    group by all
)

select
    *,
    {{ dbt_utils_surrogate(['prev_loan_id', 'instalment_number', 'instalment_version', 'due_day']) }}
                                                                     as instalment_id,
    due_amt > 0 and paid_by_due_amt < {{ thr }} * due_amt            as is_delinquent_at_due,
    due_amt > 0 and paid_by_entry_amt < {{ thr }} * due_amt          as is_collections_entry,
    due_amt > 0 and paid_in_window_amt >= {{ thr }} * due_amt        as is_paid_in_window,
    greatest(due_amt - paid_by_due_amt, 0)                           as shortfall_at_due_amt,
    greatest(due_amt - paid_by_entry_amt, 0)                         as shortfall_at_entry_amt,
    greatest(last_payment_day - due_day, 0)                          as days_late
from grouped
