-- Repayment behaviour on earlier Home Credit loans, all observed before the current application.
select
    customer_id,
    count(*)                                                        as inst_cnt,
    avg(case when is_delinquent_at_due then 1.0 else 0.0 end)       as inst_delinquent_rate,
    avg(days_late)                                                  as inst_avg_days_late,
    max(days_late)                                                  as inst_max_days_late,
    sum(shortfall_at_due_amt) / nullif(sum(due_amt), 0)             as inst_shortfall_ratio,
    count(*) filter (where is_delinquent_at_due and not is_paid_in_window)
                                                                    as inst_uncured_cnt
from {{ ref('int_instalments') }}
group by customer_id
