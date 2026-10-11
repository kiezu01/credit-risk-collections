select
    customer_id,
    count(*)                                                        as prev_app_cnt,
    count(*) filter (where contract_status = 'Approved')            as prev_approved_cnt,
    avg(case when contract_status = 'Refused' then 1.0 else 0.0 end) as prev_refused_rate,
    avg(credit_amt / nullif(applied_amt, 0))                        as prev_credit_to_applied_ratio,
    -max(days_decision)                                             as prev_days_since_last_decision
from {{ ref('stg_previous_applications') }}
group by customer_id
