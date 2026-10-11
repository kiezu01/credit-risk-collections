select
    customer_id,
    count(*)                                                    as bureau_credit_cnt,
    count(*) filter (where credit_status = 'Active')            as bureau_active_cnt,
    count(*) filter (where credit_status = 'Bad debt')          as bureau_bad_debt_cnt,
    coalesce(sum(overdue_amt), 0)                               as bureau_overdue_amt,
    max(max_overdue_amt)                                        as bureau_max_overdue_amt,
    max(days_overdue)                                           as bureau_max_days_overdue,
    sum(debt_amt) filter (where credit_status = 'Active')       as bureau_active_debt_amt,
    sum(debt_amt) / nullif(sum(credit_amt), 0)                  as bureau_debt_ratio,
    -max(days_credit_opened)                                    as bureau_days_since_last_credit,
    sum(prolong_cnt)                                            as bureau_prolong_cnt
from {{ ref('stg_bureau') }}
group by customer_id
