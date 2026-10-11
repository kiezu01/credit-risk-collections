-- Credits the customer holds at other lenders, as reported by the credit bureau.
select
    SK_ID_BUREAU                as bureau_credit_id,
    SK_ID_CURR                  as customer_id,
    CREDIT_ACTIVE               as credit_status,
    CREDIT_TYPE                 as credit_type,
    DAYS_CREDIT                 as days_credit_opened,
    CREDIT_DAY_OVERDUE          as days_overdue,
    AMT_CREDIT_MAX_OVERDUE      as max_overdue_amt,
    CNT_CREDIT_PROLONG          as prolong_cnt,
    AMT_CREDIT_SUM              as credit_amt,
    AMT_CREDIT_SUM_DEBT         as debt_amt,
    AMT_CREDIT_SUM_OVERDUE      as overdue_amt
from {{ raw('bureau') }}
