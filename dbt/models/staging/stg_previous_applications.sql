-- Earlier applications the customer made at Home Credit.
select
    SK_ID_PREV                  as prev_loan_id,
    SK_ID_CURR                  as customer_id,
    NAME_CONTRACT_TYPE          as contract_type,
    NAME_CONTRACT_STATUS        as contract_status,
    AMT_APPLICATION             as applied_amt,
    AMT_CREDIT                  as credit_amt,
    AMT_ANNUITY                 as annuity_amt,
    CNT_PAYMENT                 as term_months,
    DAYS_DECISION               as days_decision
from {{ raw('previous_application') }}
