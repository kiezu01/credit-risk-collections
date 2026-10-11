-- Payment rows for instalments of earlier Home Credit loans.
-- One instalment can have several payment rows (partial payments) or none.
select
    SK_ID_PREV                  as prev_loan_id,
    SK_ID_CURR                  as customer_id,
    NUM_INSTALMENT_VERSION      as instalment_version,
    NUM_INSTALMENT_NUMBER       as instalment_number,
    DAYS_INSTALMENT             as due_day,
    DAYS_ENTRY_PAYMENT          as payment_day,
    AMT_INSTALMENT              as due_amt,
    AMT_PAYMENT                 as payment_amt
from {{ raw('installments_payments') }}
