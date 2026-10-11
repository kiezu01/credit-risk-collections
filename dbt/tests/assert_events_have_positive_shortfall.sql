-- A collections entry must have something left to pay on the entry day.
select event_id, due_amt, shortfall_amt
from {{ ref('mart_collections_events') }}
where due_amt <= 0 or shortfall_amt <= 0
