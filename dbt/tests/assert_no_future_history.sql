-- Point-in-time check: an event can never have more known past events than past instalments.
select event_id
from {{ ref('mart_collections_events') }}
where prior_known_entry_cnt > prior_inst_cnt
   or prior_delinquent_cnt > prior_inst_cnt
