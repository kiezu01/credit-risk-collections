-- Guards against a broken label join: Home Credit's default rate is about 8%.
select avg(target_default) as default_rate
from {{ ref('mart_customer_features') }}
having avg(target_default) not between 0.02 and 0.30
