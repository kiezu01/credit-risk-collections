-- One row per loan application (train has TARGET, test does not).
-- DAYS_* columns are relative to the application date (negative = in the past).
with unioned as (
    select *, true as is_train from {{ raw('application_train') }}
    union all by name
    select *, false as is_train from {{ raw('application_test') }}
)

select
    SK_ID_CURR                                          as customer_id,
    is_train,
    cast(TARGET as integer)                             as target_default,
    NAME_CONTRACT_TYPE                                  as contract_type,
    CODE_GENDER                                         as gender,
    FLAG_OWN_CAR = 'Y'                                  as owns_car,
    FLAG_OWN_REALTY = 'Y'                               as owns_realty,
    CNT_CHILDREN                                        as children_cnt,
    CNT_FAM_MEMBERS                                     as family_members_cnt,
    AMT_INCOME_TOTAL                                    as income_amt,
    AMT_CREDIT                                          as credit_amt,
    AMT_ANNUITY                                         as annuity_amt,
    AMT_GOODS_PRICE                                     as goods_price_amt,
    NAME_INCOME_TYPE                                    as income_type,
    NAME_EDUCATION_TYPE                                 as education_type,
    NAME_FAMILY_STATUS                                  as family_status,
    NAME_HOUSING_TYPE                                   as housing_type,
    OCCUPATION_TYPE                                     as occupation_type,
    REGION_RATING_CLIENT                                as region_rating,
    REGION_POPULATION_RELATIVE                          as region_population_rel,
    -DAYS_BIRTH / 365.25                                as age_years,
    -- 365243 is the dataset's placeholder for "not employed / pensioner"
    case when DAYS_EMPLOYED = 365243 then null else -DAYS_EMPLOYED / 365.25 end
                                                        as employment_years,
    DAYS_EMPLOYED = 365243                              as is_unemployed_flag,
    -DAYS_REGISTRATION / 365.25                         as years_since_registration,
    -DAYS_ID_PUBLISH / 365.25                           as years_since_id_change,
    -DAYS_LAST_PHONE_CHANGE                             as days_since_phone_change,
    OWN_CAR_AGE                                         as own_car_age,
    EXT_SOURCE_1                                        as ext_source_1,
    EXT_SOURCE_2                                        as ext_source_2,
    EXT_SOURCE_3                                        as ext_source_3,
    AMT_REQ_CREDIT_BUREAU_YEAR                          as bureau_enquiries_year
from unioned
