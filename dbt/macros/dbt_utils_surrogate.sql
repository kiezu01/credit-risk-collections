{# Small stand-in for dbt_utils.generate_surrogate_key, to avoid a package dependency. #}
{% macro dbt_utils_surrogate(cols) -%}
    md5(concat_ws('|', {% for c in cols %}cast({{ c }} as varchar){% if not loop.last %}, {% endif %}{% endfor %}))
{%- endmacro %}
