{# Read a Home Credit CSV straight from disk, so no separate load step is needed. #}
{% macro raw(name) -%}
    read_csv_auto('{{ var("raw_dir") }}/{{ name }}.csv', header = true, sample_size = -1)
{%- endmacro %}
