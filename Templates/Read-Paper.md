---
citekey: "{{citekey}}"
title: "{{title}}"
authors: "{% for t in creators %}{{ t.firstName }} {{ t.lastName }}{{ t.name }}{% if not loop.last %}, {% endif %}{% endfor %}"
date: "{% if date %}{{ date | format("YYYY-MM") }}{% endif %}"
DOI: "{% if DOI %}{{ DOI }}{% endif %}"
tags:
{% if allTags %}- {{ allTags }}{% endif %}
- 文献笔记
期刊: "{% if publicationTitle %}{{ publicationTitle }}{% endif %}{% if university %} {{ university }}{% endif %}"
期刊简称: "{% if journalAbbreviation %}{{ journalAbbreviation }}{% endif %}"
类别: "{{ itemType }}{% if thesisType %} {{ thesisType }}{% endif %}"
languages: "{{ language }}"
url: "{{ url }}"
status: todo
---
# {{title}}

**PDF**: {{pdfLink}} · **Zotero**: {{pdfZoteroLink}} · [在线打开]({{url}})
## Notes

{% persist "research_v1" %}

### 背景与缺口

### 方法与设计

### 结果与意义

### 局限性与可质疑点


{% endpersist %}

## Annotations

{% macro calloutHeader(type, color) -%}
{%- if type == "highlight" -%}
> [!quote]
{%- elif type == "text" -%}
> [!note]
{%- else -%}
> [!quote]
{%- endif -%}
{%- endmacro %}

{% persist "annotations" %}
{% set newAnnotations = annotations | filterby("date", "dateafter", lastImportDate) %}
{% if newAnnotations.length > 0 %}
### Imported: {{importDate | format("YYYY-MM-DD h:mm a")}}

{% for each in newAnnotations %}
{{calloutHeader(each.type, each.color)}}
> {{each.annotatedText}} [jump to](zotero://open-pdf/library/items/{{each.attachment.itemKey}}?page={{each.page}}&annotation={{each.id}})
{% if each.comment %}
标注：{{each.comment}}
{% endif %}{% if each.imageBaseName %}
![[{{each.imageBaseName}}]]
{% endif %}
{% endfor %}
{% endif %}
{% endpersist %}
