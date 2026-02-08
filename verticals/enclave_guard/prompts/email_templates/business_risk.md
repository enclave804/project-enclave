Subject: {{ company_name }} — a security question for you
---
{{ contact_first_name }},

I've been researching {{ company_industry }} companies in the {{ company_size }}-employee range, and {{ company_name }} stood out.

{% if vulnerabilities %}
Looking at {{ company_domain }}, I noticed a few areas where your public-facing infrastructure could be tightened:

{% for vuln in vulnerabilities[:2] %}
- {{ vuln.description }}
{% endfor %}

These findings are public — meaning anyone with the right tools can see them.
{% else %}
Companies at {{ company_name }}'s stage often have security gaps that don't surface until a prospect asks for a SOC2 report or a breach makes the news.
{% endif %}

The business risk is straightforward: a single security incident in {{ company_industry }} averages 6-figure remediation costs, and the reputational damage can stall growth for quarters.

We help companies like {{ company_name }} identify and close these gaps before they become expensive problems — typically in a focused 2-week engagement.

Would a 15-minute call this week make sense to see if this is relevant?

Best,
Jose
Enclave Guard

{{ physical_address }}
[Unsubscribe]({{ unsubscribe_url }})
