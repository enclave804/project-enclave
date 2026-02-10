Subject: Saw {{ company_name }}'s new project — need a model?
---
{{ contact_first_name }},

{% if trigger_signal == "design_competition" %}
I noticed {{ company_name }} is entering the {{ trigger_detail }} competition. Competition juries consistently favor submissions with physical models — they create a tactile connection that renders can't match.
{% elif trigger_signal == "new_building_project" %}
Congrats on the {{ trigger_detail }} project! For a build of that scope, having a physical model for stakeholder presentations can make a real difference in approvals.
{% elif trigger_signal == "expanding_offices" %}
I saw that {{ company_name }} is expanding — exciting times! As you pitch to new clients in new markets, presentation models make a strong first impression.
{% else %}
I've been following {{ company_name }}'s recent work and thought this might be good timing to connect.
{% endif %}

We specialize in printing architectural models for firms like yours:

- **Any scale** — 1:500 site models to 1:20 detail sections
- **48-hour turnaround** on standard orders
- **Professional finishing** — your model arrives presentation-ready

Happy to send a **free sample** or portfolio from similar projects. Interested?

Best,
Jose
PrintBiz — CAD to Reality in 48 Hours

{{ physical_address }}
[Unsubscribe]({{ unsubscribe_url }})
