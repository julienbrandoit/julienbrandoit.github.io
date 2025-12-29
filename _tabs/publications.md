---
layout: page
icon: fa fa-book
order: 1
---

Here are my research publications, preprints and presentations:

## Publications

{% assign pub_posts = site.posts | where: "categories", "Publications" %}
{% for post in pub_posts %}
- [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %Y" }}
{% endfor %}

## Presentations

{% assign pres_posts = site.posts | where: "categories", "Presentations" %}
{% for post in pres_posts %}
- [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %Y" }}
{% endfor %}