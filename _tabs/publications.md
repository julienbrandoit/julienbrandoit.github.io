---
layout: page
icon: fa fa-book
order: 1
---

Here are my research publications and preprints:

{% assign pub_posts = site.posts | where: "categories", "Publications" %}
{% for post in pub_posts %}
- [{{ post.title }}]({{ post.url }}) - {{ post.date | date: "%B %Y" }}
{% endfor %}
