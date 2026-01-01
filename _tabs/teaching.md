---
layout: page
icon: fas fa-chalkboard-teacher
order: 5
---

Below are the courses I am currently teaching ! 

If you have any question about the courses, please feel free to contact me ([julien.brandoit@uliege.be](mailto:julien.brandoit@uliege.be)), the Professor or other staff members. Feel free to come to my office (B28 [Montefiore], I.140 -- my name is on the door) as well ! We can always plan a meeting if you want to be sure that I am available; however, spontaneous visits are also welcome.

## Courses (2025 - 2026)

<ul>
{% for course in site.data.courses %}
  <li>
    <a href="{{ course.url }}">{{ course.title }} -- {{ course.code }}</a>
  </li>
{% endfor %}
</ul>
