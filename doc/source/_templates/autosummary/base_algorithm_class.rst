{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :private-members:


    {% block methods %}

    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
    {% for item in members %}
        {% if not item[0:2] == '__' and item not in attributes and item not in inherited_members%}~{{ name }}.{{ item }} {% endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
    {% for item in attributes %}
        {% if not item[0] == '_' %}~{{ name }}.{{ item }}{%endif %}
    {%- endfor %}
    {% endif %}
    {% endblock %}
