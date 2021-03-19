{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :members:
    :inherited-members:

    {% block methods %}

    {% if methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
    {% for item in methods %}
        {% if not item[0] == '_' %}~{{ name }}.{{ item }}{%endif %}
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
