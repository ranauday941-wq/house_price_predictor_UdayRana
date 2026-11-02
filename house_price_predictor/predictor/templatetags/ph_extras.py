from django import template

register = template.Library()


@register.filter
def money(value):
    try:
        return format(float(value), ",.0f")
    except (TypeError, ValueError):
        return value


@register.filter
def pct(value):
    """0.9112 -> 91.1"""
    try:
        return format(float(value) * 100, ".1f")
    except (TypeError, ValueError):
        return value


@register.filter
def bar(value):
    """|correlation| 0.65 -> 65 (width of a CSS bar in percent)"""
    try:
        return int(min(abs(float(value)), 1.0) * 100)
    except (TypeError, ValueError):
        return 0
