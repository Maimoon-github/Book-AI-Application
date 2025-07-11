from django import template

register = template.Library()

@register.filter
def mul(value, arg):
    """Multiply the value by the argument."""
    try:
        return int(value) * int(arg)
    except (ValueError, TypeError):
        return 0

@register.filter  
def add(value, arg):
    """Add the argument to the value."""
    try:
        return int(value) + int(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def intdiv(value, arg):
    """Perform integer division of value by arg."""
    try:
        return int(value) // int(arg)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0
