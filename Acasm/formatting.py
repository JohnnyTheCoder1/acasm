from sympy import sstr
from sympy.printing.pretty.pretty import pretty as sympy_pretty

from .options import Options


def format_expr(expr, opts: Options) -> str:
    """
    Return a string suitable for screen readers by default.
    - ASCII (sstr) by default
    - Optional Unicode pretty printing when opts.unicode is True
    """
    if expr is None:
        return ""
    if opts.unicode:
        try:
            # width=0 disables line wrapping
            return sympy_pretty(expr, use_unicode=True, wrap_line=False)
        except Exception:
            # Fallback to sstr
            return sstr(expr)
    return sstr(expr)
