from __future__ import annotations

import sys
from typing import Optional
import sympy as sp

from .engine import Engine, Session
import os
import json

try:
    import pyperclip
except Exception:  # optional dependency
    pyperclip = None

PROMPT = ">> "

HELP_TEXT = (
    "Commands:\n"
    "  help                          Show this help.\n"
    "  quit | exit                   Leave the program.\n"
    "  set unicode on|off            Toggle Unicode pretty output.\n"
    "  set digits <n>                Set precision for evalf (default 15).\n"
    "  vars                          List assigned variables.\n"
    "  simplify <expr>               Simplify expression.\n"
    "  expand <expr>                 Expand expression.\n"
    "  factor <expr>                 Factor expression.\n"
    "  apart <expr> [, <var>]        Partial fraction decomposition.\n"
    "  diff <expr>, <var> [, <n>]    Differentiate.\n"
    "  integrate <expr>, <var>       Indefinite integral.\n"
    "  integrate <expr>, <var>, <a>, <b>  Definite integral from a to b.\n"
    "  int <expr>, <var>             Alias for integrate.\n"
    "  int(a,b) <expr>, <var>        Definite integral shorthand.\n"
    "  sum <expr>, <var>, <a>, <b>   Summation (Sigma).\n"
    "  product <expr>, <var>, <a>, <b>  Product (Pi).\n"
    "  solve <expr> [, <var>] [, <a>, <b>]  Solve expr==0; optional interval [a,b].\n"
    "  evalf <expr>                  Numeric evaluation with current digits.\n"
    "  ratsimp <expr>                Rational simplification.\n"
    "  limit <expr>, <var> [, p] [, +|-]   Limit as var->p (defaults to oo).\n"
    "  series <expr>, <var> [, p] [, n]    Series expansion (with remainder).\n"
    "  taylor <expr>, <var> [, p] [, n]    Taylor polynomial (no remainder).\n"
    "  subs <expr>, x=2 [, y=3]      Substitute values.\n"
    "  det <[[..],[..]]>             Determinant of matrix.\n"
    "  inv <[[..],[..]]>             Inverse of matrix.\n"
    "  rref <[[..],[..]]>            Row-reduced echelon form.\n"
    "  linreg <x>, <y>               Linear regression: returns (a, b) for y = a*x + b.\n"
    "  expreg <x>, <y>               Exponential regression: returns (a, b) for y = b*a^x.\n"
    "  powreg <x>, <y>               Power regression: returns (a, b) for y = b*x^a.\n"
    "  assume <sym> <prop>           Assume property for symbol.\n"
    "  forget <sym>                  Clear assumptions for symbol.\n"
    "  save <path>                   Save session to JSON file.\n"
    "  load <path>                   Load session from JSON file.\n"
    "\nFunctions:\n"
    "  def f(x) = <expr>             Define a function. Multiple args allowed.\n"
    "  f(<args>)                     Evaluate a defined function.\n"
    "  dfunc <f> [, <var>] [, <n>]   Create derivative function f_d or f_d2.\n"
    "  critical <f> [, <var>]        Critical points (f' = 0).\n"
    "  extrema <f> [, <var>]         Critical points with min/max/flat.\n"
    "  inflection <f> [, <var>]      Inflection points (f'' = 0).\n"
    "  domain <f> [, <var>]          Continuous domain over Reals.\n"
    "  range <f> [, <var>]           Function range over domain.\n"
    "  solvef <f> [, <var>] [, v] [, a, b]  Solve f(var)=v; optional interval.\n"
    "  tangent <f>, <point> [, <var>]  Tangent line at a point.\n"
    "  normal <f>, <point> [, <var>]   Normal line at a point.\n"
    "  arclength <f>, <var>, <a>, <b>  Arc length of curve on [a,b].\n"
    "  volume disk <f|expr>, <var>, <a>, <b>   Volume via disk method (rotate about x-axis).\n"
    "  volume washer <f>, <g>, <var>, <a>, <b>  Volume via washer method.\n"
    "  volume shell <f|expr>, <var>, <a>, <b>  Volume via shell method (rotate about y-axis).\n"
    "  avgval <f|expr>, <var>, <a>, <b>  Average value on [a,b].\n"
    "  table <f|expr>, <var>, <a>, <b> [, <step>]  Table of values.\n"
    "  compose <f>, <g>              Compose f(g(x)), creates f_g.\n"
    "  funcs                         List defined functions.\n"
    "  delfunc <f>                   Delete a function.\n"
    "  clearfuncs                    Delete all functions.\n"
    "\nPlotting:\n"
    "  plot <f> [, <var>] [, <xmin>] [, <xmax>] [, <width>] [, <height>]   ASCII plot of function.\n"
    "  plot <expr>, <var> [, <xmin>] [, <xmax>] [, <width>] [, <height>]    ASCII plot of expression.\n"
    "  asciplot <...>                Alias for plot command.\n"
    "\nUtilities:\n"
    "  cpresult                      Copy last result to clipboard.\n"
    "  cpopresult                    Copy last input and result to clipboard.\n"
    "  last                          Show last input and result.\n"
    "  history                       Show entered lines this session.\n"
    "  del <var>                     Delete a variable binding.\n"
    "  clearvars                     Clear all variables.\n"
    "  restart                       Reset session (variables and options).\n"
    "  cls                           Clear the screen.\n"
    "  <name> = <expr>               Assign a variable.\n"
    "  <expr>                        Evaluate and display expression.\n"
)


def print_out(text: str = "") -> None:
    # Single place to print output; could be adapted later for logging or file.
    sys.stdout.write(text + ("" if text.endswith("\n") else "\n"))
    sys.stdout.flush()


def main(argv: Optional[list[str]] = None) -> int:
    session = Session()
    eng = Engine(session)
    history: list[str] = []
    last_input: str | None = None
    last_result: str | None = None

    print_out("Accessible CAS (acasm) — type 'help' for commands. Type 'quit' to exit.")

    while True:
        try:
            print_out("")
            sys.stdout.write(PROMPT)
            sys.stdout.flush()
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            history.append(line)

            # Basic commands
            if line.lower() in {"quit", "exit"}:
                break
            if line.lower() == "help":
                print_out(HELP_TEXT)
                continue
            if line.lower().startswith("set "):
                handle_set(line, session)
                continue
            if line.lower() == "vars":
                if not session.values:
                    print_out("No variables assigned.")
                else:
                    for k, v in session.values.items():
                        if isinstance(v, list):
                            # Show lists in a compact Python-like form
                            try:
                                disp = (
                                    "["
                                    + ", ".join(str(eng.format(vi)) for vi in v)
                                    + "]"
                                )
                            except Exception:
                                disp = str(v)
                            print_out(f"{k} = {disp}")
                        else:
                            print_out(f"{k} = {eng.format(v)}")
                continue
            if line.lower() == "funcs":
                funcs = eng.list_functions()
                if not funcs:
                    print_out("No functions defined.")
                else:
                    for name, lam in funcs.items():
                        # Show a prime alias for derivative names
                        prime_alias = None
                        if name.endswith("_d"):
                            base = name[:-2]
                            prime_alias = f"{base}'"
                        elif "_d" in name:
                            base, _, order = name.partition("_d")
                            if order.isdigit():
                                prime_alias = base + ("'" * int(order))
                        disp_name = f"{name} ({prime_alias})" if prime_alias else name
                        args = (
                            lam.variables
                            if isinstance(lam.variables, tuple)
                            else (lam.variables,)
                        )
                        arglist = ", ".join(str(a) for a in args)
                        print_out(f"{disp_name}({arglist}) = {eng.format(lam.expr)}")
                continue

            # Utilities that don't require evaluation
            if line.lower() == "cls":
                clear_screen()
                continue
            if line.lower() == "history":
                if not history:
                    print_out("No history.")
                else:
                    for idx, h in enumerate(history, 1):
                        print_out(f"{idx}: {h}")
                continue
            if line.lower() == "last":
                if last_input is None:
                    print_out("No last input.")
                else:
                    print_out(f"in:  {last_input}")
                    print_out(f"out: {last_result if last_result is not None else ''}")
                continue
            if line.lower().startswith("del "):
                name = line[4:].strip()
                if eng.delete_var(name):
                    print_out(f"Deleted {name}.")
                else:
                    print_out(f"No variable named {name}.")
                continue
            if line.lower() == "clearvars":
                eng.clear_vars()
                print_out("Cleared all variables.")
                continue
            if line.lower() == "restart":
                session = Session()
                eng = Engine(session)
                history.clear()
                last_input = None
                last_result = None
                print_out("Session restarted.")
                continue
            if line.lower().startswith("delfunc "):
                fname = line[len("delfunc ") :].strip()
                if eng.delete_function(fname):
                    print_out(f"Deleted function {fname}.")
                else:
                    print_out(f"No function named {fname}.")
                continue
            if line.lower() == "clearfuncs":
                eng.clear_functions()
                print_out("Cleared all functions.")
                continue
            if line.lower() == "cpresult":
                if last_result is None:
                    print_out("No result to copy.")
                else:
                    ok, msg = copy_to_clipboard(last_result)
                    print_out(msg)
                continue
            if line.lower() == "cpopresult":
                if last_input is None:
                    print_out("No last input to copy.")
                else:
                    payload = f"{last_input} -> {last_result if last_result is not None else ''}"
                    ok, msg = copy_to_clipboard(payload)
                    print_out(msg)
                continue

            # Assignment: name = expr
            if "=" in line and not any(
                line.lower().startswith(cmd)
                for cmd in [
                    "simplify",
                    "expand",
                    "factor",
                    "apart",
                    "diff",
                    "integrate",
                    "int",
                    "int(",
                    "sum",
                    "product",
                    "solve",
                    "tangent",
                    "normal",
                    "arclength",
                    "avgval",
                    "table",
                    "compose",
                    "taylor",
                    "volume",
                ]
            ):
                # Function definition form: def f(x, y) = <expr>
                if line.lower().startswith("def ") and "(" in line and ")" in line:
                    header, body = split_once(line[4:], "=")
                    header = header.strip()
                    body = body.strip()
                    try:
                        fname_part, args_part = header.split("(", 1)
                        fname = fname_part.strip()
                        args_str = args_part.rsplit(")", 1)[0]
                        arg_names = [
                            a.strip() for a in args_str.split(",") if a.strip()
                        ]
                        gname, lam = eng.define_function(fname, arg_names, body)
                        res = (
                            f"{gname}({', '.join(arg_names)}) = {eng.format(lam.expr)}"
                        )
                        print_out(res)
                        last_input = line
                        last_result = res
                    except Exception as e:
                        print_out(f"Error: {e}")
                    continue
                # Variable assignment
                name, expr_text = split_once(line, "=")
                name = name.strip()
                # Normalize prime notation in the name (e.g., f' -> f_d, f'' -> f_d2)
                try:
                    name = eng.resolve_function_name(name)
                except Exception:
                    # If it's not a function name with primes, leave as is for variable assignment
                    pass
                expr_text = expr_text.strip()
                if not name.isidentifier():
                    print_out("Error: Left side of assignment must be a valid name.")
                    continue
                try:
                    _, expr = eng.assign(name, expr_text)
                    formatted = eng.format(expr)
                    print_out(f"{name} = {formatted}")
                    last_input = line
                    last_result = formatted
                except Exception as e:
                    print_out(f"Error: {e}")
                continue

            # Operation commands
            if line.lower().startswith("simplify "):
                arg = line[len("simplify ") :]
                try:
                    res = eng.format(eng.simplify(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("expand "):
                arg = line[len("expand ") :]
                try:
                    res = eng.format(eng.expand(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("factor "):
                arg = line[len("factor ") :]
                try:
                    res = eng.format(eng.factor(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("apart "):
                arg = line[len("apart ") :]
                try:
                    res = eng.format(eng.apart(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("diff "):
                arg = line[len("diff ") :]
                try:
                    res = eng.format(eng.diff(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if (
                line.lower().startswith("integrate ")
                or line.lower().startswith("int ")
                or _is_int_shorthand(line)
            ):
                # Handle 'int' alias and 'int(a,b) expr, var' shorthand
                if _is_int_shorthand(line):
                    arg = _parse_int_shorthand(line)
                elif line.lower().startswith("integrate "):
                    arg = line[len("integrate ") :]
                else:
                    arg = line[len("int ") :]
                try:
                    res = eng.format(eng.integrate(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("sum "):
                arg = line[len("sum ") :]
                try:
                    res = eng.format(eng.summation(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("product "):
                arg = line[len("product ") :]
                try:
                    res = eng.format(eng.product(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("solve "):
                arg = line[len("solve ") :]
                try:
                    res = eng.format(eng.solve(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("dfunc "):
                arg = line[len("dfunc ") :]
                parts = [p.strip() for p in arg.split(",")]
                try:
                    fname_in = parts[0]
                    fname = (
                        eng.resolve_function_name(fname_in) if fname_in else fname_in
                    )
                    var = parts[1] if len(parts) >= 2 and parts[1] else None
                    n = int(parts[2]) if len(parts) >= 3 and parts[2] else 1
                    gname, glam = eng.derivative_func(fname, var, n)
                    args = (
                        glam.variables
                        if isinstance(glam.variables, tuple)
                        else (glam.variables,)
                    )
                    arglist = ", ".join(str(a) for a in args)
                    res = (
                        f"{gname} defined: {gname}({arglist}) = {eng.format(glam.expr)}"
                    )
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("critical "):
                arg = line[len("critical ") :]
                parts = [p.strip() for p in arg.split(",")]
                try:
                    fname = eng.resolve_function_name(parts[0])
                    var = parts[1] if len(parts) >= 2 and parts[1] else None
                    pts = eng.critical(fname, var)
                    if not pts:
                        res = "No critical points"
                    else:
                        res = eng.format(sp.FiniteSet(*pts))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("extrema "):
                arg = line[len("extrema ") :]
                parts = [p.strip() for p in arg.split(",")]
                try:
                    fname = eng.resolve_function_name(parts[0])
                    var = parts[1] if len(parts) >= 2 and parts[1] else None
                    data = eng.extrema(fname, var)
                    if not data:
                        res = "No extrema"
                    else:
                        lam = eng.session.functions.get(fname)
                        args = (
                            lam.variables
                            if isinstance(lam.variables, tuple)
                            else (lam.variables,)
                        )
                        xvar = eng.session.get_symbol(args[0].name)
                        items = []
                        counts = {
                            "local min": 0,
                            "local max": 0,
                            "global min": 0,
                            "global max": 0,
                            "flat": 0,
                            "unknown": 0,
                        }
                        for p, kind in data:
                            y = lam.expr.subs(xvar, p)
                            items.append(
                                f"x={eng.format(p)}, y={eng.format(y)} -> {kind}"
                            )
                            counts[kind] = counts.get(kind, 0) + 1
                        res = "; ".join(items)
                        summary = ", ".join(f"{k}:{v}" for k, v in counts.items() if v)
                        if summary:
                            res = res + f"\ncounts: {summary}"
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("inflection "):
                arg = line[len("inflection ") :]
                parts = [p.strip() for p in arg.split(",")]
                try:
                    fname = eng.resolve_function_name(parts[0])
                    var = parts[1] if len(parts) >= 2 and parts[1] else None
                    pts = eng.inflection(fname, var)
                    if not pts:
                        res = "No inflection points"
                    else:
                        lam = eng.session.functions.get(fname)
                        args = (
                            lam.variables
                            if isinstance(lam.variables, tuple)
                            else (lam.variables,)
                        )
                        xvar = eng.session.get_symbol(args[0].name)
                        items = []
                        for p in pts:
                            y = lam.expr.subs(xvar, p)
                            items.append(f"x={eng.format(p)}, y={eng.format(y)}")
                        res = "; ".join(items)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("domain "):
                arg = line[len("domain ") :]
                parts = [p.strip() for p in arg.split(",")]
                try:
                    fname = eng.resolve_function_name(parts[0])
                    var = parts[1] if len(parts) >= 2 and parts[1] else None
                    dom = eng.domain(fname, var)
                    res = (
                        str(sp.Interval(-sp.oo, sp.oo))
                        if dom == sp.S.Reals
                        else str(dom)
                    )
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("range "):
                arg = line[len("range ") :]
                parts = [p.strip() for p in arg.split(",")]
                try:
                    fname = eng.resolve_function_name(parts[0])
                    var = parts[1] if len(parts) >= 2 and parts[1] else None
                    ran = eng.range(fname, var)
                    res = str(ran)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("solvef "):
                arg = line[len("solvef ") :]
                parts = [p.strip() for p in arg.split(",")]
                try:
                    # Accept forms: "f, x, 0" or equation-like "f'(x) = 0" / "f_d(x) = 0"
                    lo = None
                    hi = None
                    if "=" in arg:
                        left, right = arg.split("=", 1)
                        left = left.strip()
                        right = right.strip()
                        name_part = left.split("(", 1)[0].strip()
                        fname = eng.resolve_function_name(name_part)
                        var = None
                        if "(" in left and ")" in left:
                            inside = left[left.find("(") + 1 : left.rfind(")")].strip()
                            if inside:
                                var = inside
                        # right side may be: "0" or "0, 1, 5" (value, lo, hi)
                        rparts = [p.strip() for p in right.split(",")]
                        value = rparts[0] if rparts[0] else None
                        if len(rparts) >= 3:
                            lo = rparts[1]
                            hi = rparts[2]
                    else:
                        fname = eng.resolve_function_name(parts[0])
                        var = parts[1] if len(parts) >= 2 and parts[1] else None
                        value = parts[2] if len(parts) >= 3 and parts[2] else None
                        if len(parts) >= 5:
                            lo = parts[3]
                            hi = parts[4]
                    sols = eng.solve_function(fname, var, value, lo, hi)
                    res = eng.format(sp.FiniteSet(*sols)) if sols else "{}"
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("tangent "):
                arg = line[len("tangent ") :]
                try:
                    slope, line_expr = eng.tangent_line(arg)
                    res = f"slope = {eng.format(slope)}\ny = {eng.format(line_expr)}"
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("normal "):
                arg = line[len("normal ") :]
                try:
                    slope, line_expr = eng.normal_line(arg)
                    if slope == sp.zoo:
                        parts_n = [p.strip() for p in arg.split(",")]
                        res = f"Normal line is vertical: x = {parts_n[1] if len(parts_n) >= 2 else '?'}"
                    else:
                        res = (
                            f"slope = {eng.format(slope)}\ny = {eng.format(line_expr)}"
                        )
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("arclength "):
                arg = line[len("arclength ") :]
                try:
                    res = eng.format(eng.arclength(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("volume "):
                arg = line[len("volume ") :]
                try:
                    res = eng.format(eng.volume(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("avgval "):
                arg = line[len("avgval ") :]
                try:
                    res = eng.format(eng.avgval(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("table "):
                arg = line[len("table ") :]
                try:
                    rows = eng.table(arg)
                    lines_out = []
                    for xv, yv in rows:
                        lines_out.append(f"  {eng.format(xv)}  |  {eng.format(yv)}")
                    res = "\n".join(lines_out)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("compose "):
                arg = line[len("compose ") :]
                try:
                    new_name, new_lam = eng.compose(arg)
                    var = (
                        new_lam.variables
                        if not isinstance(new_lam.variables, tuple)
                        else new_lam.variables[0]
                    )
                    res = f"{new_name}({var}) = {eng.format(new_lam.expr)}"
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("evalf "):
                arg = line[len("evalf ") :]
                try:
                    res = eng.format(eng.evalf(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("ratsimp "):
                arg = line[len("ratsimp ") :]
                try:
                    res = eng.format(eng.ratsimp(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("limit "):
                arg = line[len("limit ") :]
                try:
                    res = eng.format(eng.limit(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("series "):
                arg = line[len("series ") :]
                try:
                    res = str(eng.series(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("taylor "):
                arg = line[len("taylor ") :]
                try:
                    res = eng.format(eng.taylor(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("subs "):
                arg = line[len("subs ") :]
                try:
                    res = eng.format(eng.subs(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("det "):
                arg = line[len("det ") :]
                try:
                    res = eng.format(eng.det(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("linreg "):
                arg = line[len("linreg ") :]
                try:
                    fname, fexpr, (a, b), r2 = eng.linreg(arg)
                    res = (
                        f"y = {eng.format(fexpr)}\n"
                        f"fitted as {fname}(x)\n"
                        f"a = {eng.format(a)}, b = {eng.format(b)}\n"
                        f"R^2 = {eng.format(r2)}"
                    )
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("expreg "):
                arg = line[len("expreg ") :]
                try:
                    fname, fexpr, (a, b), r2 = eng.expreg(arg)
                    res = (
                        f"y = {eng.format(fexpr)}\n"
                        f"fitted as {fname}(x)\n"
                        f"a = {eng.format(a)}, b = {eng.format(b)}\n"
                        f"R^2 = {eng.format(r2)}"
                    )
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("powreg "):
                arg = line[len("powreg ") :]
                try:
                    fname, fexpr, (a, b), r2 = eng.powreg(arg)
                    res = (
                        f"y = {eng.format(fexpr)}\n"
                        f"fitted as {fname}(x)\n"
                        f"a = {eng.format(a)}, b = {eng.format(b)}\n"
                        f"R^2 = {eng.format(r2)}"
                    )
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("inv "):
                arg = line[len("inv ") :]
                try:
                    res = str(eng.inv(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("rref "):
                arg = line[len("rref ") :]
                try:
                    res = str(eng.rref(arg))
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("assume "):
                arg = line[len("assume ") :]
                try:
                    res = eng.assume(arg)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("forget "):
                arg = line[len("forget ") :]
                try:
                    res = eng.forget(arg)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("save "):
                arg = line[len("save ") :].strip()
                try:
                    res = eng.save(arg)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("load "):
                arg = line[len("load ") :].strip()
                try:
                    res = eng.load(arg)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue
            if line.lower().startswith("plot ") or line.lower().startswith("asciplot "):
                if line.lower().startswith("plot "):
                    arg = line[len("plot ") :]
                else:
                    arg = line[len("asciplot ") :]
                try:
                    res = eng.plot(arg)
                    print_out(res)
                    last_input = line
                    last_result = res
                except Exception as e:
                    print_out(f"Error: {e}")
                continue

            # Otherwise, evaluate expression
            try:
                # If this is a call to a user-defined function, parsing will resolve it via local_dict
                expr = eng.parse(line)
                formatted = eng.format(expr)
                print_out(formatted)
                last_input = line
                last_result = formatted
            except Exception as e:
                # Unknown command or parse error. Offer a hint if it looks like a command.
                low = line.lower()
                known = [
                    "help",
                    "quit",
                    "exit",
                    "set ",
                    "vars",
                    "funcs",
                    "cls",
                    "history",
                    "last",
                    "del ",
                    "clearvars",
                    "restart",
                    "delfunc ",
                    "clearfuncs",
                    "cpresult",
                    "cpopresult",
                    "simplify ",
                    "expand ",
                    "factor ",
                    "apart ",
                    "diff ",
                    "integrate ",
                    "int ",
                    "int(",
                    "sum ",
                    "product ",
                    "solve ",
                    "dfunc ",
                    "critical ",
                    "extrema ",
                    "inflection ",
                    "domain ",
                    "range ",
                    "solvef ",
                    "tangent ",
                    "normal ",
                    "arclength ",
                    "volume ",
                    "avgval ",
                    "table ",
                    "compose ",
                    "evalf ",
                    "ratsimp ",
                    "limit ",
                    "series ",
                    "taylor ",
                    "subs ",
                    "det ",
                    "inv ",
                    "rref ",
                    "assume ",
                    "forget ",
                    "save ",
                    "load ",
                    "linreg ",
                    "expreg ",
                    "powreg ",
                    "plot ",
                    "asciplot ",
                ]
                msg = str(e)
                if any(low.startswith(pfx) for pfx in known):
                    print_out(f"Error: {msg}")
                else:
                    print_out("Unknown command. Type 'help' for a list of commands.")

        except KeyboardInterrupt:
            print_out("\nInterrupted. Type 'quit' to exit.")
        except EOFError:
            break

    print_out("Goodbye.")
    return 0


def split_once(text: str, sep: str):
    idx = text.find(sep)
    if idx == -1:
        return text, ""
    return text[:idx], text[idx + len(sep) :]


def handle_set(line: str, session: Session) -> None:
    low = line.lower()
    if low.startswith("set unicode "):
        tail = low[len("set unicode ") :].strip()
        if tail in {"on", "off"}:
            session.options.unicode = tail == "on"
            mode = "on" if session.options.unicode else "off"
            print_out(f"Unicode pretty output: {mode}")
            return
        print_out("Usage: set unicode on|off")
        return
    if low.startswith("set digits "):
        tail = low[len("set digits ") :].strip()
        try:
            n = int(tail)
            if n <= 0:
                raise ValueError
            session.options.digits = n
            print_out(f"digits = {n}")
        except Exception:
            print_out("Usage: set digits <n> (n >= 1)")
        return
    print_out("Unknown setting. Try: set unicode on|off")


import re as _re


def _is_int_shorthand(line: str) -> bool:
    """Detect int(a,b) expr, var  shorthand syntax."""
    return bool(_re.match(r"(?i)^int\s*\(", line))


def _parse_int_shorthand(line: str) -> str:
    """Convert  int(a,b) expr, var  ->  'expr, var, a, b' for engine.integrate."""
    m = _re.match(r"(?i)^int\s*\(([^)]*)\)\s*(.*)", line)
    if not m:
        raise ValueError("Usage: int(a,b) <expr>, <var>")
    bounds_str = m.group(1).strip()
    rest = m.group(2).strip()
    bounds = [b.strip() for b in bounds_str.split(",")]
    if len(bounds) != 2 or not bounds[0] or not bounds[1]:
        raise ValueError("Usage: int(a,b) <expr>, <var> — need exactly two bounds")
    # rest should be: expr, var
    return f"{rest}, {bounds[0]}, {bounds[1]}"


if __name__ == "__main__":
    raise SystemExit(main())


def clear_screen():
    # Cross-platform clear; on Windows 'cls', on others 'clear'.
    cmd = "cls" if os.name == "nt" else "clear"
    try:
        os.system(cmd)
    except Exception:
        pass


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    # Try pyperclip first
    if pyperclip is not None:
        try:
            pyperclip.copy(text)
            return True, "Copied to clipboard."
        except Exception as e:
            # Fall through to OS-specific methods
            pass
    # Windows fallback to 'clip'
    if os.name == "nt":
        try:
            import subprocess

            p = subprocess.run(["clip"], input=text, text=True, capture_output=True)
            if p.returncode == 0:
                return True, "Copied to clipboard."
        except Exception:
            pass
    return False, "Clipboard not available. Install 'pyperclip' to enable this feature."
