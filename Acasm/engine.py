from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Sequence
import ast
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from .options import Options, DEFAULT_OPTIONS
from .formatting import format_expr


SAFE_FUNCTIONS = {
    # Basic arithmetic and functions
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    # Additional trig/hyperbolic and aliases
    "sec": sp.sec,
    "csc": sp.csc,
    "cot": sp.cot,
    "sech": sp.sech,
    "csch": sp.csch,
    "coth": sp.coth,
    # Exponential/log/aliases
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    # Rounding and sign
    "floor": sp.floor,
    "ceiling": sp.ceiling,
    "sign": sp.sign,
    # Piecewise-like extrema
    "Max": sp.Max,
    "Min": sp.Min,
    # Combinatorics/Special
    "factorial": sp.factorial,
    "binomial": sp.binomial,
    "gamma": sp.gamma,
    # Complex helpers
    "re": sp.re,
    "im": sp.im,
    "arg": sp.arg,
    # Absolute value
    "abs": sp.Abs,
    # Constants and aliases
    "pi": sp.pi,
    "Pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "I": sp.I,
    "EulerGamma": sp.EulerGamma,
    "Catalan": sp.Catalan,
    "GoldenRatio": sp.GoldenRatio,
    "phi": sp.GoldenRatio,
    # Infinities and NaNs
    "oo": sp.oo,
    "inf": sp.oo,
    "infinity": sp.oo,
    "zoo": sp.zoo,
    "nan": sp.nan,
}


@dataclass
class Session:
    symbols: Dict[str, sp.Symbol] = field(default_factory=dict)
    # Allow values to be SymPy expressions or Python lists for data sequences
    values: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, sp.Lambda] = field(default_factory=dict)
    options: Options = field(
        default_factory=lambda: Options(**DEFAULT_OPTIONS.__dict__)
    )

    def get_symbol(self, name: str) -> sp.Symbol:
        if name not in self.symbols:
            self.symbols[name] = sp.Symbol(name)
        return self.symbols[name]

    def local_dict(self) -> Dict[str, Any]:
        # Expose known symbols, assigned values, and user-defined functions to the parser
        env: Dict[str, Any] = dict(SAFE_FUNCTIONS)
        env.update(self.symbols)
        env.update(self.values)
        env.update(self.functions)
        return env


class Engine:
    def __init__(self, session: Session | None = None):
        self.session = session or Session()

    # ---- Preprocessing helpers ----

    @staticmethod
    def _normalize_commas(text: str) -> str:
        """Convert European-style comma decimals to dots.

        Rules:
        - A comma directly between two digits (e.g. 1,5) is treated as a
          decimal separator and replaced with a dot (1.5).
        - Commas preceded or followed by a space are left alone (argument
          separators).
        """
        return re.sub(r"(\d),(\d)", r"\1.\2", text)

    # Parsing and evaluation
    def parse(self, text: str) -> sp.Expr:
        text = text.strip()
        if not text:
            raise ValueError("Empty input")
        # Convert comma-decimals to dot-decimals
        text = self._normalize_commas(text)
        # Replace caret with ** for exponentiation
        text = text.replace("^", "**")
        try:
            # Enable implicit multiplication like 2x, (x+1)(x-1), 2 sin(x)
            transformations = standard_transformations + (
                implicit_multiplication_application,
            )
            expr = parse_expr(
                text,
                local_dict=self.session.local_dict(),
                transformations=transformations,
                evaluate=True,
            )
            return expr
        except Exception as e:
            raise ValueError(f"Could not parse expression: {e}")

    # Helper: split arguments respecting parenthesis nesting
    def _smart_split(self, text: str, sep: str = ",") -> List[str]:
        """Split text on *sep* only when not inside parentheses / brackets."""
        parts: List[str] = []
        depth = 0
        current: list[str] = []
        for ch in text:
            if ch in "([{":
                depth += 1
                current.append(ch)
            elif ch in ")]}":
                depth -= 1
                current.append(ch)
            elif ch == sep and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        parts.append("".join(current))
        return parts

    # Helper: inline-expand user-defined function calls in an expression string
    def _expand_function_calls(self, expr_str: str) -> str:
        """Replace occurrences like f(x+1) with the function body substituted."""
        for fname, lam in self.session.functions.items():
            # Match fname( ... ) using a simple balanced-paren scanner
            pattern = re.compile(re.escape(fname) + r"\s*\(")
            while True:
                m = pattern.search(expr_str)
                if not m:
                    break
                start = m.start()
                # Find matching closing paren
                depth = 0
                idx = m.end() - 1  # position of '('
                for i in range(idx, len(expr_str)):
                    if expr_str[i] == "(":
                        depth += 1
                    elif expr_str[i] == ")":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                else:
                    break  # unmatched paren, bail
                inner = expr_str[m.end() : end - 1]
                args = self._smart_split(inner)
                if isinstance(lam.variables, tuple):
                    params = lam.variables
                else:
                    params = (lam.variables,)
                if len(args) != len(params):
                    break  # arg count mismatch, leave alone
                body = lam.expr
                for param, arg_text in zip(params, args):
                    try:
                        arg_val = self.parse(arg_text.strip())
                        body = body.subs(param, arg_val)
                    except Exception:
                        break
                else:
                    replacement = f"({body})"
                    expr_str = expr_str[:start] + replacement + expr_str[end:]
                    continue
                break
        return expr_str

    # Helper: handle names with trailing prime characters (e.g., f', f'')
    def _split_prime_suffix(self, name: str) -> tuple[str, int]:
        count = 0
        while name.endswith("'") and name:
            name = name[:-1]
            count += 1
        return name, count

    def resolve_function_name(self, name_or_prime: str) -> str:
        """
        Given a function name that may include prime suffixes, ensure the
        corresponding derivative function exists and return the stored name.
        Examples:
          - 'f'   -> 'f'
          - "f'"  -> 'f_d'
          - "f''" -> 'f_d2'
        """
        base, n = self._split_prime_suffix(name_or_prime.strip())
        if n == 0:
            # Return as-is (may or may not exist yet)
            return base
        # Ensure base exists
        if base not in self.session.functions:
            raise ValueError(f"No function named {base}")
        # Target derived name
        target = f"{base}_d{n if n > 1 else ''}"
        if target not in self.session.functions:
            # Create derivative function of order n
            self.derivative_func(base, None, n)
        return target

    def assign(self, name: str, expr_text: str) -> Tuple[str, sp.Expr]:
        text = expr_text.strip()
        # Treat [ ... ] on the right-hand side as a Python list literal for data
        if text.startswith("[") and text.endswith("]"):
            try:
                data = ast.literal_eval(text)
            except Exception as e:
                raise ValueError(f"Invalid list literal: {e}")
            if not isinstance(data, list):
                raise ValueError(
                    "Only flat lists are supported for sequence assignment"
                )
            # Validate contents and convert to SymPy numbers where possible
            seq: List[sp.Expr] = []
            for i, v in enumerate(data):
                if isinstance(v, (int, float)):
                    seq.append(sp.nsimplify(v))
                elif isinstance(v, (str,)):
                    # Try to parse numeric strings like "1/2" or "3.14"
                    try:
                        seq.append(sp.nsimplify(v))
                    except Exception:
                        raise ValueError(f"List element {i+1} is not numeric: {v}")
                elif isinstance(v, sp.Basic):
                    seq.append(v)
                else:
                    raise ValueError(
                        f"List element {i+1} must be a number, got {type(v).__name__}"
                    )
            self.session.values[name] = seq
            # Provide a symbol as well so name can still be used as a symbol if needed
            self.session.get_symbol(name)
            # Return a SymPy Tuple for formatting purposes
            return name, sp.Tuple(*seq)
        # Otherwise, parse as an expression
        expr = self.parse(text)
        self.session.values[name] = expr
        # Also ensure a symbol exists for that name
        self.session.get_symbol(name)
        return name, expr

    def define_function(
        self, name: str, arg_names: Sequence[str], expr_text: str
    ) -> Tuple[str, sp.Lambda]:
        if not name.isidentifier():
            raise ValueError("Function name must be a valid identifier")
        if not arg_names:
            raise ValueError("Function must have at least one argument")
        args_syms: List[sp.Symbol] = []
        for a in arg_names:
            a = a.strip()
            if not a.isidentifier():
                raise ValueError(f"Invalid parameter name: {a}")
            args_syms.append(self.session.get_symbol(a))
        # Build local env including parameters
        local_env = self.session.local_dict().copy()
        for s in args_syms:
            local_env[s.name] = s
        text = self._normalize_commas(expr_text.strip()).replace("^", "**")
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        try:
            expr = parse_expr(
                text,
                local_dict=local_env,
                transformations=transformations,
                evaluate=True,
            )
        except Exception as e:
            raise ValueError(f"Could not parse function body: {e}")
        lam = sp.Lambda(tuple(args_syms) if len(args_syms) != 1 else args_syms[0], expr)
        self.session.functions[name] = lam
        return name, lam

    # Operations
    def simplify(self, expr_text: str) -> sp.Expr:
        return sp.simplify(self.parse(expr_text))

    def expand(self, expr_text: str) -> sp.Expr:
        return sp.expand(self.parse(expr_text))

    def factor(self, expr_text: str) -> sp.Expr:
        return sp.factor(self.parse(expr_text))

    def diff(self, args_text: str) -> sp.Expr:
        # Expect: "expr, var" or "expr, var, n"
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: diff <expr>, <var> [, <n>]")
        expr = self.parse(parts[0])
        var_name = parts[1]
        var = self.session.get_symbol(var_name)
        n = int(parts[2]) if len(parts) >= 3 else 1
        return sp.diff(expr, var, n)

    def integrate(self, args_text: str) -> sp.Expr:
        """Integrate an expression.
        Indefinite: integrate <expr>, <var>
        Definite:   integrate <expr>, <var>, <a>, <b>
        Shorthand:  int(<a>,<b>) <expr>, <var>   (parsed in REPL)
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: integrate <expr>, <var> [, <a>, <b>]")
        expr = self.parse(parts[0])
        var_name = parts[1]
        var = self.session.get_symbol(var_name)
        if len(parts) >= 4:
            # Definite integral with bounds
            a = self.parse(parts[2])
            b = self.parse(parts[3])
            return sp.integrate(expr, (var, a, b))
        elif len(parts) == 3:
            raise ValueError(
                "Definite integral needs both bounds: integrate <expr>, <var>, <a>, <b>"
            )
        return sp.integrate(expr, var)

    @staticmethod
    def _filter_interval(solutions, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr):
        """Keep only solutions that lie within [lo, hi].

        *solutions* can be a list, FiniteSet, or other SymPy set.
        Returns a plain Python list of solutions inside the interval.
        """
        if isinstance(solutions, (sp.sets.sets.FiniteSet, sp.sets.sets.Set)):
            try:
                solutions = list(solutions)
            except TypeError:
                return solutions  # infinite / non-enumerable set — return as-is
        if not isinstance(solutions, list):
            return solutions
        filtered: List[sp.Expr] = []
        for s in solutions:
            try:
                val = complex(sp.N(s)).real
                if float(sp.N(lo)) <= val <= float(sp.N(hi)):
                    filtered.append(s)
            except Exception:
                filtered.append(s)  # keep if can't evaluate
        return filtered

    def solve(self, args_text: str) -> sp.Expr:
        """
        Solve equations or expressions.
        Usage:
          - solve <expr>                    # interprets as expr == 0
          - solve <expr>, <var>             # solve for a variable
          - solve <expr>, <var>, <a>, <b>   # solve for var in [a, b]

        In <expr>, you can write equality with a single '=' or '==', e.g.:
          solve x^2 = 4, x
          solve x^2 = 4, x, 0, 5    <- only solutions in [0, 5]

        Handles f(x) notation by substituting defined functions, and
        complex expressions involving sqrt, abs, etc.
        """
        # Split only on commas that are NOT inside parentheses
        parts = self._smart_split(args_text)
        if len(parts) == 0 or parts[0] == "":
            raise ValueError("Usage: solve <expr> [, <var>] [, <a>, <b>]")

        expr_str = parts[0].strip()

        # Inline-expand known function calls like f(x) -> expr
        expr_str = self._expand_function_calls(expr_str)

        # Detect equality with single '=' or '==' and build Eq(lhs, rhs)
        eq_obj = None
        if "==" in expr_str:
            lhs_str, rhs_str = expr_str.split("==", 1)
            lhs_str = self._expand_function_calls(lhs_str.strip())
            rhs_str = self._expand_function_calls(rhs_str.strip())
            eq_obj = sp.Eq(self.parse(lhs_str), self.parse(rhs_str))
        elif "=" in expr_str:
            lhs_str, rhs_str = expr_str.split("=", 1)
            lhs_str = self._expand_function_calls(lhs_str.strip())
            rhs_str = self._expand_function_calls(rhs_str.strip())
            eq_obj = sp.Eq(self.parse(lhs_str), self.parse(rhs_str))

        if eq_obj is not None:
            target_expr = eq_obj
        else:
            target_expr = self.parse(expr_str)

        # Detect interval bounds: last two numeric-looking args after the var
        interval = None
        vars_list: List[sp.Symbol] = []
        remaining = [p.strip() for p in parts[1:] if p.strip()]

        if len(remaining) >= 3:
            # form: var, a, b
            vars_list.append(self.session.get_symbol(remaining[0]))
            interval = (self.parse(remaining[1]), self.parse(remaining[2]))
        elif len(remaining) >= 1:
            for v in remaining:
                vars_list.append(self.session.get_symbol(v))

        try:
            if vars_list:
                result = sp.solve(target_expr, *vars_list)
            elif isinstance(target_expr, sp.Equality):
                result = sp.solve(target_expr)
            else:
                free = list(target_expr.free_symbols)
                if not free:
                    result = sp.solve(target_expr)
                else:
                    result = sp.solve(target_expr, *free)
            if interval and isinstance(result, list):
                result = self._filter_interval(result, vars_list[0], *interval)
            return result
        except NotImplementedError:
            # Fallback: try solveset for single-variable cases
            if len(vars_list) == 1:
                try:
                    ss = sp.solveset(target_expr, vars_list[0], domain=sp.S.Reals)
                    if interval:
                        ss = self._filter_interval(ss, vars_list[0], *interval)
                    return ss
                except Exception:
                    pass
            raise
        except Exception:
            # For complex expressions (sqrt, nested, etc.) try simplifying first
            try:
                simplified = sp.simplify(target_expr)
                if vars_list:
                    result = sp.solve(simplified, *vars_list)
                else:
                    result = sp.solve(simplified)
                if interval and isinstance(result, list):
                    result = self._filter_interval(result, vars_list[0], *interval)
                return result
            except Exception:
                raise

    def evalf(self, expr_text: str) -> sp.Expr:
        expr = self.parse(expr_text)
        return expr.evalf(self.session.options.digits)

    def ratsimp(self, expr_text: str) -> sp.Expr:
        return sp.ratsimp(self.parse(expr_text))

    def limit(self, args_text: str) -> sp.Expr:
        # Usage: limit <expr>, <var> [, <point>] [, dir]
        # dir in {+, -} for right/left
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: limit <expr>, <var> [, <point>] [, +|-]")
        expr = self.parse(parts[0])
        var = self.session.get_symbol(parts[1])
        point = sp.oo if len(parts) >= 3 and parts[2] != "" else None
        if len(parts) >= 3 and parts[2] not in {"", "+", "-"}:
            point = self.parse(parts[2])
        dir = None
        if parts[-1] in {"+", "-"}:
            dir = parts[-1]
        limit_point = sp.oo if point is None else point
        if dir is None:
            return sp.limit(expr, var, limit_point)
        return sp.limit(expr, var, limit_point, dir=dir)

    def series(self, args_text: str) -> sp.Expr:
        # Usage: series <expr>, <var> [, <point>] [, <n>]
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: series <expr>, <var> [, <point>] [, <n>]")
        expr = self.parse(parts[0])
        var = self.session.get_symbol(parts[1])
        point = 0
        n = 6
        if len(parts) >= 3 and parts[2]:
            point = self.parse(parts[2])
        if len(parts) >= 4 and parts[3]:
            n = int(parts[3])
        return sp.series(expr, var, point, n)

    def subs(self, args_text: str) -> sp.Expr:
        # Usage: subs <expr>, x=2 [, y=3, ...]
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: subs <expr>, name=value [, ...]")
        expr = self.parse(parts[0])
        repls: List[tuple] = []
        for p in parts[1:]:
            if "=" not in p:
                raise ValueError("Substitution must be name=value")
            name, val = p.split("=", 1)
            name = name.strip()
            val_expr = self.parse(val.strip())
            repls.append((self.session.get_symbol(name), val_expr))
        return expr.subs(repls)

    # --- Data helpers and regressions ---
    def _get_sequence(self, text: str) -> List[sp.Expr]:
        """Resolve a sequence argument which can be a variable name or a list literal."""
        t = text.strip()
        if not t:
            raise ValueError("Missing sequence argument")
        # Inline list literal
        if t.startswith("[") and t.endswith("]"):
            try:
                data = ast.literal_eval(t)
            except Exception as e:
                raise ValueError(f"Invalid list literal: {e}")
            if not isinstance(data, list):
                raise ValueError("Sequence must be a list")
            seq: List[sp.Expr] = []
            for i, v in enumerate(data):
                try:
                    seq.append(sp.nsimplify(v))
                except Exception:
                    raise ValueError(f"List element {i+1} is not numeric: {v}")
            return seq
        # Variable reference
        if t not in self.session.values:
            raise ValueError(f"Unknown variable: {t}")
        val = self.session.values[t]
        if isinstance(val, list):
            return [sp.nsimplify(v) if not isinstance(v, sp.Basic) else v for v in val]
        raise ValueError(f"{t} is not a list; assign with: {t} = [1, 2, ...]")

    def _validate_xy(self, x: List[sp.Expr], y: List[sp.Expr]) -> None:
        if not isinstance(x, list) or not isinstance(y, list):
            raise ValueError("x and y must be lists")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for regression")

    def _linfit(self, x: List[sp.Expr], y: List[sp.Expr]) -> Tuple[sp.Expr, sp.Expr]:
        n = sp.Integer(len(x))
        sx = sum(x)
        sy = sum(y)
        sxx = sum([xi * xi for xi in x])
        sxy = sum([xi * yi for xi, yi in zip(x, y)])
        denom = n * sxx - sx * sx
        if denom == 0:
            raise ValueError("Regression undefined: all x values are identical")
        m = (n * sxy - sx * sy) / denom
        b = (sy - m * sx) / n
        return m, b

    def _r2(self, y: List[sp.Expr], yhat: List[sp.Expr]) -> sp.Expr:
        n = len(y)
        if n == 0:
            return sp.Integer(0)
        ybar = sum(y) / sp.Integer(n)
        ss_res = sum([(yi - hi) ** 2 for yi, hi in zip(y, yhat)])
        ss_tot = sum([(yi - ybar) ** 2 for yi in y])
        if ss_tot == 0:
            # All y equal; perfect fit if residuals are zero
            return sp.Integer(1) if ss_res == 0 else sp.Integer(0)
        return sp.simplify(1 - ss_res / ss_tot)

    def _register_fit_function(
        self, expr: sp.Expr, var_name: str = "x", base: str = "f"
    ) -> Tuple[str, sp.Lambda]:
        x = self.session.get_symbol(var_name)
        lam = sp.Lambda(x, expr)
        # Find an available name: f, f2, f3, ...
        name = base
        if name in self.session.functions:
            i = 2
            while f"{base}{i}" in self.session.functions:
                i += 1
            name = f"{base}{i}"
        self.session.functions[name] = lam
        return name, lam

    def linreg(
        self, args_text: str
    ) -> Tuple[str, sp.Expr, Tuple[sp.Expr, sp.Expr], sp.Expr]:
        # Usage: linreg <x>, <y>
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) != 2:
            raise ValueError("Usage: linreg <x>, <y>")
        x = self._get_sequence(parts[0])
        y = self._get_sequence(parts[1])
        self._validate_xy(x, y)
        # Slope (a) and intercept (b) for y = a*x + b
        a, b = self._linfit(x, y)
        xsym = self.session.get_symbol("x")
        expr = sp.simplify(a * xsym + b)
        name, _ = self._register_fit_function(expr, var_name="x", base="f")
        yhat = [expr.subs(xsym, xi) for xi in x]
        r2 = self._r2(y, yhat)
        return name, expr, (a, b), r2

    def expreg(
        self, args_text: str
    ) -> Tuple[str, sp.Expr, Tuple[sp.Expr, sp.Expr], sp.Expr]:
        # Model: y = b * a^x  (equivalently ln(y) = ln(b) + x*ln(a))
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) != 2:
            raise ValueError("Usage: expreg <x>, <y>")
        x = self._get_sequence(parts[0])
        y = self._get_sequence(parts[1])
        self._validate_xy(x, y)
        ly: List[sp.Expr] = []
        for i, yi in enumerate(y):
            if yi.is_real is False:
                raise ValueError("y values must be positive real for expreg")
            try:
                # Ensure positivity
                if yi <= 0:
                    raise ValueError
            except Exception:
                # Fall back to numerical check
                if sp.N(yi) <= 0:
                    raise ValueError("y values must be positive for expreg")
            ly.append(sp.log(yi))
        # ln y = ln b + (ln a)*x -> slope = ln a, intercept = ln b
        slope, intercept = self._linfit(x, ly)
        a = sp.exp(slope)
        b = sp.exp(intercept)
        xsym = self.session.get_symbol("x")
        expr = sp.simplify(b * (a**xsym))
        name, _ = self._register_fit_function(expr, var_name="x", base="f")
        yhat = [expr.subs(xsym, xi) for xi in x]
        r2 = self._r2(y, yhat)
        return name, expr, (a, b), r2

    def powreg(
        self, args_text: str
    ) -> Tuple[str, sp.Expr, Tuple[sp.Expr, sp.Expr], sp.Expr]:
        # Model: y = b * x^a  (equivalently ln y = ln b + a*ln x)
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) != 2:
            raise ValueError("Usage: powreg <x>, <y>")
        x = self._get_sequence(parts[0])
        y = self._get_sequence(parts[1])
        self._validate_xy(x, y)
        lx: List[sp.Expr] = []
        ly: List[sp.Expr] = []
        for i, (xi, yi) in enumerate(zip(x, y)):
            try:
                if xi <= 0 or yi <= 0:
                    raise ValueError
            except Exception:
                if sp.N(xi) <= 0 or sp.N(yi) <= 0:
                    raise ValueError("x and y must be positive for powreg")
            lx.append(sp.log(xi))
            ly.append(sp.log(yi))
        # ln y = ln b + a*ln x -> slope = a, intercept = ln b
        a, ln_b = self._linfit(lx, ly)
        b = sp.exp(ln_b)
        xsym = self.session.get_symbol("x")
        expr = sp.simplify(b * (xsym**a))
        name, _ = self._register_fit_function(expr, var_name="x", base="f")
        yhat = [expr.subs(xsym, xi) for xi in x]
        r2 = self._r2(y, yhat)
        return name, expr, (a, b), r2

    # Linear algebra helpers
    def matrix(self, text: str) -> sp.Matrix:
        # Accept Python-like list input: [[1,2],[3,4]]
        try:
            data = eval(text, {"__builtins__": {}}, {})  # restricted eval of literals
        except Exception as e:
            raise ValueError(f"Invalid matrix literal: {e}")
        try:
            return sp.Matrix(data)
        except Exception as e:
            raise ValueError(f"Could not build matrix: {e}")

    def det(self, text: str) -> sp.Expr:
        return self.matrix(text).det()

    def inv(self, text: str) -> sp.Matrix:
        return self.matrix(text).inv()

    def rref(self, text: str) -> sp.Matrix:
        M = self.matrix(text)
        R, _ = M.rref()
        return R

    # Assumptions — recreate symbols with SymPy keyword assumptions
    def assume(self, args_text: str) -> str:
        # Usage: assume x real|integer|positive|negative|nonzero
        parts = args_text.split()
        if len(parts) != 2:
            raise ValueError("Usage: assume <symbol> <property>")
        name, prop = parts
        allowed = {
            "real",
            "integer",
            "positive",
            "negative",
            "nonzero",
            "nonnegative",
            "nonpositive",
            "finite",
            "commutative",
            "even",
            "odd",
        }
        if prop not in allowed:
            raise ValueError("Property must be one of: " + ", ".join(sorted(allowed)))
        # Build a new symbol with the assumption kwarg
        kwargs = {prop: True}
        new_sym = sp.Symbol(name, **kwargs)
        old_sym = self.session.symbols.get(name)
        self.session.symbols[name] = new_sym
        # Propagate into values and function bodies that reference this symbol
        if old_sym is not None and old_sym != new_sym:
            for k, v in list(self.session.values.items()):
                if isinstance(v, sp.Basic):
                    self.session.values[k] = v.subs(old_sym, new_sym)
            for k, lam in list(self.session.functions.items()):
                new_expr = lam.expr.subs(old_sym, new_sym)
                if isinstance(lam.variables, tuple):
                    new_vars = tuple(
                        new_sym if v == old_sym else v for v in lam.variables
                    )
                else:
                    new_vars = new_sym if lam.variables == old_sym else lam.variables
                self.session.functions[k] = sp.Lambda(new_vars, new_expr)
        return f"Assumed {name} is {prop}."

    def forget(self, name: str) -> str:
        name = name.strip()
        old_sym = self.session.symbols.get(name)
        # Replace with a plain symbol (no assumptions)
        new_sym = sp.Symbol(name)
        self.session.symbols[name] = new_sym
        if old_sym is not None and old_sym != new_sym:
            for k, v in list(self.session.values.items()):
                if isinstance(v, sp.Basic):
                    self.session.values[k] = v.subs(old_sym, new_sym)
            for k, lam in list(self.session.functions.items()):
                new_expr = lam.expr.subs(old_sym, new_sym)
                if isinstance(lam.variables, tuple):
                    new_vars = tuple(
                        new_sym if v == old_sym else v for v in lam.variables
                    )
                else:
                    new_vars = new_sym if lam.variables == old_sym else lam.variables
                self.session.functions[k] = sp.Lambda(new_vars, new_expr)
        return f"Forgot assumptions about {name}."

    # Session save/load (variables and options)
    def save(self, path: str) -> str:
        import json

        # Values: support lists explicitly for robust reloads
        values_out: Dict[str, Any] = {}
        for k, v in self.session.values.items():
            if isinstance(v, list):
                # Store as plain JSON list of strings for readability
                values_out[k] = [
                    str(sp.nsimplify(e)) if not isinstance(e, sp.Basic) else str(e)
                    for e in v
                ]
            else:
                values_out[k] = str(v)
        data = {
            "values": values_out,
            "functions": {
                k: {
                    "args": [
                        str(a)
                        for a in (
                            v.variables
                            if isinstance(v.variables, tuple)
                            else (v.variables,)
                        )
                    ],
                    "expr": str(v.expr),
                }
                for k, v in self.session.functions.items()
            },
            "options": {
                "unicode": self.session.options.unicode,
                "digits": self.session.options.digits,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return f"Saved session to {path}"

    def load(self, path: str) -> str:
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.session.values.clear()
        self.session.functions.clear()
        for k, v in data.get("values", {}).items():
            try:
                if isinstance(v, list):
                    # Rehydrate list of values
                    seq: List[sp.Expr] = []
                    for e in v:
                        try:
                            seq.append(sp.nsimplify(e))
                        except Exception:
                            seq.append(self.parse(str(e)))
                    self.session.values[k] = seq
                    self.session.get_symbol(k)
                elif (
                    isinstance(v, str)
                    and v.strip().startswith("[")
                    and v.strip().endswith("]")
                ):
                    # Old format serialized as string list; try literal_eval
                    try:
                        data_list = ast.literal_eval(v)
                        if isinstance(data_list, list):
                            seq = [sp.nsimplify(e) for e in data_list]
                            self.session.values[k] = seq
                            self.session.get_symbol(k)
                            continue
                    except Exception:
                        pass
                    # Fallback: try parse as expression
                    self.session.values[k] = self.parse(v)
                    self.session.get_symbol(k)
                else:
                    self.session.values[k] = self.parse(str(v))
                    self.session.get_symbol(k)
            except Exception:
                pass
        for k, fv in data.get("functions", {}).items():
            try:
                args = fv.get("args", [])
                expr = fv.get("expr", "")
                self.define_function(k, args, expr)
            except Exception:
                pass
        opts = data.get("options", {})
        self.session.options.unicode = bool(opts.get("unicode", False))
        self.session.options.digits = int(opts.get("digits", 15))
        return f"Loaded session from {path}"

    # Formatting helpers
    def format(self, expr: sp.Expr) -> str:
        return format_expr(expr, self.session.options)

    # Variable utilities
    def delete_var(self, name: str) -> bool:
        name = name.strip()
        if name in self.session.values:
            del self.session.values[name]
            return True
        return False

    def clear_vars(self) -> None:
        self.session.values.clear()

    # Functions utilities and analysis
    def delete_function(self, name: str) -> bool:
        name = name.strip()
        if name in self.session.functions:
            del self.session.functions[name]
            return True
        return False

    def clear_functions(self) -> None:
        self.session.functions.clear()

    def list_functions(self) -> Dict[str, sp.Lambda]:
        return dict(self.session.functions)

    def _get_func_expr_var(
        self, name: str, var_name: str | None = None
    ) -> Tuple[sp.Lambda, sp.Expr, sp.Symbol]:
        """Helper: retrieve a named function's Lambda, body expression, and
        the actual internal variable used inside the expression.

        When a user specifies *var_name* we locate the matching Lambda
        parameter rather than creating a new session symbol (which could be
        a different object due to SymPy Dummy variables inside Lambda).
        """
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError(f"Specify variable for function {name}")
        if var_name:
            # Try to match the requested name to one of the Lambda params
            match = None
            for v in vars_seq:
                if str(v) == var_name or v.name == var_name:
                    match = v
                    break
            if match is None:
                # Fallback: use session symbol (may work if names align)
                match = self.session.get_symbol(var_name)
            var = match
        else:
            var = vars_seq[0]
        return lam, lam.expr, var

    def derivative_func(
        self, name: str, var_name: str | None = None, n: int = 1
    ) -> Tuple[str, sp.Lambda]:
        lam, expr, var = self._get_func_expr_var(name, var_name)
        expr_d = sp.diff(expr, var, n)
        lam_d = sp.Lambda(lam.variables, expr_d)
        gen = f"{name}_d{n if n!=1 else ''}"
        self.session.functions[gen] = lam_d
        return gen, lam_d

    def critical(self, name: str, var_name: str | None = None) -> List[sp.Expr]:
        lam, expr, var = self._get_func_expr_var(name, var_name)
        fprime = sp.diff(expr, var)
        sols = sp.solve(sp.Eq(fprime, 0), var)
        return sols

    def extrema(
        self, name: str, var_name: str | None = None
    ) -> List[Tuple[sp.Expr, str]]:
        lam, expr, var = self._get_func_expr_var(name, var_name)
        fprime = sp.diff(lam.expr, var)
        fsecond = sp.diff(lam.expr, var, 2)
        sols = sp.solve(sp.Eq(fprime, 0), var)
        # Attempt to determine global vs local by using function range
        ran = None
        try:
            ran = self.range(name, var_name)
        except Exception:
            ran = None
        out: List[Tuple[sp.Expr, str]] = []
        for s in sols:
            # Local classification via second derivative
            nature = "unknown"
            try:
                val2 = fsecond.subs(var, s)
                if val2.is_positive:
                    nature = "local min"
                elif val2.is_negative:
                    nature = "local max"
                elif val2.is_zero:
                    nature = "flat"
                else:
                    nature = "unknown"
            except Exception:
                nature = "unknown"
            # Check potential global by comparing to range endpoints
            try:
                yval = sp.simplify(lam.expr.subs(var, s))

                def eq0(a, b):
                    try:
                        return sp.simplify(a - b) == 0 or sp.Eq(a, b) is True
                    except Exception:
                        return False

                if isinstance(ran, sp.Interval):
                    # Global min if closed lower bound equals yval
                    if not ran.left_open and (eq0(yval, ran.inf)):
                        nature = "global min"
                    # Global max if closed upper bound equals yval
                    if not ran.right_open and (eq0(yval, ran.sup)):
                        nature = "global max"
            except Exception:
                pass
            out.append((s, nature))
        return out

    def inflection(self, name: str, var_name: str | None = None) -> List[sp.Expr]:
        lam, expr, var = self._get_func_expr_var(name, var_name)
        fsecond = sp.diff(expr, var, 2)
        sols = sp.solve(sp.Eq(fsecond, 0), var)
        return sols

    def solve_function(
        self,
        name: str,
        var_name: str | None = None,
        value_text: str | None = None,
        lo_text: str | None = None,
        hi_text: str | None = None,
    ) -> List[sp.Expr]:
        """
        Solve f(var) = value for the variable, optionally on an interval.
        Usage patterns (REPL will help parse):
          - solvef f               -> solve f(x) = 0 for x (single-var function)
          - solvef f, x            -> solve f(x) = 0 for x
          - solvef f, x, 2         -> solve f(x) = 2 for x
          - solvef f, x, 0, 0, 5  -> solve f(x) = 0 for x in [0, 5]

        Robust against complex function bodies (sqrt, abs, nested, etc.).
        Falls back to solveset and numerical solving when symbolic fails.
        """
        lam, expr, var = self._get_func_expr_var(name, var_name)
        value = self.parse(value_text) if value_text else sp.Integer(0)
        equation = sp.Eq(expr, value)

        interval = None
        if lo_text and hi_text:
            interval = (self.parse(lo_text), self.parse(hi_text))

        # Try symbolic solve first
        try:
            result = sp.solve(equation, var)
            if result:  # non-empty
                if interval:
                    result = self._filter_interval(result, var, *interval)
                return result
        except (NotImplementedError, Exception):
            pass

        # Fallback: solveset (better for sqrt, piecewise, etc.)
        try:
            if interval:
                dom = sp.Interval(interval[0], interval[1])
            else:
                dom = sp.S.Reals
            ss = sp.solveset(equation, var, domain=dom)
            if ss is not sp.S.EmptySet and ss.is_FiniteSet:
                return sorted(list(ss), key=lambda s: complex(sp.N(s)).real)
        except Exception:
            pass

        # Last resort: numerical roots via nsolve over a grid
        try:
            f_eq = expr - value
            numerical_sols = set()
            import warnings

            lo_val = int(float(sp.N(interval[0]))) if interval else -20
            hi_val = int(float(sp.N(interval[1]))) if interval else 20
            # Expand grid slightly and use finer steps for small intervals
            step = max(1, (hi_val - lo_val) // 40)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for guess_int in range(lo_val - 1, hi_val + 2, step):
                    guess = float(guess_int)
                    try:
                        sol = sp.nsolve(f_eq, var, guess, tol=1e-10)
                        # Round to avoid near-duplicates
                        rounded = complex(sol).real
                        is_dup = any(
                            abs(rounded - existing) < 1e-8
                            for existing in numerical_sols
                        )
                        if not is_dup:
                            if interval:
                                if (
                                    float(sp.N(interval[0]))
                                    <= rounded
                                    <= float(sp.N(interval[1]))
                                ):
                                    numerical_sols.add(rounded)
                            else:
                                numerical_sols.add(rounded)
                    except Exception:
                        continue
            if numerical_sols:
                return [
                    sp.nsimplify(sp.Float(s), rational=False)
                    for s in sorted(numerical_sols)
                ]
        except Exception:
            pass

        return []

    def domain(self, name: str, var_name: str | None = None) -> sp.Set:
        lam, expr, var = self._get_func_expr_var(name, var_name)
        from sympy.calculus.util import continuous_domain

        return continuous_domain(expr, var, sp.S.Reals)

    def range(self, name: str, var_name: str | None = None) -> sp.Set:
        lam, expr, var = self._get_func_expr_var(name, var_name)
        from sympy.calculus.util import function_range, continuous_domain

        dom = continuous_domain(expr, var, sp.S.Reals)
        try:
            return function_range(expr, var, dom)
        except Exception:
            return function_range(expr, var, sp.S.Reals)

    # ---- New calculus / utility commands ----

    def arclength(self, args_text: str) -> sp.Expr:
        """Arc length of a curve.
        Usage:
          arclength <f>, <var>, <a>, <b>       — named function
          arclength <expr>, <var>, <a>, <b>     — raw expression
        Formula: L = ∫_a^b √(1 + (dy/dx)²) dx
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 3:
            raise ValueError("Usage: arclength <f|expr>, <var>, <a>, <b>")
        first = parts[0]
        if first in self.session.functions:
            idx = 1
            var_override = None
            if (
                len(parts) >= 2
                and parts[1].strip().isidentifier()
                and not parts[1].strip().lstrip("-").replace(".", "").isdigit()
            ):
                var_override = parts[1].strip()
                idx = 2
            if len(parts) < idx + 2:
                raise ValueError("Usage: arclength <f>, [<var>,] <a>, <b>")
            _lam, expr, var = self._get_func_expr_var(first, var_override)
            a = self.parse(parts[idx])
            b = self.parse(parts[idx + 1])
        else:
            if len(parts) < 4:
                raise ValueError("Usage: arclength <expr>, <var>, <a>, <b>")
            expr = self.parse(parts[0])
            var = self.session.get_symbol(parts[1])
            a = self.parse(parts[2])
            b = self.parse(parts[3])
        deriv = sp.diff(expr, var)
        integrand = sp.sqrt(1 + deriv**2)
        integrand = sp.nsimplify(integrand, rational=True)
        result = sp.integrate(integrand, (var, a, b))
        return sp.simplify(result)

    def volume(self, args_text: str) -> sp.Expr:
        """Volume of solid of revolution.
        Usage:
          volume disk <f|expr>, <var>, <a>, <b>
            — Disk method (rotation about x-axis):
              V = π ∫_a^b [f(x)]² dx

          volume washer <f|expr>, <g|expr>, <var>, <a>, <b>
            — Washer method (rotation about x-axis, region between f and g):
              V = π ∫_a^b ([f(x)]² − [g(x)]²) dx

          volume shell <f|expr>, <var>, <a>, <b>
            — Shell method (rotation about y-axis):
              V = 2π ∫_a^b x·|f(x)| dx
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError(
                "Usage: volume disk|washer|shell <f|expr>, <var>, <a>, <b>"
            )
        method = parts[0].lower().split()[0] if parts[0] else ""

        if method == "disk":
            # disk <f|expr>, <var>, <a>, <b>
            first_rest = parts[0].split(None, 1)
            if len(first_rest) < 2:
                raise ValueError("Usage: volume disk <f|expr>, <var>, <a>, <b>")
            first = first_rest[1].strip()
            if first in self.session.functions:
                _lam, expr, var = self._get_func_expr_var(first)
                var_idx = 1
            else:
                expr = self.parse(first)
                var_idx = 1
                var = self.session.get_symbol(parts[var_idx])
                var_idx = 2
            if first in self.session.functions:
                var_override = parts[1].strip() if len(parts) >= 2 else None
                if (
                    var_override
                    and var_override.isidentifier()
                    and not var_override.lstrip("-").replace(".", "").isdigit()
                ):
                    _lam, expr, var = self._get_func_expr_var(first, var_override)
                    var_idx = 2
                else:
                    var_idx = 1
            if len(parts) < var_idx + 2:
                raise ValueError("Usage: volume disk <f|expr>, <var>, <a>, <b>")
            a = self.parse(parts[var_idx])
            b = self.parse(parts[var_idx + 1])
            integrand = sp.pi * expr**2
            integrand = sp.nsimplify(integrand, rational=True)
            result = sp.integrate(integrand, (var, a, b))
            return sp.simplify(result)
            # washer <f|expr>, <g|expr>, <var>, <a>, <b>
            first_rest = parts[0].split(None, 1)
            if len(first_rest) < 2:
                raise ValueError(
                    "Usage: volume washer <f|expr>, <g|expr>, <var>, <a>, <b>"
                )
            first = first_rest[1].strip()
            if first in self.session.functions:
                _lam, outer_expr, var = self._get_func_expr_var(first)
            else:
                outer_expr = self.parse(first)
            if len(parts) < 4:
                raise ValueError(
                    "Usage: volume washer <f|expr>, <g|expr>, <var>, <a>, <b>"
                )
            second = parts[1].strip()
            if second in self.session.functions:
                _lam2, inner_expr, _var2 = self._get_func_expr_var(second)
            else:
                inner_expr = self.parse(second)
            var = self.session.get_symbol(parts[2])
            a = self.parse(parts[3])
            b = self.parse(parts[4]) if len(parts) >= 5 else self.parse(parts[3])
            if len(parts) < 5:
                raise ValueError(
                    "Usage: volume washer <f|expr>, <g|expr>, <var>, <a>, <b>"
                )
            b = self.parse(parts[4])
            integrand = sp.pi * (outer_expr**2 - inner_expr**2)
            integrand = sp.nsimplify(integrand, rational=True)
            result = sp.integrate(integrand, (var, a, b))
            return sp.simplify(result)

        elif method == "shell":
            # shell <f|expr>, <var>, <a>, <b>
            first_rest = parts[0].split(None, 1)
            if len(first_rest) < 2:
                raise ValueError("Usage: volume shell <f|expr>, <var>, <a>, <b>")
            first = first_rest[1].strip()
            if first in self.session.functions:
                _lam, expr, var = self._get_func_expr_var(first)
                var_idx = 1
            else:
                expr = self.parse(first)
                var_idx = 1
                var = self.session.get_symbol(parts[var_idx])
                var_idx = 2
            if first in self.session.functions:
                var_override = parts[1].strip() if len(parts) >= 2 else None
                if (
                    var_override
                    and var_override.isidentifier()
                    and not var_override.lstrip("-").replace(".", "").isdigit()
                ):
                    _lam, expr, var = self._get_func_expr_var(first, var_override)
                    var_idx = 2
                else:
                    var_idx = 1
            if len(parts) < var_idx + 2:
                raise ValueError("Usage: volume shell <f|expr>, <var>, <a>, <b>")
            a = self.parse(parts[var_idx])
            b = self.parse(parts[var_idx + 1])
            integrand = 2 * sp.pi * var * sp.Abs(expr)
            integrand = sp.nsimplify(integrand, rational=True)
            result = sp.integrate(integrand, (var, a, b))
            return sp.simplify(result)

        else:
            raise ValueError(
                "Unknown method. Usage:\n"
                "  volume disk <f|expr>, <var>, <a>, <b>\n"
                "  volume washer <outer>, <inner>, <var>, <a>, <b>\n"
                "  volume shell <f|expr>, <var>, <a>, <b>"
            )

    def summation(self, args_text: str) -> sp.Expr:
        """Symbolic summation.
        Usage: sum <expr>, <var>, <a>, <b>
        Computes Σ_{var=a}^{b} expr
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 4:
            raise ValueError("Usage: sum <expr>, <var>, <a>, <b>")
        expr = self.parse(parts[0])
        var = self.session.get_symbol(parts[1])
        a = self.parse(parts[2])
        b = self.parse(parts[3])
        return sp.summation(expr, (var, a, b))

    def product(self, args_text: str) -> sp.Expr:
        """Symbolic product.
        Usage: product <expr>, <var>, <a>, <b>
        Computes Π_{var=a}^{b} expr
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 4:
            raise ValueError("Usage: product <expr>, <var>, <a>, <b>")
        expr = self.parse(parts[0])
        var = self.session.get_symbol(parts[1])
        a = self.parse(parts[2])
        b = self.parse(parts[3])
        return sp.product(expr, (var, a, b))

    def apart(self, args_text: str) -> sp.Expr:
        """Partial fraction decomposition.
        Usage: apart <expr> [, <var>]
        """
        parts = [p.strip() for p in args_text.split(",")]
        if not parts or not parts[0]:
            raise ValueError("Usage: apart <expr> [, <var>]")
        expr = self.parse(parts[0])
        if len(parts) >= 2 and parts[1]:
            var = self.session.get_symbol(parts[1])
            return sp.apart(expr, var)
        return sp.apart(expr)

    def taylor(self, args_text: str) -> sp.Expr:
        """Taylor polynomial (no remainder term, unlike series).
        Usage: taylor <expr>, <var> [, <point>] [, <n>]
        Default point=0, n=6.
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: taylor <expr>, <var> [, <point>] [, <n>]")
        expr = self.parse(parts[0])
        var = self.session.get_symbol(parts[1])
        point = 0
        n = 6
        if len(parts) >= 3 and parts[2]:
            point = self.parse(parts[2])
        if len(parts) >= 4 and parts[3]:
            n = int(parts[3])
        s = sp.series(expr, var, point, n)
        # Remove the O(...) term to get just the polynomial
        return s.removeO()

    def tangent_line(self, args_text: str) -> Tuple[sp.Expr, sp.Expr]:
        """Tangent line to a function at a point.
        Usage: tangent <f>, <point> [, <var>]
        Returns (slope, line_expr).
        Line: y = f(a) + f'(a)*(x - a)
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: tangent <f>, <point> [, <var>]")
        fname = parts[0]
        var_name = parts[2].strip() if len(parts) >= 3 and parts[2] else None
        lam, expr, var = self._get_func_expr_var(fname, var_name)
        pt = self.parse(parts[1])
        deriv = sp.diff(expr, var)
        slope = deriv.subs(var, pt)
        y_at = expr.subs(var, pt)
        line = sp.simplify(y_at + slope * (var - pt))
        return slope, line

    def normal_line(self, args_text: str) -> Tuple[sp.Expr, sp.Expr]:
        """Normal line to a function at a point.
        Usage: normal <f>, <point> [, <var>]
        Returns (slope, line_expr).
        Normal slope = -1/f'(a).
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: normal <f>, <point> [, <var>]")
        fname = parts[0]
        var_name = parts[2].strip() if len(parts) >= 3 and parts[2] else None
        lam, expr, var = self._get_func_expr_var(fname, var_name)
        pt = self.parse(parts[1])
        deriv = sp.diff(expr, var)
        tangent_slope = deriv.subs(var, pt)
        if tangent_slope == 0:
            # Normal is vertical: x = pt
            return sp.zoo, sp.Symbol("x") - pt  # vertical marker
        slope = sp.Rational(-1, 1) / tangent_slope
        y_at = expr.subs(var, pt)
        line = sp.simplify(y_at + slope * (var - pt))
        return slope, line

    def table(self, args_text: str) -> List[Tuple[sp.Expr, sp.Expr]]:
        """Table of values.
        Usage: table <f>, <var>, <a>, <b> [, <step>]
        Default step=1. Works with named functions or expressions.
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 4:
            raise ValueError("Usage: table <f|expr>, <var>, <a>, <b> [, <step>]")
        first = parts[0]
        if first in self.session.functions:
            lam = self.session.functions[first]
            expr = lam.expr
        else:
            expr = self.parse(first)
        var = self.session.get_symbol(parts[1])
        a = self.parse(parts[2])
        b = self.parse(parts[3])
        step = self.parse(parts[4]) if len(parts) >= 5 and parts[4] else sp.Integer(1)
        rows: List[Tuple[sp.Expr, sp.Expr]] = []
        current = a
        # Safety limit to avoid infinite loops
        max_rows = 200
        count = 0
        while sp.N(current) <= sp.N(b) and count < max_rows:
            y = expr.subs(var, current)
            try:
                y = sp.simplify(y)
            except Exception:
                pass
            rows.append((current, y))
            current = current + step
            count += 1
        return rows

    def avgval(self, args_text: str) -> sp.Expr:
        """Average value of a function on [a, b].
        Usage: avgval <f|expr>, <var>, <a>, <b>
        Formula: 1/(b-a) * ∫_a^b f(x) dx
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 4:
            raise ValueError("Usage: avgval <f|expr>, <var>, <a>, <b>")
        first = parts[0]
        if first in self.session.functions:
            lam = self.session.functions[first]
            expr = lam.expr
        else:
            expr = self.parse(first)
        var = self.session.get_symbol(parts[1])
        a = self.parse(parts[2])
        b = self.parse(parts[3])
        integral = sp.integrate(expr, (var, a, b))
        return sp.simplify(integral / (b - a))

    def compose(self, args_text: str) -> Tuple[str, sp.Lambda]:
        """Compose two functions: (f∘g)(x) = f(g(x)).
        Usage: compose <f>, <g>
        Creates a new function named f_g.
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) < 2:
            raise ValueError("Usage: compose <f>, <g>")
        fname, gname = parts[0], parts[1]
        f_lam = self.session.functions.get(fname)
        g_lam = self.session.functions.get(gname)
        if f_lam is None:
            raise ValueError(f"No function named {fname}")
        if g_lam is None:
            raise ValueError(f"No function named {gname}")
        # Only works with single-variable functions
        f_var = (
            f_lam.variables
            if not isinstance(f_lam.variables, tuple)
            else f_lam.variables[0]
        )
        g_var = (
            g_lam.variables
            if not isinstance(g_lam.variables, tuple)
            else g_lam.variables[0]
        )
        composed_expr = f_lam.expr.subs(f_var, g_lam.expr)
        composed_expr = sp.simplify(composed_expr)
        new_name = f"{fname}_{gname}"
        new_lam = sp.Lambda(g_var, composed_expr)
        self.session.functions[new_name] = new_lam
        return new_name, new_lam

    def plot(self, args_text: str) -> str:
        """
        ASCII plot of a function or expression.
        Usage: plot <function_name> [, <var>] [, <xmin>] [, <xmax>] [, <width>] [, <height>]
        Or: plot <expr>, <var> [, <xmin>] [, <xmax>] [, <width>] [, <height>]
        """
        import math

        parts = [p.strip() for p in args_text.split(",")]
        if not parts or not parts[0]:
            raise ValueError(
                "Usage: plot <function_name>|<expr> [, <var>] [, <xmin>] [, <xmax>] [, <width>] [, <height>]"
            )

        # Default parameters
        x_min, x_max = -10, 10
        width, height = 60, 20
        expr = None
        var = None

        # Try to parse first argument as function name
        func_name = parts[0]
        if func_name in self.session.functions:
            # Plotting a defined function
            lam = self.session.functions[func_name]
            expr = lam.expr
            if isinstance(lam.variables, tuple):
                if len(lam.variables) != 1:
                    raise ValueError("Can only plot single-variable functions")
                var = lam.variables[0]
            else:
                var = lam.variables

            # Parse additional parameters for function plot: var, xmin, xmax, width, height
            if len(parts) >= 2 and parts[1]:
                var = self.session.get_symbol(parts[1])
            if len(parts) >= 3 and parts[2]:
                x_min = float(self.parse(parts[2]).evalf(6))
            if len(parts) >= 4 and parts[3]:
                x_max = float(self.parse(parts[3]).evalf(6))
            if len(parts) >= 5 and parts[4]:
                width = int(parts[4])
            if len(parts) >= 6 and parts[5]:
                height = int(parts[5])
        else:
            # Plotting an expression
            if len(parts) < 2:
                raise ValueError(
                    "For expressions, specify variable: plot <expr>, <var> [, <xmin>] [, <xmax>] [, <width>] [, <height>]"
                )

            expr = self.parse(parts[0])
            var = self.session.get_symbol(parts[1])

            if len(parts) >= 3 and parts[2]:
                x_min = float(self.parse(parts[2]).evalf(6))
            if len(parts) >= 4 and parts[3]:
                x_max = float(self.parse(parts[3]).evalf(6))
            if len(parts) >= 5 and parts[4]:
                width = int(parts[4])
            if len(parts) >= 6 and parts[5]:
                height = int(parts[5])

        # Validate parameters
        if width < 10 or width > 200:
            width = 60
        if height < 5 or height > 50:
            height = 20
        if x_min >= x_max:
            x_min, x_max = -10, 10

        # Convert sympy expression to evaluable function
        try:
            func = sp.lambdify(var, expr, "math")
        except Exception:
            # Fallback for expressions that don't work with math module
            func = sp.lambdify(var, expr)

        # Generate x values
        x_vals = [x_min + (x_max - x_min) * i / (width - 1) for i in range(width)]

        # Evaluate function
        y_vals = []
        for x_val in x_vals:
            try:
                y = func(x_val)
                if isinstance(y, complex):
                    y = y.real
                if hasattr(math, "isinf") and math.isinf(y):
                    y = 1e6 if y > 0 else -1e6
                elif hasattr(math, "isnan") and math.isnan(y):
                    y = 0.0
                y_vals.append(float(y))
            except Exception:
                y_vals.append(0.0)

        # Scale to fit height
        y_min, y_max = min(y_vals), max(y_vals)
        if abs(y_max - y_min) < 1e-10:  # Nearly constant function
            y_center = y_min
            y_min = y_center - 0.5
            y_max = y_center + 0.5

        # Create plot
        lines = []
        for row in range(height):
            line = []
            y_threshold = y_min + (y_max - y_min) * (height - 1 - row) / (height - 1)

            for col in range(width):
                if abs(y_vals[col] - y_threshold) < (y_max - y_min) / height * 1.2:
                    line.append("*")
                else:
                    line.append(" ")
            lines.append("".join(line))

        # Add axis labels and info
        plot_text = []
        if func_name in self.session.functions:
            plot_text.append(f"Plot of function {func_name}({var}):")
        else:
            plot_text.append(f"Plot of {expr}:")
        plot_text.extend(lines)
        plot_text.append(
            f"X: {x_min:.2g} to {x_max:.2g}, Y: {y_min:.2g} to {y_max:.2g}"
        )

        return "\n".join(plot_text)
