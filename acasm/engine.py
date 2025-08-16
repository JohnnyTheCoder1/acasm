from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Sequence

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
    values: Dict[str, sp.Expr] = field(default_factory=dict)
    functions: Dict[str, sp.Lambda] = field(default_factory=dict)
    options: Options = field(default_factory=lambda: Options(**DEFAULT_OPTIONS.__dict__))

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

    # Parsing and evaluation
    def parse(self, text: str) -> sp.Expr:
        text = text.strip()
        if not text:
            raise ValueError("Empty input")
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
        expr = self.parse(expr_text)
        self.session.values[name] = expr
        # Also ensure a symbol exists for that name
        self.session.get_symbol(name)
        return name, expr

    def define_function(self, name: str, arg_names: Sequence[str], expr_text: str) -> Tuple[str, sp.Lambda]:
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
        text = expr_text.strip().replace("^", "**")
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        try:
            expr = parse_expr(text, local_dict=local_env, transformations=transformations, evaluate=True)
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
        # Expect: "expr, var"
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) != 2:
            raise ValueError("Usage: integrate <expr>, <var>")
        expr = self.parse(parts[0])
        var_name = parts[1]
        var = self.session.get_symbol(var_name)
        return sp.integrate(expr, var)

    def solve(self, args_text: str) -> sp.Expr:
        """
        Solve equations or expressions.
        Usage:
          - solve <expr>                # interprets as expr == 0
          - solve <expr>, <var>         # solve for a variable

        In <expr>, you can write equality with a single '=' or '==', e.g.:
          solve x^2 = 4, x
        """
        parts = [p.strip() for p in args_text.split(",")]
        if len(parts) == 0 or parts[0] == "":
            raise ValueError("Usage: solve <expr> [, <var>]")

        expr_str = parts[0]
        # Detect equality with single '=' or '==' and build Eq(lhs, rhs)
        eq_obj = None
        if "==" in expr_str:
            lhs_str, rhs_str = expr_str.split("==", 1)
            eq_obj = sp.Eq(self.parse(lhs_str), self.parse(rhs_str))
        elif "=" in expr_str:
            lhs_str, rhs_str = expr_str.split("=", 1)
            eq_obj = sp.Eq(self.parse(lhs_str), self.parse(rhs_str))

        if eq_obj is not None:
            target_expr = eq_obj
        else:
            target_expr = self.parse(expr_str)

        # Variables to solve for (support one or more)
        vars_list: List[sp.Symbol] = []
        if len(parts) >= 2:
            for v in parts[1:]:
                if v:
                    vars_list.append(self.session.get_symbol(v))

        if vars_list:
            return sp.solve(target_expr, *vars_list)
        # No variables provided: default to free symbols or equation solve
        if isinstance(target_expr, sp.Equality):
            return sp.solve(target_expr)
        free = list(target_expr.free_symbols)
        if not free:
            return sp.solve(target_expr)
        return sp.solve(target_expr, *free)

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

    # Assumptions
    def assume(self, args_text: str) -> str:
        # Usage: assume x real|integer|positive|negative|nonzero
        parts = args_text.split()
        if len(parts) != 2:
            raise ValueError("Usage: assume <symbol> <property>")
        name, prop = parts
        sym = self.session.get_symbol(name)
        props = {
            "real": sp.Q.real,
            "integer": sp.Q.integer,
            "positive": sp.Q.positive,
            "negative": sp.Q.negative,
            "nonzero": sp.Q.nonzero,
        }
        if prop not in props:
            raise ValueError("Property must be one of: real, integer, positive, negative, nonzero")
        sp.assume(props[prop](sym))
        return f"Assumed {name} is {prop}."

    def forget(self, name: str) -> str:
        sym = self.session.get_symbol(name.strip())
        sp.forget(sym)
        return f"Forgot assumptions about {name}."

    # Session save/load (variables and options)
    def save(self, path: str) -> str:
        import json
        data = {
            "values": {k: str(v) for k, v in self.session.values.items()},
            "functions": {
                k: {
                    "args": [str(a) for a in (v.variables if isinstance(v.variables, tuple) else (v.variables,))],
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
                self.session.values[k] = self.parse(v)
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

    def derivative_func(self, name: str, var_name: str | None = None, n: int = 1) -> Tuple[str, sp.Lambda]:
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError("Specify variable for multivariate function: derivative <f>, <var> [, <n>]")
        var = self.session.get_symbol(var_name) if var_name else vars_seq[0]
        expr_d = sp.diff(lam.expr, var, n)
        lam_d = sp.Lambda(lam.variables, expr_d)
        gen = f"{name}_d{n if n!=1 else ''}"
        self.session.functions[gen] = lam_d
        return gen, lam_d

    def critical(self, name: str, var_name: str | None = None) -> List[sp.Expr]:
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError("Specify variable: critical <f>, <var>")
        var = self.session.get_symbol(var_name) if var_name else vars_seq[0]
        fprime = sp.diff(lam.expr, var)
        sols = sp.solve(sp.Eq(fprime, 0), var)
        return sols

    def extrema(self, name: str, var_name: str | None = None) -> List[Tuple[sp.Expr, str]]:
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError("Specify variable: extrema <f>, <var>")
        var = self.session.get_symbol(var_name) if var_name else vars_seq[0]
        fprime = sp.diff(lam.expr, var)
        fsecond = sp.diff(lam.expr, var, 2)
        sols = sp.solve(sp.Eq(fprime, 0), var)
        out: List[Tuple[sp.Expr, str]] = []
        for s in sols:
            try:
                val = fsecond.subs(var, s)
                nature = "min" if val.is_positive else ("max" if val.is_negative else "flat")
            except Exception:
                nature = "unknown"
            out.append((s, nature))
        return out

    def inflection(self, name: str, var_name: str | None = None) -> List[sp.Expr]:
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError("Specify variable: inflection <f>, <var>")
        var = self.session.get_symbol(var_name) if var_name else vars_seq[0]
        fsecond = sp.diff(lam.expr, var, 2)
        sols = sp.solve(sp.Eq(fsecond, 0), var)
        return sols

    def solve_function(self, name: str, var_name: str | None = None, value_text: str | None = None) -> List[sp.Expr]:
        """
        Solve f(var) = value for the variable.
        Usage patterns (REPL will help parse):
          - solvef f               -> solve f(x) = 0 for x (single-var function)
          - solvef f, x            -> solve f(x) = 0 for x
          - solvef f, x, 2         -> solve f(x) = 2 for x
        """
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError("Specify variable: solvef <f>, <var> [, <value>]")
        var = self.session.get_symbol(var_name) if var_name else vars_seq[0]
        value = self.parse(value_text) if value_text else sp.Integer(0)
        return sp.solve(sp.Eq(lam.expr, value), var)

    def domain(self, name: str, var_name: str | None = None) -> sp.Set:
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError("Specify variable: domain <f>, <var>")
        var = self.session.get_symbol(var_name) if var_name else vars_seq[0]
        from sympy.calculus.util import continuous_domain
        return continuous_domain(lam.expr, var, sp.S.Reals)

    def range(self, name: str, var_name: str | None = None) -> sp.Set:
        lam = self.session.functions.get(name)
        if lam is None:
            raise ValueError(f"No function named {name}")
        if isinstance(lam.variables, tuple):
            vars_seq = lam.variables
        else:
            vars_seq = (lam.variables,)
        if len(vars_seq) != 1 and not var_name:
            raise ValueError("Specify variable: range <f>, <var>")
        var = self.session.get_symbol(var_name) if var_name else vars_seq[0]
        from sympy.calculus.util import function_range, continuous_domain
        dom = continuous_domain(lam.expr, var, sp.S.Reals)
        try:
            return function_range(lam.expr, var, dom)
        except Exception:
            return function_range(lam.expr, var, sp.S.Reals)

