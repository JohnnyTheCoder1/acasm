from pathlib import Path
import sys

# Ensure the project root is on sys.path for imports when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Acasm.engine import Engine
import sympy as sp

eng = Engine()

print("-- simplify --")
print(eng.format(eng.simplify("(x^2 - 1)/(x - 1)")))

print("-- diff --")
print(eng.format(eng.diff("sin(x)^2, x")))

print("-- integrate --")
print(eng.format(eng.integrate("exp(-x^2), x")))

print("-- definite integrate --")
print(eng.format(eng.integrate("x^2, x, 0, 1")))
print(eng.format(eng.integrate("sin(x), x, 0, pi")))

print("-- solve --")
print(eng.format(eng.solve("x^2 - 4")))
print(eng.format(eng.solve("x^2 = 4, x")))

print("-- solve with sqrt --")
print(eng.format(eng.solve("sqrt(x) = 3, x")))
print(eng.format(eng.solve("sqrt(x + 1) - 2, x")))

print("-- evalf --")
print(eng.format(eng.evalf("pi")))

print("-- ratsimp --")
print(eng.format(eng.ratsimp("(x^3 - 1)/(x - 1)")))

print("-- limit --")
print(eng.format(eng.limit("sin(x)/x, x, 0")))

print("-- series --")
print(str(eng.series("exp(x), x, 0, 4")))

print("-- subs --")
print(eng.format(eng.subs("x^2 + y, x=2, y=3")))

print("-- det --")
print(eng.format(eng.det("[[1,2],[3,4]]")))

print("-- constants/functions --")
print(eng.format(eng.evalf("pi")))
print(eng.format(eng.evalf("EulerGamma")))
print(eng.format(eng.simplify("ln(E)")))
print(eng.format(eng.simplify("sec(x)*cos(x)")))

print("-- functions & analysis --")
name, lam = eng.define_function("f", ["x"], "x^3 - 3*x")
print(name, lam)
print("critical", eng.critical("f"))
print("extrema", eng.extrema("f"))
print("inflection", eng.inflection("f"))
print("domain", eng.domain("f"))
print("range", eng.range("f"))

print("-- solvef basic --")
print("solvef f=0:", eng.solve_function("f"))

print("-- solvef with sqrt function --")
eng.define_function("g", ["x"], "sqrt(x + 4) - 2")
print("solvef g=0:", eng.solve_function("g"))

print("-- solvef with complex function --")
eng.define_function("h", ["x"], "x^4 - 5*x^2 + 4")
print("solvef h=0:", eng.solve_function("h"))

print("-- solve f(x) = value --")
print("solvef f=2:", eng.solve_function("f", value_text="2"))

print("-- apart (partial fractions) --")
print(eng.format(eng.apart("1/(x^2 - 1), x")))

print("-- sum --")
print(eng.format(eng.summation("k, k, 1, 10")))
print(eng.format(eng.summation("1/k^2, k, 1, oo")))

print("-- product --")
print(eng.format(eng.product("k, k, 1, 5")))

print("-- taylor --")
print(eng.format(eng.taylor("sin(x), x, 0, 6")))

print("-- tangent line --")
slope, line_expr = eng.tangent_line("f, 0")
print(f"tangent at x=0: slope={eng.format(slope)}, y={eng.format(line_expr)}")

print("-- normal line --")
slope, line_expr = eng.normal_line("f, 0")
print(f"normal at x=0: slope={eng.format(slope)}, y={eng.format(line_expr)}")

print("-- arclength --")
eng.define_function("line_fn", ["x"], "x")
print("arclength y=x on [0,1]:", eng.format(eng.arclength("line_fn, x, 0, 1")))

print("-- avgval --")
print("avgval x^2 on [0,3]:", eng.format(eng.avgval("x^2, x, 0, 3")))

print("-- table --")
rows = eng.table("f, x, -2, 2, 1")
for xv, yv in rows:
    print(f"  f({eng.format(xv)}) = {eng.format(yv)}")

print("-- compose --")
eng.define_function("p", ["x"], "x^2")
eng.define_function("q", ["x"], "x + 1")
cname, clam = eng.compose("p, q")
print(f"{cname}(x) = {eng.format(clam.expr)}")

print("-- assume/forget --")
print(eng.assume("x positive"))
print(eng.forget("x"))

print("-- comma decimals --")
eng.define_function("cd", ["x"], "1,2*x^1,5")
print("f(x) = 1.2*x^1.5:", eng.format(eng.session.functions["cd"].expr))

print("-- solve with interval --")
print("x^2-4=0 in [0,5]:", eng.format(eng.solve("x^2-4, x, 0, 5")))
print("x^2-4=0 in [-5,0]:", eng.format(eng.solve("x^2-4, x, -5, 0")))

print("-- solvef with interval --")
# f(x)=x^3-3x, solve f=0 in [0, 5]
sols = eng.solve_function("f", "x", "0", "0", "5")
print("solvef f=0 in [0,5]:", eng.format(sp.FiniteSet(*sols)) if sols else "{}")

print("-- volume disk --")
res = eng.volume("disk x^2, x, 0, 1")
print("V disk x^2 on [0,1]:", eng.format(res))

print("-- volume shell --")
res = eng.volume("shell x^2, x, 0, 1")
print("V shell x^2 on [0,1]:", eng.format(res))

print("-- plotting --")
print("Plotting sin(x):")
plot_result = eng.plot("sin(x), x, -6, 6, 30, 8")
print(plot_result[:200] + "..." if len(plot_result) > 200 else plot_result)
print("Plotting function f:")
plot_result = eng.plot("f, x, -2, 2, 25, 6")
print(plot_result[:200] + "..." if len(plot_result) > 200 else plot_result)
