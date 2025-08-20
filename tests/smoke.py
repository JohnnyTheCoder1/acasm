from pathlib import Path
import sys

# Ensure the project root is on sys.path for imports when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from Acasm.engine import Engine

eng = Engine()

print("-- simplify --")
print(eng.format(eng.simplify('(x^2 - 1)/(x - 1)')))

print("-- diff --")
print(eng.format(eng.diff('sin(x)^2, x')))

print("-- integrate --")
print(eng.format(eng.integrate('exp(-x^2), x')))

print("-- solve --")
print(eng.format(eng.solve('x^2 - 4')))
print(eng.format(eng.solve('x^2 = 4, x')))

print("-- evalf --")
print(eng.format(eng.evalf('pi')))

print("-- ratsimp --")
print(eng.format(eng.ratsimp('(x^3 - 1)/(x - 1)')))

print("-- limit --")
print(eng.format(eng.limit('sin(x)/x, x, 0')))

print("-- series --")
print(str(eng.series('exp(x), x, 0, 4')))

print("-- subs --")
print(eng.format(eng.subs('x^2 + y, x=2, y=3')))

print("-- det --")
print(eng.format(eng.det('[[1,2],[3,4]]')))

print("-- constants/functions --")
print(eng.format(eng.evalf('pi')))
print(eng.format(eng.evalf('EulerGamma')))
print(eng.format(eng.simplify('ln(E)')))
print(eng.format(eng.simplify('sec(x)*cos(x)')))

print("-- functions & analysis --")
name, lam = eng.define_function('f', ['x'], 'x^3 - 3*x')
print(name, lam)
print('critical', eng.critical('f'))
print('extrema', eng.extrema('f'))
print('inflection', eng.inflection('f'))
print('domain', eng.domain('f'))
print('range', eng.range('f'))

print("-- plotting --")
print("Plotting sin(x):")
plot_result = eng.plot('sin(x), x, -6, 6, 30, 8')
print(plot_result[:200] + "..." if len(plot_result) > 200 else plot_result)
print("Plotting function f:")
plot_result = eng.plot('f, x, -2, 2, 25, 6')
print(plot_result[:200] + "..." if len(plot_result) > 200 else plot_result)
