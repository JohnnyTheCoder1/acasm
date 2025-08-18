- Utilities: `cpresult`, `cpopresult`, `last`, `history`, `del <var>`, `clearvars`, `restart`, `cls`.

Note: Clipboard features (`cpresult`, `cpopresult`) use `pyperclip`. If not installed, the commands will report that clipboard is unavailable. To enable:

```
pip install pyperclip
```
# Accessible CAS (Acasm)

A simple, screen-reader-friendly computer algebra system (CAS) for Windows terminals. It uses SymPy under the hood and focuses on clear, unambiguous text output suitable for JAWS and other screen readers.

## Why CLI?

- Terminal UIs tend to be more reliable with screen readers.
- No custom widgets or screen-reader drivers required.
- Everything is plain text with predictable formatting.

## Features

- REPL (read–eval–print loop) with line editing via your terminal.
- Evaluate expressions and assign variables: `x = 2`, `sin(x)^2 + cos(x)^2`.
- Algebra: `simplify`, `expand`, `factor`, `ratsimp`, `subs`.
- Calculus: `diff`, `integrate`, `limit`, `series`.
- Solve equations: `solve x^2 - 4 = 0`, `solve x^2 = 4, x`.
- Numeric evaluation: `evalf` with adjustable precision (`set digits n`).
- Linear algebra: `det`, `inv`, `rref` with matrix literals like `[[1,2],[3,4]]`.
- Data and regression: assign lists with `x = [1,2,3]`, then `linreg x, y`, `expreg x, y`, `powreg x, y`.
- Assumptions: `assume x real|integer|positive|negative|nonzero`, `forget x`.
- Session save/load: `save mysession.json`, `load mysession.json`.
- Options: ASCII output by default (better for screen readers), toggle Unicode if preferred, set digits.
 - Functions: define and analyze
	 - Define: `def f(x) = x^2 + 1` (also supports multiple arguments)
	 - Evaluate: `f(2)`
	 - Derivative function: `dfunc f` or `dfunc f, x, 2` (creates `f_d`/`f_d2`)
	 - Critical points: `critical f` (or specify variable)
	 - Extrema classification: `extrema f` (reports min/max/flat for critical points)
	 - Inflection points: `inflection f`
	 - Domain and range: `domain f`, `range f`
	 - Solve for x: `solvef f` (solves f(x)=0), `solvef f, x, 2` (solves f(x)=2)
	 - Manage: `funcs`, `delfunc f`, `clearfuncs`

### Built-ins

- Constants: `pi`/`Pi`, `E`/`e`, `I`, `EulerGamma`, `Catalan`, `GoldenRatio`/`phi`, `oo`/`inf`/`infinity`, `zoo`, `nan`.
- Trig/hyperbolic: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sec`, `csc`, `cot`, `sinh`, `cosh`, `tanh`, `sech`, `csch`, `coth`.
- Exponential/log: `exp`, `log`/`ln`, `sqrt`.
- Rounding/sign: `floor`, `ceiling`, `sign`.
- Extrema: `Max`, `Min`.
- Special/combinatorics: `gamma`, `factorial`, `binomial`.
- Complex helpers: `re`, `im`, `arg`, and `abs`.

## Quick start

1) Create a Python environment and install dependencies:

```
pip install -r requirements.txt
```

2) Run the REPL:

```
python main.py
```

3) Try some commands:

```
help
x = 3
sense = [1, 2, 3, 4]
meas  = [2.1, 4.0, 5.9, 8.2]
linreg sense, meas      # -> (m, b) for y = m*x + b
expreg sense, meas      # -> (a, b) for y = a*exp(b*x)
powreg sense, meas      # -> (a, b) for y = a*x^b
simplify (x^2 - 1)/(x - 1)
diff sin(x)^2, x
integrate exp(-x^2), x
solve x^2 - 4 = 0
solve x^2 = 4, x
vars
set digits 30
evalf pi
ratsimp (x^3 - 1)/(x - 1)
limit (sin(x))/x, x, 0
series exp(x), x, 0, 6
subs x^2 + y, x=2, y=3
det [[1,2],[3,4]]
assume x real
forget x
def f(x) = x^3 - 3*x
critical f
extrema f
inflection f
dfunc f, x, 2
domain f
range f
quit
```

## Tips for screen readers

- Output avoids heavy box-drawing. Fractions are shown inline like (a)/(b).
- Powers use `^`. Multiplication often omits `*` in math; we keep `*` visible (e.g., `2*x`).
- You can switch to Unicode pretty output: `set unicode on`. Switch back: `set unicode off`.
- For lists: assign as Python-like literal `x = [1,2,3]`. Elements must be numeric. Use these with `linreg`, `expreg`, or `powreg`.

## Roadmap

- Script mode to run a file of commands.
- Unit tests and packaging as an installable tool.
- Support for doing regression on a dataset e.g linear regression, exponential regression and more.

## License

MIT License. See the `LICENSE` file for details.
