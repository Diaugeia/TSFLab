---
name: "natural_cubic_spline"
description: "Natural cubic spline control path for neural CDEs: coefficients from a sampled [..., L, C] window on strictly increasing times, then X(t) and dX/dt at any scalar time. Use to drive a neural controlled differential equation from discrete observations; not for windows with missing values (NaN) or as a smoothing filter."
---

# natural_cubic_spline

## What it does

For observations `x_0 .. x_{L-1}` at times `t_0 < ... < t_{L-1}` it builds the
natural cubic spline (second derivative zero at both ends) through every point:
with `h_i = t_{i+1} - t_i` and `D_i = x_{i+1} - x_i` the knot slopes `k` solve a
tridiagonal system (solved densely, `L x L`), and piece `i` is

```
X(t) = a_i + b_i s + c_i s^2 + d_i s^3,   s = t - t_i
a_i = x_i, b_i = k_i, 2c_i = (6 D_i/h_i - 4 k_i - 2 k_{i+1}) / h_i,
3d_i = (-6 D_i/h_i + 3 (k_i + k_{i+1})) / h_i^2
dX/dt = b_i + (2c_i + 3d_i s) s
```

Times outside `[t_0, t_{L-1}]` extend the first or last piece.

## When to use

Use as the control path of a neural CDE (`dz = f(z) dX`) solved with an ODE
solver: the path interpolates every observation and is twice differentiable, so
the solver sees a smooth `dX/dt`. Do not use with NaN inputs (impute first), as
a smoother (it interpolates noise exactly), or for very long windows (the dense
solve is `O(L^3)` once per call).

## Interface

- `natural_cubic_spline_coeffs(times, x) -> (a, b, two_c, three_d)`: `times`
  `[L]` strictly increasing, `L >= 2` (cast to `x`'s dtype and device); `x`
  `[..., L, C]`. Each output is `[..., L - 1, C]`. Raises `ValueError` on bad
  shapes or non-increasing times. Differentiable in `x`.
- `NaturalCubicSpline(times, coeffs)`: `.evaluate(t)` and `.derivative(t)` take a
  scalar time (tensor or float) and return `[..., C]`.
- Stateless: no parameters, buffers, or state-dict keys.
