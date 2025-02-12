import casadi as ca
import sympy as sy

def test_casadi():
  ϕ = ca.SX.sym('phi')
  ζ = ca.SX.sym('zeta', 2)
  t = ca.SX.sym('t')
  ζρ,ζϕ = ζ.elements()
  θ = 0.
  dθ = 6.

  sq = lambda x: x.T @ x

  r = 0.8
  k = 1 / (1 + 3/5)

  p = ca.SX.sym('p', 2)
  J = ca.DM([
    [0, -1],
    [1, 0]
  ])
  ρ = ca.norm_2(p)
  dϕ = -r * ζρ / ρ
  dρ = r * ζϕ

  dζρ = dθ * (k + 1) * ζφ + dφ * ζφ
  dζφ = ρ * k / r * dθ**2 - ζρ * (k * dθ + dθ + dφ)
  dp = p * r * ζϕ / ca.norm_2(p) - J @ p * r * ζρ / ca.norm_2(p) + J @ p * dθ
  dζ = ca.vertcat(dζρ, dζφ)
  ddp = ca.jtimes(dp, p, dp) + ca.jtimes(dp, ζ, dζ)
  curv = ddp / sq(dp) - dp * (dp.T @ ddp) / sq(dp)**2
  scalar_curv = dp.T @ J @ curv / ca.norm_2(dp)
  Z = ca.jtimes(scalar_curv, p, dp) + ca.jtimes(scalar_curv, ζ, dζ)

  ans = ca.substitute(Z, ca.vertcat(p, ζ), ca.DM([1,-7,3,4]))
  print(ans)

def norm(v):
  return sy.sqrt(v.T @ v)[0,0]

def jtimes(expr, x, dx):
  return expr.jacobian(x) @ dx

def vertcat(a_, b_):
  a = a_ if isinstance(a_, sy.Matrix) else sy.Matrix([[a_]])
  b = b_ if isinstance(b_, sy.Matrix) else sy.Matrix([[b_]])
  return a.col_join(b)

def test_sympy():
  sin = sy.sin
  cos = sy.cos
  sin = sy.sin
  sq = lambda v: (v.T @ v)[0,0]

  ϕ = sy.symbols('phi', real=True)
  ζρ = sy.symbols('zeta_rho', real=True)
  ζϕ = sy.symbols('zeta_phi', real=True)
  ζ = sy.Matrix([ζρ, ζϕ])
  dθ = 6
  r = sy.sympify('8/10')
  k = 1 / (1 + sy.sympify('3/5'))

  p_ = sy.symbols('p_(x:y)', real=True)
  p = sy.Matrix(p_)
  J = sy.Matrix([
    [0, -1],
    [1, 0]
  ])
  ρ = norm(p)
  dϕ = -r * ζρ / ρ
  dρ = r * ζϕ

  dζρ = dθ * (k + 1) * ζφ + dφ * ζφ
  dζφ = ρ * k / r * dθ**2 - ζρ * (k * dθ + dθ + dφ)
  dp = p * r * ζϕ / norm(p) - J @ p * r * ζρ / norm(p) + J @ p * dθ
  dζ = vertcat(dζρ, dζφ)

  ddp = jtimes(dp, p, dp) + jtimes(dp, ζ, dζ)
  ddp.simplify()
  sy.pprint(ddp)

  curv = ddp / sq(dp) - dp * (dp.T @ ddp) / sq(dp)**2
  curv.simplify()
  sy.pprint(curv)

  scalar_curv = dp.T @ J @ curv / norm(dp)
  scalar_curv.simplify()
  sy.pprint(scalar_curv)

  Z = jtimes(scalar_curv, p, dp) + jtimes(scalar_curv, ζ, dζ)
  Z.simplify()
  sy.pprint(Z)
  sy.pprint(Z.is_zero)
  # val = Z.subs({
  #     p[0]: 3,
  #     p[1]: 5,
  #     ζ[0]: 8,
  #     ζ[1]: -2,
  #   })
  # print(val.evalf())

# test_casadi()
test_sympy()
