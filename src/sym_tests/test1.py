import sympy as sy
from sympy.physics.mechanics import dynamicsymbols

def vertcat(a_, b_):
  a = a_ if isinstance(a_, sy.Matrix) else sy.Matrix([[a_]])
  b = b_ if isinstance(b_, sy.Matrix) else sy.Matrix([[b_]])
  return a.col_join(b)

def horzcat(a_, b_):
  a = a_ if isinstance(a_, sy.Matrix) else sy.Matrix([[a_]])
  b = b_ if isinstance(b_, sy.Matrix) else sy.Matrix([[b_]])
  return a.row_join(b)

def sq(v):
    return (v.T @ v)[0,0]

def norm(v):
    return sy.sqrt(sq(v))

def test1():
    J = sy.Matrix([
        [0, -1],
        [1, 0]
    ])
    p_ = sy.symbols('p_(x:y)', real=True)
    p = sy.Matrix(p_)

    T = horzcat(-J @ p, p)
    # 1
    Tinv = T.T / sq(p)
    Tinv.simplify()
    # 2
    Tinv = vertcat(p.T @ J, p.T) / sq(p)
    Tinv.simplify()

    I = Tinv @ T
    I.simplify()
    sy.pprint(I)

def test2():
    J = sy.Matrix([
        [0, -1],
        [1, 0]
    ])
    p_ = sy.symbols('p_(x:y)', real=True)
    p = sy.Matrix(p_)
    expr1 = p @ p.T @ J + J @ p @ p.T
    expr2 = J * sq(p)
    sy.pprint(expr1 - expr2)

def test3():
    J = sy.Matrix([
        [0, -1],
        [1, 0]
    ])
    p_ = sy.symbols('p_(x:y)', real=True)
    θ = sy.symbols('theta', real=True)
    dθ = sy.symbols('dtheta', real=True)
    ddθ = sy.symbols('ddtheta', real=True)
    p = sy.Matrix(p_)
    ζϕ = sy.symbols('zeta_phi', real=True)
    ζρ = sy.symbols('zeta_rho', real=True)
    k = sy.symbols('k', real=True, positive=True)
    r = sy.symbols('r', real=True, positive=True)
    ρ = norm(p)
    dφ = -r / ρ * ζρ
    dζϕ = dθ * (k + 1) * ζφ + dφ * ζϕ + k * ρ * ddθ / r
    dζρ = ρ * k * dθ**2 / r - ζρ * (k * dθ + dθ + dφ)
    dp = p * r / norm(p) * ζϕ - J @ p * r * ζρ / norm(p) + J * p * dθ

def test4():
    J = sy.Matrix([
        [0, -1],
        [1, 0]
    ])
    p_ = dynamicsymbols('p_(x:y)')
    θ = dynamicsymbols('theta')
    k = sy.symbols('k', real=True, positive=True)
    r = sy.symbols('r', real=True, positive=True)
    ζρ = dynamicsymbols('zeta_rho')
    ζϕ = dynamicsymbols('zeta_phi')
    t, = θ.args
    p = sy.Matrix(p_)
    ρ = norm(p)
    dϕ = -r/ρ * ζρ
    dθ = θ.diff()
    ddθ = dθ.diff()
    dζρ = dθ * (k + 1) * ζϕ + dφ * ζϕ + k * ρ / r * ddθ
    dζϕ = ρ * k / r * dθ**2 - ζρ * (k * dθ + dθ + dϕ)
    dp = p * r / norm(p) * ζϕ - J @ p * r * ζρ / norm(p) + J * p * dθ
    dp.simplify()
    eq = dp - p.diff(t)
    sol = sy.solve(eq, [ζρ, ζϕ])
    ζρ_val = sol[ζρ].simplify()
    ζϕ_val = sol[ζϕ].simplify()

    ddp = dp.diff(t)
    expr = ddp.subs(ζρ.diff(), dζρ)
    expr = expr.subs(ζϕ.diff(), dζϕ)
    expr = expr.subs(ζϕ, ζϕ_val)
    expr = expr.subs(ζρ, ζρ_val)
    expr.simplify()
    print(sy.latex(expr))

test4()
