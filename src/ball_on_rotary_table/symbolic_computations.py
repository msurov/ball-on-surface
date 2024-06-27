from sympy import *
from sympy.physics.mechanics import dynamicsymbols

def angleaxis(l, a):
  I = eye(3)
  l_x = wedge(l)
  R = I * cos(a) + l_x * sin(a) + l @ l.T * (1 - cos(a))
  return R

def wedge(a):
  x,y,z = a
  return Matrix([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ])

def compute_coefs():
  ρ = dynamicsymbols('rho', real=True, positive=True)
  ϕ = dynamicsymbols('phi', real=True)
  dρ = dynamicsymbols('rho', 1, real=True, positive=True)
  dϕ = dynamicsymbols('phi', 1, real=True)
  θ = dynamicsymbols('theta', real=True)
  dθ = dynamicsymbols('theta', 1, real=True)
  t, = ρ.args
  α = symbols('alpha', real=True)
  tan_α = tan(α)
  k = symbols('k', real=True, positive=True)
  r = symbols('r', real=True, positive=True)
  g = symbols('g', real=True, positive=True)
  z = ρ * k
  σ = Matrix([ρ * cos(ϕ), ρ * sin(ϕ), tan_α * ρ])
  σ_ρ = σ.diff(ρ)
  σ_ϕ = σ.diff(ϕ)
  σ_x = wedge(σ)
  n = σ_ρ.cross(σ_ϕ) # normal vector
  n = n / n.norm()
  n_x = wedge(n)
  n.simplify()
  dn = n.diff(t)
  dn_x = wedge(dn)
  ez = Matrix([0, 0, 1])
  ex = Matrix([1, 0, 0])
  ey = Matrix([0, 1, 0])
  ez_x = wedge(ez)

  I3 = eye(3)

  term1 = -k*g/r * n_x @ ez
  term2 = k/r * n_x @ diag(1, 1, 0) @ σ
  term3 = -ez_x * dθ + k * n_x @ dn_x - k * dθ * n_x @ n_x @ ez_x + 2 * k * dθ * n_x @ ez_x @ n_x
  term4 = (k * n_x @ wedge(σ/r - n) - I3) @ ez

  A = angleaxis(ey, α) @ angleaxis(ez, -ϕ)

  tmp = -wedge(A @ ez) * dϕ
  tmp.simplify()
  pprint(tmp)
  return

  tmp = -k*g/r * n_x @ ez
  tmp = A @ tmp
  tmp.simplify()
  pprint(tmp)

  tmp = k / r * n_x @ diag(1, 1, 0) @ σ 
  tmp = A @ tmp
  tmp.simplify()
  pprint(tmp)

  tmp = -ez_x - k * n_x @ n_x @ ez_x + 2 * k * n_x @ ez_x @ n_x
  tmp = A @ tmp @ A.T
  tmp.simplify()
  pprint(tmp)

  tmp = k * n_x @ dn_x
  tmp = A @ tmp @ A.T
  tmp.simplify()
  pprint(tmp)

  tmp = (k / r * n_x @ σ_x - eye(3) - k * n_x @ n_x) @ ez
  tmp = A @ tmp
  tmp.simplify()
  pprint(tmp)

  tmp = -wedge(A @ ez)
  tmp.simplify()
  pprint(tmp)

  T = Matrix([
    [cos(ϕ), sin(ϕ)],
    [-sin(ϕ)/ρ, cos(ϕ)/ρ],
  ])
  tmp = T @ eye(2, 3) @ (-r * n_x @ A.T)
  tmp.simplify()
  pprint(tmp)

def energy():
  ρ = dynamicsymbols('rho', real=True, positive=True)
  ϕ = dynamicsymbols('phi', real=True)
  dρ = dynamicsymbols('rho', 1, real=True, positive=True)
  dϕ = dynamicsymbols('phi', 1, real=True)
  ζn = dynamicsymbols('zeta_n', real=True)
  θ = dynamicsymbols('theta', real=True)
  dθ = dynamicsymbols('theta', 1, real=True)
  t, = ρ.args
  α = symbols('alpha', real=True)
  tan_α = tan(α)
  k = symbols('k', real=True, positive=True)
  r = symbols('r', real=True, positive=True)
  g = symbols('g', real=True, positive=True)
  m = symbols('m', real=True, positive=True)
  M = symbols('M', real=True, positive=True)
  z = ρ * k
  σ = Matrix([ρ * cos(ϕ), ρ * sin(ϕ), tan_α * ρ])
  σ_ρ = σ.diff(ρ)
  σ_ϕ = σ.diff(ϕ)
  σ_x = wedge(σ)
  n = σ_ρ.cross(σ_ϕ) # normal vector
  n = n / n.norm()
  n_x = wedge(n)
  n.simplify()
  dn = n.diff(t)
  dn_x = wedge(dn)
  ez = Matrix([0, 0, 1])
  ex = Matrix([1, 0, 0])
  ey = Matrix([0, 1, 0])
  ez_x = wedge(ez)
  cos_α = cos(α)

  I3 = eye(3)
  A = angleaxis(ey, α) @ angleaxis(ez, -ϕ)

  T = Matrix([
    [0, r * cos_α, 0],
    [-r/ρ, 0, 0],
    [0, 0, 1]
  ])
  ζ = T.inv() @ Matrix([dρ, dϕ, ζn])

  p = angleaxis(ez, θ) @ σ
  v = p.diff(t)
  ω = A.T @ ζ + ez * dθ
  K = ω.T @ ω * M / 2 + v.T @ v * m / 2
  K.simplify()
  Ldϕ = K.diff(dϕ)
  Lϕ = K.diff(ϕ)
  Ldϕ.simplify()
  Lϕ.simplify()
  print(latex(Ldϕ))

# compute_coefs()
energy()
