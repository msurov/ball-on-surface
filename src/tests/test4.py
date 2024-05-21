from sympy import symbols, sin, cos, tan, Matrix, pprint, eye, zeros
from sympy.physics.mechanics import *


def rotmat(a):
  s = sin(a)
  c = cos(a)
  return Matrix([
    [c, -s],
    [s, c]
  ])

x, y = dynamicsymbols('x y')
t, = x.args
theta = dynamicsymbols('theta')

point_position = rotmat(theta) @ Matrix([x, y])
point_velocity = point_position.diff()
energy = point_velocity.T @ point_velocity / 2

u_ = dynamicsymbols('u_1 u_2')
u = Matrix(u_)
ax, ay = symbols('a_x a_y', real=True)
A = Matrix([
  [ax, -ay],
  [ax, ay]
])
B = A.inv()
dx,dy = B @ u
energy_quasi = energy.subs(
  [(x.diff(), dx), (y.diff(), dy)]
)
p = energy_quasi.jacobian(u)
energy_quasi.jacobian([x, y])
dp = p.diff(t)
dp.simplify()
pprint(dp[0])
pprint(dp[1])