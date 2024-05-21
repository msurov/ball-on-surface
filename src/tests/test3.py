from sympy import symbols, sin, cos, tan, Matrix, pprint, eye, zeros
from sympy.physics.mechanics import *

q = dynamicsymbols('x y')
u = dynamicsymbols('u_1 u_2')

t, = q[0].args
ax, ay = symbols('a_x a_y', real=True)

theta = dynamicsymbols('theta')
dtheta = theta.diff()
ddtheta = dtheta.diff()

world_frame = ReferenceFrame('WorldFrame')
world_center = Point('WorldCenter')
world_center.set_vel(world_frame, 0)

table_frame = ReferenceFrame('TableFrame')
table_center = world_center
quat = (cos(theta/2), 0, 0, sin(theta/2))
table_frame.orient_quaternion(world_frame, quat)

point = Point('A')
point.set_pos(table_center, table_frame.x * q[0] + table_frame.y * q[1])
mass_point = Particle('A', point, 1)
energy = mass_point.kinetic_energy(world_frame)
energy = energy.simplify()

kd_eqs = [
  u[0] - q[0].diff(),
  u[1] - q[1].diff(),
]
vel_constr = [
  u[0] * ax + u[1] * ay
]
km = KanesMethod(world_frame, q_ind=q, u_ind=[u[0]], kd_eqs=kd_eqs, u_dependent=[u[1]], velocity_constraints=vel_constr)
bodies = [mass_point]

fr,fr_star = km.kanes_equations(bodies)
rhs = km.rhs()
rhs.simplify()
pprint(rhs)

# m = km.mass_matrix.inv()
# pprint(m)