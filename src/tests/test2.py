from sympy import symbols, sin, cos, tan, Matrix, pprint, eye, zeros
from sympy.physics.mechanics import *

def wedge(a):
  x,y,z = a
  return Matrix([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ])

def get_angvel_euler_mat(theta, phi, psi):
  sin_theta = sin(theta)
  cos_theta = cos(theta)
  sin_phi = sin(phi)
  cos_phi = cos(phi)

  return Matrix([
    [cos_phi, 0, sin_theta*sin_phi],
    [-sin_phi, 0, sin_theta*cos_phi],
    [0, 1, cos_theta]
  ])

q = dynamicsymbols('theta phi psi x y')
u = dynamicsymbols('u_(1:6)')
dq = dynamicsymbols('theta phi psi x y', 1)
du = dynamicsymbols('u_(1:6)', 1)
mechanics_printing(pretty_print=False)
I = eye(3)
ex = I[:,0]
ey = I[:,1]
ez = I[:,2]

X = get_angvel_euler_mat(*q[0:3])
kd = zeros(5,1)
kd[0:3,0] = Matrix(u[0:3]) - X @ Matrix(dq[0:3])
kd[3,0] = dq[3] + (ex.T @ wedge(ez) @ X @ Matrix(dq[0:3]))[0,0]
kd[4,0] = dq[4] + (ey.T @ wedge(ez) @ X @ Matrix(dq[0:3]))[0,0]

WorldFrame = ReferenceFrame('WorldFrame')
KM = KanesMethod(WorldFrame, q_ind=q, u_ind=u, kd_eqs=kd)
kdd = KM.kindiffdict()

BodyFrame = ReferenceFrame('BodyFrame')
ball_center_point = Point('ball_center')

I = outer(BodyFrame.x, BodyFrame.x) + outer(BodyFrame.y, BodyFrame.y) + outer(BodyFrame.z, BodyFrame.z)
inertia_tuple = (I, I)
# point.set_vel(N, N.x * dq[3] + N.y * dq[4])
ball = RigidBody('Ball', ball_center_point, BodyFrame, 1, inertia_tuple)
# KM.kanes_equations([ball])
