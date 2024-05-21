import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from common.quat_casadi import (
  quat_rot,
  quat_mul,
  get_quat_mat,
  get_quat_right_mat,
  quat_mul_vec,
  quat_vec_part,
  quat_conj, 
  quat_conj_mul
)
from scipy.integrate import solve_ivp
import common.quat as quat
from common.lagrange_method import lagrange_1st_kind

def cross(a, b):
  ax,ay,az = a.elements()
  bx,by,bz = b.elements()
  return ca.vertcat(
    ay*bz - az*by,
    -ax*bz + az*bx,
    ax*by - ay*bx
  )

def dynamics(table_angvel, table_angacc, J, m, r):
  I = ca.DM.eye(3)
  ex = I[:,0]
  ey = I[:,1]
  ez = I[:,2]
  t = ca.SX.sym('t') # time
  theta = table_angvel * t + table_angacc * t**2 / 2 # table angular positon
  mu = ca.vertcat(ca.cos(theta / 2), ez * ca.sin(theta / 2)) # quaternion of table orientation
  phi = ca.SX.sym('phi', 4) # quaternion of ball orientation wrt table
  dphi = ca.SX.sym('phi', 4) # derivative of quaternion of ball orientation wrt table
  psi = quat_mul(mu, phi) # quaternion of ball orientation wrt world
  x = ca.SX.sym('x') # ball x coordinate in table frame
  y = ca.SX.sym('y') # ball y coordinate in table frame
  dx = ca.SX.sym('x') # ball x coordinate in table frame
  dy = ca.SX.sym('y') # ball y coordinate in table frame
  gencoords = ca.vertcat(phi, x, y)
  genvels = ca.vertcat(dphi, dx, dy)

  dpsi = ca.jtimes(psi, phi, dphi) + ca.jacobian(psi, t)
  w = 2 * quat_vec_part(quat_conj_mul(psi, dpsi)) # ball angular velocity
  lagrangian = 0.5 * w.T @ J @ w # ball rot energy
  lagrangian += 0.5 * (phi.T @ dphi)**2

  p = quat_rot(mu, ca.vertcat(x, y, 0)) # ball position vector wrt world frame
  v = ca.jtimes(p, ca.vertcat(x, y, t), ca.vertcat(dx, dy, 1)) # ball velocity wrt world frame
  lagrangian += 0.5 * m * v.T @ v # ball trans energy

  # matrix of constraints
  w_wrt_table = 2 * quat_vec_part(quat_conj_mul(phi, dphi))
  constraints = ca.vertcat(
    phi.T @ dphi,
    dx + r * ex.T @ cross(ez, w_wrt_table),
    dy + r * ey.T @ cross(ez, w_wrt_table),
  )
  A = ca.substitute(ca.jacobian(constraints, genvels), genvels, 0).T

  # lagrange equations
  genacc = lagrange_1st_kind(lagrangian, A, t, gencoords, genvels)
  rhs_fun = ca.Function('rhs', [t, gencoords, genvels], [ca.vertcat(genvels, genacc)])
  rhs = lambda t, st: np.reshape(rhs_fun(t, st[0:6], st[6:12]), (12,))
  return rhs

def unpack(st):
  q = ca.DM(st[0:6])
  dq = ca.DM(st[6:12])
  phi = q[0:4]
  x = q[4]
  y = q[5]
  dphi = dq[0:4]
  dx = dq[4]
  dy = dq[5]
  w = 2 * quat_conj_mul(phi, dphi)
  return {
    'phi': phi,
    'x': x,
    'y': y,
    'dphi': dphi,
    'dx': dx,
    'dy': dy,
    'w': w
  }

def test():
  simtime = 5.
  rhs = dynamics(0.3, 0.0)
  q0 = np.array([1., 0., 0., 0.])
  p0 = np.array([1., 0.])
  dq0 = np.zeros(4)
  v0 = np.zeros(2)
  st0 = np.concatenate((q0, p0, dq0, v0))
  sol = solve_ivp(rhs, [0., simtime], st0, max_step=1e-3)
  phi = sol.y[0:4].T
  dphi = sol.y[6:10].T
  t = sol.t
  x = sol.y[4].T
  y = sol.y[5].T
  w = 2 * quat.quat_conj_mul(phi, dphi)[:,1:4]
  plt.figure('autocomp')
  plt.plot(t, w)
  plt.legend(['wx', 'wy', 'wz'])
  plt.grid(True)
  plt.show()

def compute_velocity(w, r):
  ez = np.array([0., 0., 1.])
  v = r * np.cross(w, ez)
  return v

def test2():
  simtime = 5.

  table_angvel = 0.3
  table_angacc = 0.08
  m = 0.1
  M = 0.07
  r = 0.18
  q0 = np.array([1., 0., 0., 0.])
  p0 = np.array([0.3, 0.7])
  w0 = np.array([1., 3., 2.])

  rhs = dynamics(
    table_angvel=table_angvel,
    table_angacc=table_angacc,
    J = M * ca.DM.eye(3),
    m = m,
    r = r
  )
  dq0 = 0.5 * quat.quat_mul_vec(q0, w0)
  v0 = compute_velocity(w0, r)
  st0 = np.concatenate((q0, p0, dq0, v0))
  dst = rhs(0., st0)
  ddq0 = dst[6:10]
  dw = 2 * quat.quat_conj_mul(q0, ddq0)[1:4]
  print('dw:', dw)

if __name__ == '__main__':
  test2()
