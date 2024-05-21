import casadi as ca
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
from common.lagrange_method import lagrange_1st_kind
import numpy as np
import common.quat as quat


def cross(a, b):
  ax,ay,az = a.elements()
  bx,by,bz = b.elements()
  return ca.vertcat(
    ay*bz - az*by,
    -ax*bz + az*bx,
    ax*by - ay*bx
  )

def comp_lin_vel(p, w_fixed, r, k):
  x = ca.SX.sym('x')
  y = ca.SX.sym('y')
  z = k * ca.sqrt(x**2 + y**2)
  q = ca.vertcat(x, y, z)
  Dq = ca.jacobian(q, ca.vertcat(x, y))
  n = cross(Dq[:,0], Dq[:,1])
  n = n / ca.norm_2(n)
  v = r * cross(ca.DM(w_fixed), n)
  v = ca.DM(ca.substitute(v, ca.vertcat(x, y), p))
  v = np.reshape(v, (3,))
  return v

def calc_gen_acc_expression(table_angvel : float, table_angacc : float, J : ca.DM, m : float, r : float, k : float, g : float):
  R"""
    :param table_angvel: table initial angular velocity
    :param table_angacc: table angular acceleration
    :param J: ball inertia tensor
    :param m: ball mass
    :param r: ball radius
    :param k: cone parameter
    :param g: gravity acceleration
    :return: ca.Function (t, q, dq) -> ddq
  """
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

  z = k * ca.sqrt(x**2 + y**2)
  p = quat_rot(mu, ca.vertcat(x, y, z)) # ball position vector wrt world frame
  v = ca.jtimes(p, ca.vertcat(x, y, t), ca.vertcat(dx, dy, 1)) # ball velocity wrt world frame
  lagrangian += 0.5 * m * v.T @ v # ball trans energy
  lagrangian -= m * g * z

  # matrix of constraints
  w_wrt_table = 2 * quat_vec_part(quat_conj_mul(phi, dphi))
  r = ca.sqrt(x**2 + y**2)
  cone_normal_vec = ca.vertcat(
    -x * ca.sin(ca.arctan(k)) / r, -y * ca.sin(ca.arctan(k)) / r, ca.cos(ca.arctan(k))
  )
  constraints = ca.vertcat(
    phi.T @ dphi,
    dx + r * ex.T @ cross(cone_normal_vec, w_wrt_table),
    dy + r * ey.T @ cross(cone_normal_vec, w_wrt_table),
  )
  A = ca.substitute(ca.jacobian(constraints, genvels), genvels, 0).T

  # lagrange equations
  genacc = lagrange_1st_kind(lagrangian, A, t, gencoords, genvels)
  genacc_fun = ca.Function('ddq', [t, gencoords, genvels], [genacc])
  return genacc_fun
