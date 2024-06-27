import casadi as ca
from common.quat_casadi import (
  quat_rot,
  quat_mul,
  get_quat_mat,
  get_quat_right_mat,
  quat_mul_vec,
  quat_vec_part,
  quat_conj, 
  quat_conj_mul,
  quat_mul_conj
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

def calc_gen_acc_expression(table_angvel : float, table_angacc : float, J : ca.DM, m : float, r : float):
  R"""
    :param table_angvel: table initial angular velocity
    :param table_angacc: table angular acceleration
    :param J: ball inertia tensor
    :param m: ball mass
    :param r: ball radius
    :return: ca.Function (t, q, dq) -> ddq
  """
  I = ca.DM.eye(3)
  ex = I[:,0]
  ey = I[:,1]
  ez = I[:,2]
  t = ca.SX.sym('t') # time
  phi = ca.SX.sym('phi', 4) # quaternion of ball orientation wrt table
  dphi = ca.SX.sym('dphi', 4) # derivative of quaternion of ball orientation wrt table
  x = ca.SX.sym('x') # ball x coordinate in table frame
  y = ca.SX.sym('y') # ball y coordinate in table frame
  dx = ca.SX.sym('x') # ball x coordinate in table frame
  dy = ca.SX.sym('y') # ball y coordinate in table frame
  gencoords = ca.vertcat(phi, x, y)
  genvels = ca.vertcat(dphi, dx, dy)

  theta = table_angvel * t + table_angacc * t**2 / 2 # table angular positon
  mu = ca.vertcat(ca.cos(theta / 2), ez * ca.sin(theta / 2)) # quaternion of table orientation
  psi = quat_mul(mu, phi) # quaternion of ball orientation wrt world
  dpsi = ca.jtimes(psi, phi, dphi) + ca.jacobian(psi, t)
  w = 2 * quat_vec_part(quat_conj_mul(psi, dpsi)) # ball angular velocity
  lagrangian = 0.5 * w.T @ J @ w # ball rot energy
  lagrangian += 0.5 * (phi.T @ dphi)**2

  p = quat_rot(mu, ca.vertcat(x, y, 0)) # ball position vector wrt world frame
  v = ca.jtimes(p, ca.vertcat(x, y, t), ca.vertcat(dx, dy, 1)) # ball velocity wrt world frame
  lagrangian += 0.5 * m * v.T @ v # ball trans energy

  # matrix of constraints
  ball_angvel_wrt_table_in_table_frame = 2 * quat_vec_part(quat_mul_conj(dphi, phi))
  constraints = ca.vertcat(
    phi.T @ dphi,
    dx + r * ex.T @ cross(ez, ball_angvel_wrt_table_in_table_frame),
    dy + r * ey.T @ cross(ez, ball_angvel_wrt_table_in_table_frame),
  )
  A = ca.substitute(ca.jacobian(constraints, genvels), genvels, 0).T

  # lagrange equations
  genacc = lagrange_1st_kind(lagrangian, A, t, gencoords, genvels)
  genacc_fun = ca.Function('ddq', [t, gencoords, genvels], [genacc])
  return genacc_fun
