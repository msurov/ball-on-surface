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

def compute_lin_vel(angvel_wrt_table_in_table_frame : float, x : float, y : float, cone_side_coef : float, ball_radius : float):
  atan_k = np.arctan(cone_side_coef)
  radial_coord = np.sqrt(x**2 + y**2)
  normal = np.array([
    -np.sin(atan_k) * x / radial_coord, -np.sin(atan_k) * y / radial_coord, np.cos(atan_k)
  ])
  v = ball_radius * np.cross(angvel_wrt_table_in_table_frame, normal)
  return v

def calc_gen_acc_expression(table_angvel : float, table_angacc : float, 
                            ball_inertia_tensor : ca.DM, ball_mass : float, 
                            ball_radius : float, cone_side_coef : float, gravity_accel : float):
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
  lagrangian = 0.5 * w.T @ ball_inertia_tensor @ w # ball rot energy
  lagrangian += 0.5 * (phi.T @ dphi)**2

  radial_coord = ca.sqrt(x**2 + y**2)
  z = cone_side_coef * radial_coord
  ball_pos_wrt_world = quat_rot(mu, ca.vertcat(x, y, z)) # ball position vector wrt world frame
  ball_vel_wrt_world = ca.jtimes(ball_pos_wrt_world, ca.vertcat(x, y, t), ca.vertcat(dx, dy, 1)) # ball velocity wrt world frame
  lagrangian += 0.5 * ball_mass * ball_vel_wrt_world.T @ ball_vel_wrt_world # ball trans energy
  lagrangian -= ball_mass * gravity_accel * z

  # matrix of constraints
  atan_k = ca.arctan(cone_side_coef)
  normal = ca.vertcat(
    -ca.sin(atan_k) * x / radial_coord, -ca.sin(atan_k) * y / radial_coord, ca.cos(atan_k)
  )
  ball_angvel_wrt_table_in_table_frame = 2 * quat_vec_part(quat_mul_conj(dphi, phi))
  unit_quat = phi.T @ dphi
  constraints = ca.vertcat(
    unit_quat,
    dx + ball_radius * ex.T @ cross(normal, ball_angvel_wrt_table_in_table_frame),
    dy + ball_radius * ey.T @ cross(normal, ball_angvel_wrt_table_in_table_frame)
  )
  A = ca.substitute(ca.jacobian(constraints, genvels), genvels, 0).T

  # lagrange equations
  genacc = lagrange_1st_kind(lagrangian, A, t, gencoords, genvels)
  genacc_fun = ca.Function('ddq', [t, gencoords, genvels], [genacc])
  return genacc_fun
