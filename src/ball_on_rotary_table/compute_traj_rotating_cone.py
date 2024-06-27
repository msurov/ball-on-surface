import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from common.surface import ConeSurface
from common.integrate import integrate_table
from common.quat import quat_rot, quat_mul
from common.rotations import solve_poisson_kinematics
from common.trajectory import RigidBodyTrajectory
from common.linalg import normalized
from .dynamics import SystemParameters
from .dynamics import (
  SystemParameters,
  Dynamics,
)
from common.frame_rotation import (
  FrameRotation,
  OscillatingFrame,
  FrameConstRot,
  FrameAccelRot
)

def get_bodies_traj(t : np.ndarray, st : np.ndarray, tabrot : FrameRotation, par : SystemParameters):
  n = len(t)
  q_table_ball = solve_poisson_kinematics(t, st[:,2:5], np.array([1., 0., 0., 0.]), 'fixed')
  q_world_ball = np.zeros((n, 4))
  p_world_ball = np.zeros((n, 3))
  q_world_table = np.zeros((n, 4))
  p_world_table = np.zeros((n, 3))
  w_world_table = np.zeros((n, 3))

  for i in range(n):
    q_world_table[i,:] = tabrot.rot(t[i])
    p_table_ball = par.surface.coords(*st[i, 0:2])
    p_world_ball[i,:] = quat_rot(q_world_table[i,:], p_table_ball)
    q_world_ball[i,:] = quat_mul(q_world_table[i,:], q_table_ball[i,:])
    w_world_table[i,:] = tabrot.angvel(t[i])

  ball_traj = RigidBodyTrajectory(
    t = t,
    p = p_world_ball,
    q = q_world_ball
  )
  table_traj = RigidBodyTrajectory(
    t = t,
    p = p_world_table,
    q = q_world_table,
    w = w_world_table
  )
  return ball_traj, table_traj

def get_auxiliary_signals(dynamics : Dynamics, t : np.ndarray, st : np.ndarray):
  n = len(t)
  friction_force = np.zeros((n, 3))
  normal_force = np.zeros((n,))
  for i in range(n):
    friction_force[i,:] = dynamics.friction_force(t[i], st[i])
    normal_force[i] = dynamics.normal_force(t[i], st[i])
  return friction_force, normal_force

def save_sim_data(time : np.ndarray, state : np.ndarray, tablerot : FrameRotation, par : SystemParameters):
  ball_traj, table_traj = get_bodies_traj(time, state, tablerot, par)
  np.save('../data/table_trajectory.npy', table_traj)
  np.save('../data/ball_trajectory.npy', ball_traj)
  np.save('../data/parameters.npy', par)

def bounded_trajectory():
  simtime = 40
  cone_side_coef = -np.tan(np.deg2rad(2))
  ball_radius = 0.06
  angvel_initial = 5.
  angaccel = 0.08
  surf = ConeSurface(cone_side_coef, eps=cone_side_coef**2 * ball_radius**2)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], angvel_initial, angaccel)
  d = Dynamics(par, tablerot)
  x0 = 0.01
  y0 = 0.002
  w0 = np.array([0, 0., 0.])
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(d, [0, simtime], st0, max_step=1e-2)
  save_sim_data(sol.t, sol.y.T, tablerot, par)

  fricforce, normforce = get_auxiliary_signals(d, sol.t, sol.y.T)

  plt.subplot(221)
  plt.plot(sol.y[0,0], sol.y[1,0], 'o')
  plt.plot(sol.y[0], sol.y[1])
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(sol.t, sol.y[2:5].T)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\omega$')
  plt.grid(True)

  plt.subplot(223)
  plt.plot(sol.t, normforce)
  plt.plot(sol.t, fricforce)
  plt.xlabel(R'$t$')
  plt.ylabel(R'normal force')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def periodic_traj_1():
  simtime = 15.69602
  cone_side_coef = -0.1
  ball_radius = 0.06
  angvel_initial = 5.
  angaccel = 0.0
  surf = ConeSurface(cone_side_coef, eps=cone_side_coef**2 * ball_radius**2)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], angvel_initial, angaccel)
  d = Dynamics(par, tablerot)
  x0 = 0.4
  y0 = 0.0
  vec = normalized(surf.coords(x0, y0))
  w0 = vec * 20.6
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(d, [0, simtime], st0, max_step=1e-2)

  plt.subplot(221)
  plt.plot(sol.y[0,0], sol.y[1,0], 'o')
  plt.plot(sol.y[0], sol.y[1])
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(sol.t, sol.y[2:5].T)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\omega$')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def circular_traj():
  simtime = 1.798 * 6
  cone_side_coef = -0.05
  ball_radius = 0.06
  angvel_initial = 5.
  angaccel = 0.0
  surf = ConeSurface(cone_side_coef, eps=cone_side_coef**2 * ball_radius**2)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], angvel_initial, angaccel)
  d = Dynamics(par, tablerot)
  x0 = 0.4
  y0 = 0.0
  vec1 = normalized(surf.coords(x0, y0))
  norm = surf.normal(x0, y0)
  vec2 = np.cross(norm, vec1)
  w0 = vec1 * 23.2396  - vec2 * 0.
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(d, [0, simtime], st0, max_step=1e-2)
  save_sim_data(sol.t, sol.y.T, tablerot, par)

  x = sol.y[0]
  y = sol.y[1]
  w = sol.y[2:5].T

  plt.subplot(221)
  plt.plot(sol.y[0,0], sol.y[1,0], 'o')
  plt.plot(sol.y[0], sol.y[1])
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(sol.t, sol.y[2:5].T)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\omega$')
  plt.grid(True)

  wn = np.zeros(x.shape)
  wp = np.zeros(w.shape)
  rho = np.zeros(x.shape)
  for i in range(len(x)):
    n = surf.normal(x[i], y[i])
    wn[i] = n @ w[i]
    rho[i] = np.sqrt(x[i]**2 + y[i]**2)

  plt.subplot(223)
  plt.plot(sol.t, wn)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\omega \cdot n$')
  plt.grid(True)

  plt.subplot(224)
  plt.plot(sol.t, rho)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$\rho$')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def circular_traj_analysis():
  period = 1.803
  simtime = period * 6
  cone_side_coef = -0.05
  ball_radius = 0.06
  angvel_initial = 5.
  angaccel = 0.0
  surf = ConeSurface(cone_side_coef, eps=cone_side_coef**2 * ball_radius**2)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], angvel_initial, angaccel)
  d = Dynamics(par, tablerot)
  x0 = 0.4
  y0 = 0.0
  vec1 = normalized(surf.coords(x0, y0))
  norm = surf.normal(x0, y0)
  vec2 = np.cross(norm, vec1)
  w0 = vec1 * 23.2396  - vec2 * 0.
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(d, [0, simtime], st0, max_step=1e-2)
  save_sim_data(sol.t, sol.y.T, tablerot, par)

  x = sol.y[0]
  y = sol.y[1]
  w = sol.y[2:5].T

  plt.subplot(111)
  plt.plot(sol.t, sol.y[0], '--')
  plt.plot(sol.t, 0.4 * np.cos(2 * np.pi * sol.t / period))
  plt.plot(sol.t, sol.y[1], '--')
  plt.plot(sol.t, -0.4 * np.sin(2 * np.pi * sol.t / period))
  plt.xlabel('$t$')
  plt.ylabel('$x,y$')
  plt.grid(True)
  plt.gca().ticklabel_format(useOffset=False)

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  circular_traj_analysis()
