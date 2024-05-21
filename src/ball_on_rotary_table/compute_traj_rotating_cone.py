import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from common.surface import ConeSurface
from common.math import integrate_table
from common.quat import quat_rot, quat_mul
from common.rotations import solve_poisson_kinematics
from common.trajectory import RigidBodyTrajectory
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
  q_table_ball = solve_poisson_kinematics(t, st[:,2:5], np.array([1., 0., 0., 0.]), 'fixed' )
  q_world_ball = np.zeros((n, 4))
  p_world_ball = np.zeros((n, 3))
  q_world_table = np.zeros((n, 4))
  p_world_table = np.zeros((n, 3))

  for i in range(n):
    q_world_table[i,:] = tabrot.rot(t[i])
    p_table_ball = par.surface.coords(*st[i, 0:2])
    p_world_ball[i,:] = quat_rot(q_world_table[i,:], p_table_ball)
    q_world_ball[i,:] = quat_mul(q_world_table[i,:], q_table_ball[i,:])

  ball_traj = RigidBodyTrajectory(
    t = t,
    p = p_world_ball,
    q = q_world_ball
  )
  table_traj = RigidBodyTrajectory(
    t = t,
    p = p_world_table,
    q = q_world_table
  )
  return ball_traj, table_traj

def get_auxiliary_signals(dynamics : Dynamics, t : np.ndarray, st : np.ndarray):
  n = len(t)
  friction_force = np.zeros((n, 3))
  normal_force = np.zeros((n,))
  energy = np.zeros((n,))
  power = np.zeros((n,))
  for i in range(n):
    friction_force[i,:] = dynamics.friction_force(t[i], st[i])
    normal_force[i] = dynamics.normal_force(t[i], st[i])
    energy[i] = dynamics.hamiltonian(t[i], st[i])
    power[i] = dynamics.hamiltonian_time_deriv(t[i], st[i])
  energy_change = integrate_table(t, power)
  return friction_force, normal_force, energy, energy_change

def main():
  simtime = 5
  surf = ConeSurface(np.tan(np.deg2rad(1e-6)), eps=0.13)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 0.,
    ball_mass = 1.,
    ball_radius = 1.,
    ball_inertia = 1.
  )
  tablerot = FrameAccelRot([0, 0, 1], 1., 0.05)
  d = Dynamics(par, tablerot)
  x0 = 0.1
  y0 = 0.2
  w0 = np.array([0, 0., 0.])
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(d, [0, simtime], st0, max_step=1e-2)

  ball_traj, table_traj = get_bodies_traj(sol.t, sol.y.T, tablerot, par)

  np.save('../data/table_trajectory.npy', table_traj)
  np.save('../data/ball_trajectory.npy', ball_traj)
  np.save('../data/parameters.npy', par)

  fricforce, normforce, energy, energy_change = get_auxiliary_signals(d, sol.t, sol.y.T)

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

  plt.subplot(224)
  plt.plot(sol.t, energy - energy_change)
  plt.xlabel(R'$t$')
  plt.ylabel(R'$H_0 = H - \int H_t dt$')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def test():
  simtime = 5.
  surf = ConeSurface(np.tan(np.deg2rad(1e-6)), eps=0.13)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 0.,
    ball_mass = 1.,
    ball_radius = 1.,
    ball_inertia = 1.
  )
  tablerot = FrameAccelRot([0, 0, 1], 0.3, 0.0)
  d = Dynamics(par, tablerot)
  x0 = 1.0
  y0 = 0.0
  w0 = np.array([0., 0., 0.])
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(d, [0, simtime], st0, max_step=1e-2)
  t = sol.t
  w = sol.y[2:5].T
  x = sol.y[0]
  y = sol.y[1]
  plt.figure('manual')
  plt.plot(t, w)
  plt.legend(['wx', 'wy', 'wz'])
  plt.grid(True)
  plt.show()

def test2():
  np.set_printoptions(suppress=True)

  table_angvel = 0.3
  table_angacc = 0.08
  m = 0.1
  M = 0.07
  r = 0.18
  q0 = np.array([1., 0., 0., 0.])
  p0 = np.array([0.3, 0.7])
  w0 = np.array([1., 3., 2.])

  surf = ConeSurface(np.tan(np.deg2rad(1e-8)), eps=1e-8)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 0.,
    ball_mass = m,
    ball_radius = r,
    ball_inertia = M
  )
  tablerot = FrameAccelRot([0., 0., 1.], table_angvel, table_angacc)
  d = Dynamics(par, tablerot)
  st0 = np.concatenate((p0, w0))
  dst0 = d(0., st0)
  dw0 = dst0[2:5]
  print('dw:', dw0)

if __name__ == '__main__':
  test2()
