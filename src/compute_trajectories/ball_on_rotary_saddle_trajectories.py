import numpy as np
import matplotlib.pyplot as plt   # type: ignore
from scipy.integrate import solve_ivp   # type: ignore
from common.surface import ParaboloidSurface
from common.quat import quat_rot, quat_mul
from common.integrate import integrate_table
from common.rotations import solve_poisson_kinematics
from common.trajectory import RigidBodyTrajectory
from ball_on_rotary_surface.ball_on_rotary_surface_dynamics import (
  BallOnRotarySurfaceParameters,
  BallOnRotarySurfaceDynamics
)
from common.frame_rotation import (
  FrameRotation,
  OscillatingFrame,
  FrameConstRot,
  FrameAccelRot
)


def get_bodies_traj(t : np.ndarray, st : np.ndarray, tabrot : FrameRotation, par : BallOnRotarySurfaceParameters):
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

def get_auxiliary_signals(dynamics : BallOnRotarySurfaceDynamics, t : np.ndarray, st : np.ndarray):
  n = len(t)
  friction_force = np.zeros((n, 3))
  normal_force = np.zeros((n,))
  for i in range(n):
    friction_force[i,:] = dynamics.friction_force(t[i], st[i])
    normal_force[i] = dynamics.normal_force(t[i], st[i])
  return friction_force, normal_force

def main():
  simtime = 20
  angvel_initial = 2.
  angaccel = 0.17
  surf = ParaboloidSurface(0.3, -0.3)
  par = BallOnRotarySurfaceParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.03,
    ball_radius = 0.04
  )
  tablerot = FrameAccelRot([0., 0., 1.], angvel_initial, angaccel)
  d = BallOnRotarySurfaceDynamics(par, tablerot)
  x0 = -0.2
  y0 = 0.06
  w0 = np.array([0., 0., 5.])
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(d, [0, simtime], st0, max_step=1e-2)

  ball_traj, table_traj = get_bodies_traj(sol.t, sol.y.T, tablerot, par)

  np.save('./data/table_trajectory.npy', table_traj)
  np.save('./data/ball_trajectory.npy', ball_traj)
  np.save('./data/parameters.npy', par)

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
  plt.ylabel(R'ball ang speed')
  plt.grid(True)

  plt.subplot(223)
  plt.plot(sol.t, normforce)
  plt.plot(sol.t, fricforce)
  plt.xlabel(R'$t$')
  plt.ylabel(R'reaction force')
  plt.grid(True)

  plt.subplot(224)
  plt.plot(sol.t, [tablerot.angvel(t) for t in sol.t])
  plt.xlabel(R'$t$')
  plt.ylabel(R'table ang speed')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
