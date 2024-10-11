import numpy as np
import matplotlib.pyplot as plt   # type: ignore
from scipy.integrate import solve_ivp   # type: ignore
from common.surface import Plane, ConeSurface
from common.quat import quat_rot, quat_mul
from common.rotations import solve_poisson_kinematics
from common.trajectory import RigidBodyTrajectory
import ball_on_rotary_surface.ball_on_rotary_plane_dynamics as plane_dynamics
import ball_on_rotary_surface.ball_on_rotary_surface_dynamics as surf_dynamics
import ball_on_rotary_surface.ball_on_rotary_cone_dynamics as cone_dynamics
from ball_on_rotary_surface.parameters import BallOnRotarySurfaceParameters
from common.frame_rotation import (
  FrameRotation,
  FrameAccelRot
)
from copy import deepcopy
from common.rotations import rotate_vec


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

def compute_traj_v1(ic : np.ndarray, simtime : float, tablespeed : float, par : BallOnRotarySurfaceParameters):
  R"""
    Dynamics on Rotarty Plane
  """
  dynamics = plane_dynamics.BallOnRotaryPlaneDynamics(par)
  def rhs(t, st):
    return dynamics(st, tablespeed, 0.)

  ρφζ0 = plane_dynamics.transform(xyω=ic)
  sol = solve_ivp(rhs, [0, simtime], ρφζ0, max_step=1e-2)
  return sol.t, plane_dynamics.transform(ρφζ=sol.y.T)

def compute_traj_v2(ic : np.ndarray, simtime : float, tablespeed : float, par : BallOnRotarySurfaceParameters):
  R"""
    Dynamics on Rotarty Cone
  """
  par = deepcopy(par)
  par.surface = ConeSurface(0.)

  dynamics = cone_dynamics.BallOnRotaryConeDynamics(par)
  def rhs(t, st):
    return dynamics(st, tablespeed, 0.)

  ρφζ0 = cone_dynamics.transform(xyω=ic)
  sol = solve_ivp(rhs, [0, simtime], ρφζ0, max_step=1e-2)
  return sol.t, np.array([cone_dynamics.transform(ρφζ=e) for e in sol.y.T])

def compute_traj_v3(ic : np.ndarray, simtime : float, tablespeed : float, par : BallOnRotarySurfaceParameters):
  R"""
    Dynamics on Rotarty Surface
  """
  tablerot = FrameAccelRot([0, 0, 1], tablespeed, 0.)
  rhs = surf_dynamics.BallOnRotarySurfaceDynamics(par, tablerot)
  sol = solve_ivp(rhs, [0, simtime], ic, max_step=1e-2)
  return sol.t, sol.y.T

def compare_dynamics():
  simtime = 5
  table_angspeed = 5.
  surf = Plane()
  par = BallOnRotarySurfaceParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = 0.07,
  )
  ic = [0.1, 0.2, 0.3, -0.2, 0.1]
  t1, st1 = compute_traj_v1(ic, simtime, table_angspeed, par)
  t2, st2 = compute_traj_v2(ic, simtime, table_angspeed, par)
  t3, st3 = compute_traj_v3(ic, simtime, table_angspeed, par)
  x1 = st1[:,0]
  y1 = st1[:,1]
  x2 = st2[:,0]
  y2 = st2[:,1]
  x3 = st3[:,0]
  y3 = st3[:,1]

  plt.plot(x1, y1, lw=3, ls='--', label='plane')
  plt.plot(x2, y2, lw=2, ls='--', label='cone')
  plt.plot(x3, y3, lw=1, ls='--', label='surf')

  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()

def fxied_frame_trajectory():
  simtime = 30
  table_angspeed = 0.1
  surf = Plane()
  par = BallOnRotarySurfaceParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = 0.5,
  )
  ic = [20.2, -0.5, 2.3, 0.6, 3.]
  t, st = compute_traj_v2(ic, simtime, table_angspeed, par)
  x = st[:,0]
  r = np.zeros(x.shape + (3,))
  r[:,0] = st[:,0]
  r[:,1] = st[:,1]
  r[:,2] = par.ball_radius
  table_angle = table_angspeed * t
  ball_pos = rotate_vec([0., 0., 1.], table_angle, r)

  d = np.max(ball_pos, axis=0) - np.min(ball_pos, axis=0)
  print(f'axes difference: {(d[0] - d[1]) / d[0]}')
  
  plt.figure('ball on rotary plane')
  plt.axis('equal')
  plt.plot(ball_pos[:,0], ball_pos[:,1])
  plt.plot(ball_pos[0,0], ball_pos[0,1], 'o')
  plt.grid(True)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.axhline(0, color='gray')
  plt.axvline(0, color='gray')
  plt.tight_layout()
  plt.show()
