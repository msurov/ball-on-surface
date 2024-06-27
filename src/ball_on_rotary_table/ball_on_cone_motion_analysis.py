from ball_on_rotary_table.ball_on_rotating_cone_dynamics import (
  BallOnRotatingConeDynamics,
  BallOnRotatingConeParameters
)
from common.linalg import wedge
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def find_circular_trajectory_ic(par : BallOnRotatingConeParameters, table_angvel : float, radius : float):
  ρ = radius
  ϕ = 0.
  dθ = table_angvel
  r = par.ball_radius
  mr2 = par.ball_mass * r**2
  k = mr2 / (mr2 + par.ball_inertia)
  tan_α = np.tan(par.cone_side_angle)
  g = par.gravity_accel
  ζn = 0.
  ζϕ = 0.
  D = ρ**2 * dθ**2 * (k - 1)**2 + 4 * ρ * g * k * tan_α
  assert D >= 0
  ζρ = (ρ*dθ*(k + 1) - np.sqrt(D)) / (2 * r)
  period = 2 * np.pi * ρ / (ζρ * r)
  return np.array([ρ, ϕ, ζρ, ζϕ, ζn]), period

def compute_traj():
  cone_angle = -np.deg2rad(2)
  ball_radius = 0.08
  gravity_accel = 9.81
  ball_mass = 0.050
  par = BallOnRotatingConeParameters(
    cone_side_angle = cone_angle,
    gravity_accel = gravity_accel,
    ball_mass = ball_mass,
    ball_radius = ball_radius
  )

  traj_radius = 0.5
  table_angvel = 8.0
  st_ref, period = find_circular_trajectory_ic(par, table_angvel, traj_radius)
  print('radius', traj_radius)
  print('table speed', table_angvel)
  print('ref state', st_ref)
  dnx = BallOnRotatingConeDynamics(par)
  sys = lambda t, st: dnx(st, table_angvel, 0.)
  sol = solve_ivp(sys, [0, 2*period], st_ref, max_step=1e-2)

  plt.subplot(221)
  plt.grid(True)
  plt.plot(sol.t, sol.y[0])
  plt.ylabel('ball radial coord')

  plt.subplot(222)
  plt.grid(True)
  plt.plot(sol.t, sol.y[1])
  plt.ylabel('ball polar coord')

  plt.subplot(223)
  plt.grid(True)
  plt.plot(sol.t, sol.y[2:5].T)
  plt.ylabel('ball ang vel')

  plt.subplot(224)
  plt.grid(True)
  plt.plot(sol.t[[0,-1]], [table_angvel, table_angvel])
  
  plt.tight_layout()
  plt.show()

def compute_traj():
  cone_angle = -np.deg2rad(2)
  r = 0.08
  g = 9.81
  m = 0.050
  par = BallOnRotatingConeParameters(
    cone_side_angle = cone_angle,
    gravity_accel = g,
    ball_mass = m,
    ball_radius = r
  )

  Ω = 5.0
  M = par.ball_inertia
  dnx = BallOnRotatingConeDynamics(par)
  sys = lambda t, st: dnx(st, Ω, 0.)
  np.random.seed(0)
  state_initial = np.random.normal(size=5)
  sol = solve_ivp(sys, [0, 5], state_initial, max_step=1e-2)

  def f(st):
    r = par.ball_radius
    ρ,ϕ = st[0:2]
    ζ = st[2:5]
    ζρ = ζ[0]
    K = r / ρ * ζρ - Ω
    return K

  vals = [f(e) for e in sol.y.T]

  plt.subplot(211)
  plt.plot(sol.t, vals)
  plt.ylabel('motion integral')
  plt.subplot(212)
  plt.plot(sol.t, sol.y[0])
  plt.show()

if __name__ == '__main__':
  compute_traj()
