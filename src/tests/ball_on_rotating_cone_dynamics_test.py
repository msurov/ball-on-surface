import src.ball_on_rotary_surface.ball_on_rotary_surface_dynamics as ball_on_rotary_surface_dynamics
from src.ball_on_rotary_surface.ball_on_rotary_cone_dynamics import (
  BallOnRotaryConeDynamics,
  BallOnRotatingConeParameters
)
from common.linalg import wedge
from common.surface import ConeSurface
from common.rotations import angleaxis, solve_poisson_kinematics
from common.quat import quat_rot, quat_conj
from common.frame_rotation import (
  FrameAccelRot
)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def cart2cyl_state(st, par : BallOnRotatingConeParameters):
  x,y = st[0:2]
  w = st[2:5]
  ρ = np.sqrt(x**2 + y**2)
  α = par.cone_side_angle
  ϕ = np.arctan2(y, x)
  sin_ϕ = np.sin(ϕ)
  cos_ϕ = np.cos(ϕ)
  sin_α = np.sin(α)
  cos_α = np.cos(α)
  A = np.array([
    [cos_ϕ * cos_α, sin_ϕ * cos_α, sin_α],
    [-sin_ϕ, cos_ϕ, 0],
    [-cos_ϕ * sin_α, -sin_ϕ * sin_α, cos_α]
  ])
  ζ = A @ w
  newst = np.concatenate(([ρ, ϕ], ζ))
  return newst

def cyl2cart(st, par : BallOnRotatingConeParameters):
  ρ,ϕ = st[0:2]
  ζ = st[2:5]
  sin_ϕ = np.sin(ϕ)
  cos_ϕ = np.cos(ϕ)
  α = par.cone_side_angle
  sin_α = np.sin(α)
  cos_α = np.cos(α)
  A = np.array([
    [cos_ϕ * cos_α, sin_ϕ * cos_α, sin_α],
    [-sin_ϕ, cos_ϕ, 0],
    [-cos_ϕ * sin_α, -sin_ϕ * sin_α, cos_α]
  ])
  ω = A.T @ ζ
  x = ρ * cos_ϕ
  y = ρ * sin_ϕ
  return np.array([x, y, *ω])

def test():
  cone_side_coef = -0.2
  ball_radius = 0.08
  angvel_initial = 2.2
  angaccel = 0.7
  surf = ConeSurface(cone_side_coef, eps=1e-5)
  par = ball_on_rotary_surface_dynamics.SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.07,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], angvel_initial, angaccel)
  d1 = ball_on_rotary_surface_dynamics.Dynamics(par, tablerot)
  st0 = np.random.normal(size=5)
  sol1 = solve_ivp(d1, [0., 5.], st0, max_step=1e-2)
  x1 = sol1.y[0]
  y1 = sol1.y[1]

  par2 = BallOnRotatingConeParameters(
    cone_side_angle = np.arctan(cone_side_coef),
    gravity_accel = par.gravity_accel,
    ball_mass = par.ball_mass,
    ball_radius = par.ball_radius,
    ball_inertia = par.ball_inertia,
  )
  d2 = BallOnRotaryConeDynamics(par2)
  def rhs(t, st):
    angvel = angvel_initial + angaccel * t
    return dnx(t, st, angvel, angaccel)

  sol2 = solve_ivp(rhs, [sol1.t[0], sol1.t[-1]], cart2cyl_state(st0, par2), max_step=1e-2, t_eval=sol1.t)
  ρ = sol2.y[0]
  ϕ = sol2.y[1]
  x2 = ρ * np.cos(ϕ)
  y2 = ρ * np.sin(ϕ)
  t = sol2.t

  plt.subplot(211)
  plt.plot(t, x1, '--', lw=2)
  plt.plot(t, x2)
  plt.grid(True)
  plt.subplot(212)
  plt.plot(t, y1, '--', lw=2)
  plt.plot(t, y2)
  plt.grid(True)
  plt.show()

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

def test_circular_trajectory():
  cone_angle = -np.deg2rad(2)
  ball_radius = 0.08
  table_angvel = 5.
  table_angaccel = 0.
  ball_mass = 0.050
  gravity_accel = 9.81
  traj_radius = 0.3

  par = BallOnRotatingConeParameters(
    cone_side_angle = cone_angle,
    gravity_accel = gravity_accel,
    ball_mass = ball_mass,
    ball_radius = ball_radius
  )
  st0, period = find_circular_trajectory_ic(par, table_angvel, traj_radius)
  st0 = cyl2cart(st0, par)

  par2 = ball_on_rotary_surface_dynamics.SystemParameters(
    surface = ConeSurface(np.tan(cone_angle), eps=1e-5),
    gravity_accel = gravity_accel,
    ball_mass = ball_mass,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], table_angvel, table_angaccel)
  dnx2 = ball_on_rotary_surface_dynamics.Dynamics(par2, tablerot)
  sol1 = solve_ivp(dnx2, [0., period], st0, max_step=1e-2)
  x = sol1.y[0]
  y = sol1.y[1]
  plt.plot(x, y)
  plt.show()

def test_invariant():
  cone_angle = -np.deg2rad(2)
  ball_radius = 0.08
  Ω = 5.
  table_angaccel = 0.
  ball_mass = 0.050
  gravity_accel = 9.81

  par = ball_on_rotary_surface_dynamics.SystemParameters(
    surface = ConeSurface(np.tan(cone_angle), eps=1e-5),
    gravity_accel = gravity_accel,
    ball_mass = ball_mass,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], Ω, table_angaccel)
  dnx = ball_on_rotary_surface_dynamics.Dynamics(par, tablerot)
  st0 = np.random.normal(size=5)
  sol = solve_ivp(dnx, [0., 2.], st0, max_step=1e-3)

  r_tb = sol.y[0:2].T
  ω_ttb = sol.y[2:5].T
  t = sol.t
  q0 = np.array([1., 0., 0., 0.])
  q_tb = solve_poisson_kinematics(t, ω_ttb, q0, 'fixed')
  q_bt = quat_conj(q_tb)
  ez = np.array([0., 0., 1.])
  ω_twb = ω_ttb + ez * Ω
  ω_bwb = quat_rot(q_bt, ω_twb)
  # θ = Ω * t
  # q_wt = np.zeros((len(t), 4))
  # q_wt[:,0] = np.cos(θ/2)
  # q_wt[:,3] = np.sin(θ/2)
  # ω_wwb = quat_rot(q_wt, ω_twb)

  D = par.ball_mass * par.ball_radius**2
  μ = par.ball_inertia
  M = np.zeros((len(t), 3))
  I = np.zeros(len(t))
  R = par.ball_radius

  for i in range(len(t)):
    x, y = r_tb[i]
    γ = par.surface.normal(x, y)
    ω = ω_ttb[i] + ez * Ω
    # table frame:
    r3 = par.surface(x, y)
    k = np.tan(cone_angle)
    ω3 = ω[2]
    σ1 = ω3 + D * Ω * r3 / np.sqrt((1 + k**2) * (μ + D)) / R
    I[i] = quat_rot(q_bt[i], σ1)

  plt.plot(t, I)
  plt.show()

  # qinv = quat_conj(q)
  # quat_rot(qinv, ω)

  # plt.subplot(2, 1, 1)
  # plt.grid(True)
  # plt.plot(sol.t, sol.y[2])
  # plt.subplot(2, 1, 2)
  # plt.grid(True)
  # plt.plot(sol.t, sol.y[0])
  # plt.plot(sol.t, sol.y[1])
  # plt.show()

if __name__ == '__main__':
  np.set_printoptions(suppress=True)
  # test()
  # test2()
  # test_circular_trajectory()
  test_invariant()
