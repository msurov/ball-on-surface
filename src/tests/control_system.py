from ball_on_rotary_table.ball_on_rotating_cone_dynamics import (
  BallOnRotatingConeDynamics,
  BallOnRotatingConeParameters
)
from common.linalg import wedge
from common.surface import ConeSurface
from common.rotations import angleaxis
from common.lqr import lqr_lti
from common.frame_rotation import (
  FrameAccelRot
)
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from .linearization import linearize_circular_trajectory


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

def construct_feedback(par : BallOnRotatingConeParameters, dθ_ref, ζ_ref, ρ_ref):
  Q = np.diag([1., 100., 1e-2, 1., 1e-8])
  R = 0.03 * np.eye(1)
  A, B = linearize_circular_trajectory(par, dθ_ref, ρ_ref)

  L = np.concatenate((B, A@B, A@A@B, A@A@A@B, A@A@A@A@B), axis=1)
  # U,s,Vt = np.linalg.svd(R)
  # p1 = U[:,3]
  # p2 = U[:,4]
  # print(p1)
  # print(p2)
  evals,evecs = np.linalg.eig(L.T)
  z = np.abs(evals) < 1e-8
  p = evecs[:,z].T
  p[0,:] /= p[0,0]
  p[1,:] /= p[1,0]
  print(p)
  exit()
  # print('controllability matrix')
  # print(L)
  # print('controllability matrix eigvals')
  # print(evals)

  """
  print(A)
  evals, evecs = np.linalg.eig(A)
  print(evals)

  print(B)
  exit()

  exit()
  """

  K, P = lqr_lti(A, B, Q, R)

  def fb(t, full_state):
    dθ = full_state[0]
    ρ = full_state[1]
    ζ = full_state[3:6]
    δdθ = dθ - dθ_ref
    δρ = ρ - ρ_ref
    δζ = ζ - ζ_ref
    z = np.array([δdθ, δρ, *δζ])
    u = K @ z
    return u

  return fb

def simulate(dynamics, fb, ic, table_angvel_initial, st_ref):
  def rhs(t, full_state):
    dθ = full_state[0]
    ball_state = full_state[1:]
    ddθ = fb(t, full_state)
    dball_state = dynamics(ball_state, dθ, ddθ)
    dst = np.concatenate([ddθ, dball_state])
    return dst

  full_state_initial = np.array([table_angvel_initial, *ic])
  np.random.seed(0)
  full_state_initial[1] += 0.1 * np.random.normal()
  sol = solve_ivp(rhs, [0, 15], full_state_initial, max_step=1e-2)

  dθ = sol.y[0]
  ρ = sol.y[1]
  ϕ = sol.y[2]
  x = np.cos(ϕ) * ρ
  y = np.sin(ϕ) * ρ
  ρ_ref = st_ref[0]
  ζ_ref = st_ref[2:5]
  x_ref = np.cos(ϕ) * ρ_ref
  y_ref = np.sin(ϕ) * ρ_ref
  ζ = sol.y[3:6].T
  # plt.plot(x_ref, y_ref, '--', lw=2, color='red')
  # plt.plot(x, y)
  # plt.plot(x[0], y[0], 'o')
  # plt.show()

  plt.subplot(221)
  plt.plot(sol.t, ρ)
  plt.plot([sol.t[0], sol.t[-1]], [ρ_ref, ρ_ref], '--')
  plt.xlabel('t')
  plt.ylabel('ball radial coord')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(sol.t, ζ)
  plt.plot([sol.t[0], sol.t[-1]], [ζ_ref, ζ_ref], '--')
  plt.xlabel('t')
  plt.ylabel('ang vel')
  plt.grid(True)

  plt.subplot(223)
  plt.plot(x, y)
  plt.plot(x[0], y[0], 'o')
  plt.plot(x_ref, y_ref, '--')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)

  plt.subplot(224)
  plt.plot(sol.t, sol.y[0])
  plt.plot([sol.t[0], sol.t[-1]], [table_angvel_initial, table_angvel_initial], '--')
  plt.xlabel('t')
  plt.ylabel('table angvel')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

def test():
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
  table_angvel = 7.0
  st_ref, period = find_circular_trajectory_ic(par, table_angvel, traj_radius)
  print('radius', traj_radius, '\ntable speed', table_angvel, '\nref state', st_ref)
  dnx = BallOnRotatingConeDynamics(par)
  fb = construct_feedback(par, table_angvel, st_ref[2:5], traj_radius)
  simulate(dnx, fb, st_ref, table_angvel, st_ref)

def test2():
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
  table_angvel = 5.0
  table_angvel2 = 9.0
  st_ref, period = find_circular_trajectory_ic(par, table_angvel, traj_radius)
  st_ref2, period = find_circular_trajectory_ic(par, table_angvel2, traj_radius)
  dnx = BallOnRotatingConeDynamics(par)

  def sys(t, st):
    dθ = st[0]
    ball_st = st[1:]
    ddθ = 0.2 if dθ < table_angvel2 else 0.
    dball_st = dnx(ball_st, dθ, ddθ)
    return np.concatenate(([ddθ], dball_st))

  st = np.concatenate(([table_angvel], st_ref))
  sol = solve_ivp(sys, [0, 30], st, max_step=1e-3)
  dθ = sol.y[0]
  ρ = sol.y[1]
  ϕ = sol.y[2]
  plt.subplot(211)
  plt.plot(sol.t, ρ)
  plt.axhline(st_ref[0])
  plt.axhline(st_ref2[0])
  plt.ylabel('ball radial coordinate')
  plt.subplot(212)
  plt.plot(sol.t, dθ)
  plt.ylabel('table ang speed')
  plt.show()

def solve_linsys(A, tspan, x0, **kwargs):
  def sys(t, x):
    return A @ x
  sol = solve_ivp(sys, tspan, x0, **kwargs)
  return sol.t, sol.y.T

def test3():
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

  traj_radius = 0.4
  table_angvel = 5.0
  st_ref, period = find_circular_trajectory_ic(par, table_angvel, traj_radius)
  dnx = BallOnRotatingConeDynamics(par)

  np.random.seed(0)
  δ = 0.02 * np.random.normal(size=5)
  δ[1] = 0.
  x0 = st_ref + δ
  def sys(t, st):
    return dnx(st, table_angvel, 0.)
  sol = solve_ivp(sys, [0, 1], x0, max_step=1e-2)
  ρ1 = sol.y[0]
  ϕ1 = sol.y[1]
  ζ1 = sol.y[2:5].T

  z0 = np.array([0, δ[0], δ[1], δ[1], δ[3]])
  A, B = get_linearization_v2()
  _,y = solve_linsys(A, [sol.t[0], sol.t[-1]], z0, t_eval=sol.t)
  ρ2 = st_ref[0] + y[:,1]
  ζ2 = st_ref[2:5] + y[:,2:5]

  plt.subplot(211)
  plt.gca().set_prop_cycle(None)
  plt.plot(sol.t, ζ1, lw=2, ls='--')
  plt.gca().set_prop_cycle(None)
  plt.plot(sol.t, ζ2)

  plt.subplot(212)
  plt.gca().set_prop_cycle(None)
  plt.plot(sol.t, ρ1, lw=2, ls='--')
  plt.gca().set_prop_cycle(None)
  plt.plot(sol.t, ρ2)

  plt.show()


np.set_printoptions(suppress=True, linewidth=200)
test()
# test2()
# test3()