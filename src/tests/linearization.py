import casadi as ca
from ball_on_rotary_table.ball_on_rotating_cone_dynamics import (
  BallOnRotatingConeParameters,
  BallOnRotatingConeDynamics
)
import numpy as np

def wedge(a):
  x,y,z = a.elements()
  return ca.vertcat(
    ca.horzcat(0, -z, y),
    ca.horzcat(z, 0, -x),
    ca.horzcat(-y, x, 0)
  )

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

def dynamics(par : BallOnRotatingConeParameters):
  g = par.gravity_accel
  r = par.ball_radius
  M = par.ball_inertia
  m = par.ball_mass
  α = par.cone_side_angle
  I = ca.DM.eye(3)
  ex = I[:,0]
  ey = I[:,1]
  ez = I[:,2]
  ez_x = wedge(ez)
  k = m*r**2 / (m*r**2 + M)

  state = ca.SX.sym('state', 5)
  dθ = ca.SX.sym('dtheta')
  ddθ = ca.SX.sym('ddtheta')

  ρ = state[0]
  ϕ = state[1]
  ζ = state[2:5]

  sin_α = ca.sin(α)
  cos_α = ca.cos(α)
  T = ca.vcat([
    ca.hcat([0, cos_α, 0]),
    ca.hcat([-1/ρ, 0, 0])
  ])
  tmp = r * T @ ζ
  dρ = tmp[0]
  dϕ = tmp[1]

  a1 = -g*k/r * sin_α * ey
  a2 = k * cos_α / r * ey
  A3 = ca.vcat([
    ca.hcat([0, (k + 1) * cos_α, 0]),
    ca.hcat([-(k + 1) * cos_α, 0, (1 - k) * sin_α]),
    ca.hcat([0, -sin_α, 0 ])
  ])
  A4 = ca.vcat([
    ca.hcat([0, cos_α, 0]),
    ca.hcat([-cos_α, 0, (1 - k) * sin_α]),
    ca.hcat([0, -sin_α, 0 ])
  ])
  a5 = ca.vcat([
      (k - 1) * sin_α,
      0,
      -cos_α
    ])
  a6 = ex * k / r
  dζ = a1 + ρ * a2 * dθ**2 + dθ * A3 @ ζ  \
    + dϕ * A4 @ ζ + (a5 + ρ * a6) * ddθ

  d = ca.vertcat(dρ, dϕ, dζ)
  fun = ca.Function('dynamics', [state, dθ, ddθ], [d])
  return fun

def linearize_circular_trajectory(par : BallOnRotatingConeParameters, table_angvel : float, traj_radius : float):
  dynamics_fun = dynamics(par)
  st_ref, _ = find_circular_trajectory_ic(par, table_angvel, traj_radius)
  ρ_ref = st_ref[0]
  ϕ_ref = 0
  ζ_ref = st_ref[2:5]
  Ω_ref = table_angvel
  δ = ca.SX.sym('delta', 5)
  u = ca.SX.sym('u', 1)
  δΩ = δ[0]
  δρ = δ[1]
  δζ = δ[2:5]
  st = ca.vertcat(
    ρ_ref + δρ,
    0,
    ζ_ref + δζ
  )
  f = dynamics_fun(st, Ω_ref + δΩ, 0)

  A11 = ca.DM.zeros(1, 1)
  A12 = ca.DM.zeros(1, 1)
  A13 = ca.DM.zeros(1, 3)

  fρ = ca.horzcat(1, 0, 0, 0, 0) @ f
  A21 = ca.substitute(ca.jacobian(fρ, δΩ), δ, 0)
  A22 = ca.substitute(ca.jacobian(fρ, δρ), δ, 0)
  A23 = ca.substitute(ca.jacobian(fρ, δζ), δ, 0)

  fζ = ca.DM([
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
  ]) @ f
  A31 = ca.substitute(ca.jacobian(fζ, δΩ), δ, 0)
  A32 = ca.substitute(ca.jacobian(fζ, δρ), δ, 0)
  A33 = ca.substitute(ca.jacobian(fζ, δζ), δ, 0)

  A = ca.vertcat(
    ca.horzcat(A11, A12, A13),
    ca.horzcat(A21, A22, A23),
    ca.horzcat(A31, A32, A33)
  )
  A = ca.DM(A)
  A = np.array(A)
  
  g = dynamics_fun(st_ref, Ω_ref, u)
  g = ca.substitute(ca.jacobian(g, u), u, 0)
  B = ca.DM([
    1,
    g[0],
    g[2],
    g[3],
    g[4]
  ])
  B = np.array(B)
  return A, B
