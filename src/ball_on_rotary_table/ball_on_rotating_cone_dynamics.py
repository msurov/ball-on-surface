from dataclasses import dataclass
import numpy as np
from common.surface import Surface
from common.quat import quat_rot, quat_conj
from common.linalg import wedge
from typing import Callable
from .dynamics import SystemParameters, cross, dot, outer


@dataclass 
class BallOnRotatingConeParameters:
  cone_side_angle : float # angle between two opposite generatrix line
  gravity_accel : float
  ball_mass : float
  ball_radius : float
  ball_inertia : float = None

  def __post_init__(self):
    if self.ball_inertia is None:
      self.ball_inertia = self.ball_mass * self.ball_radius**2 * 2 / 3

class BallOnRotatingConeDynamics:
  def __init__(self, par : BallOnRotatingConeParameters) -> None:
    self.par = par

  def __call__(self, st, dθ, ddθ):
    return self.dynamics(st, dθ, ddθ)

  def dynamics(self, state, dθ, ddθ):
    R"""
      :param state: is composed from [ρ,ϕ,ζ], 
        where ρ,ϕ are ball cylindrical cordinates 
        and ζ is ball angular velocity in frame [eρ,eϕ,en]
      :param dθ: table angular velocity
      :param ddθ: table angular acceleration
    """
    g = self.par.gravity_accel
    r = self.par.ball_radius
    M = self.par.ball_inertia
    m = self.par.ball_mass
    α = self.par.cone_side_angle
    I = np.eye(3)
    ex,ey,ez = I
    ez_x = wedge(ez)
    k = m*r**2 / (m*r**2 + M)

    ρ,ϕ = state[0:2]
    ζ = state[2:5]

    sin_α = np.sin(α)
    cos_α = np.cos(α)

    T = np.array([
      [0, cos_α, 0],
      [-1/ρ, 0, 0]
    ])
    dρ,dϕ = r * T @ ζ

    a1 = -g*k/r * sin_α * ey
    a2 = k * cos_α / r * ey
    A3 = np.array([
      [0, (k + 1) * cos_α, 0],
      [-(k + 1) * cos_α, 0, (1 - k) * sin_α],
      [0, -sin_α, 0 ]
    ])
    A4 = np.array([
      [0, cos_α, 0],
      [-cos_α, 0, (1 - k) * sin_α],
      [0, -sin_α, 0 ]
    ])
    a5 = np.array([
      (k - 1) * sin_α,
      0,
      -cos_α
    ])
    a6 = ex * k / r
    dζ = a1 + ρ * a2 * dθ**2 + dθ * A3 @ ζ \
      + dϕ * A4 @ ζ + (a5 + ρ * a6) * ddθ

    dstate = np.concatenate(([dρ, dϕ], dζ))
    return dstate
