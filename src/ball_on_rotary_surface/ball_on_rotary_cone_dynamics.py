import numpy as np
from common.linalg import wedge
from common.surface import ConeSurface
from .ball_on_rotary_surface_dynamics import BallOnRotarySurfaceParameters
from typing import Optional
from common.rotations import angleaxis, rotate_vec

cross = np.cross
dot = np.dot
outer = np.outer


class BallOnRotaryConeDynamics:
  def __init__(self, par : BallOnRotarySurfaceParameters) -> None:
    assert isinstance(par.surface, ConeSurface)
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
    α = self.par.surface.cone_side_angle
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

__ey = np.array([0., 1., 0.])
__ez = np.array([0., 0., 1.])

def forward_transform(xyω : Optional[np.ndarray], cone_side_angle : float = 0.) -> np.ndarray:
  α = float(cone_side_angle)
  xyω = np.array(xyω)
  x = xyω[...,0]
  y = xyω[...,1]
  ω = xyω[...,2:5]
  ρ = np.sqrt(x**2 + y**2)
  φ = np.arctan2(y, x)
  ρφζ = np.zeros(xyω.shape)
  ρφζ[...,0] = ρ
  ρφζ[...,1] = φ
  tmp = rotate_vec(-__ez, φ, ω)
  ρφζ[...,2:5] = rotate_vec(__ey, α, tmp)
  return ρφζ

def backward_transform(ρφζ : Optional[np.ndarray], cone_side_angle : float = 0.) -> np.ndarray:
  α = float(cone_side_angle)
  ρφζ = np.array(ρφζ)
  ρ = ρφζ[...,0]
  φ = ρφζ[...,1]
  ζ = ρφζ[...,2:5]
  xyω = np.zeros(ρφζ.shape)
  xyω[...,0] = ρ * np.cos(φ)
  xyω[...,1] = ρ * np.sin(φ)
  tmp = rotate_vec(-__ey, α, ζ)
  xyω[...,2:5] = rotate_vec(__ez, φ, tmp)
  return xyω

def transform(xyω : Optional[np.ndarray] = None, ρφζ : Optional[np.ndarray] = None, cone_side_angle : float = 0.):
  """
    Make transformations
    \[
      x&=&\rho\cos\phi\\y&=&\rho\sin\phi \\
      \zeta&=&\mathbf{R}_{\mathbf{e}_{z},-\phi}\omega
    \]
  """
  if xyω is not None:
    return forward_transform(xyω, cone_side_angle)

  if ρφζ is not None:
    return backward_transform(ρφζ, cone_side_angle)

  assert False
