import numpy as np
from common.surface import Plane
from .ball_on_rotary_surface_dynamics import BallOnRotarySurfaceParameters
from typing import Optional
from common.rotations import angleaxis, rotate_vec

cross = np.cross
dot = np.dot
outer = np.outer


class BallOnRotaryPlaneDynamics:
  def __init__(self, par : BallOnRotarySurfaceParameters) -> None:
    assert isinstance(par.surface, Plane)
    self.par = par

  def __call__(self, st, dθ, ddθ):
    return self.dynamics(st, dθ, ddθ)

  def dynamics(self, state, dθ, ddθ):
    R"""
      :param state: is composed of [ρ,ϕ,ζ], 
        where ρ,ϕ are ball cylindrical cordinates 
        and ζ is ball angular velocity in frame [eρ,eϕ,en]
      :param dθ: table angular velocity
      :param ddθ: table angular acceleration
    """
    g = self.par.gravity_accel
    r = self.par.ball_radius
    M = self.par.ball_inertia
    m = self.par.ball_mass
    k = m * r**2 / (m * r**2 + M)

    ρ, φ, ζρ, ζφ, ζn = state

    dρ = r * ζφ
    dφ = -r/ρ * ζρ
    dζρ = dθ * (k + 1) * ζφ + dφ * ζφ + k * ρ / r * ddθ
    dζφ = ρ * k / r * dθ**2 - ζρ * (k * dθ + dθ + dφ)
    dζn = -ddθ
    return np.array([
      dρ, dφ, dζρ, dζφ, dζn
    ])

__ey = np.array([0., 1., 0.])
__ez = np.array([0., 0., 1.])

def forward_transform(xyω : Optional[np.ndarray]) -> np.ndarray:
  xyω = np.array(xyω)
  x = xyω[...,0]
  y = xyω[...,1]
  ω = xyω[...,2:5]
  ρ = np.sqrt(x**2 + y**2)
  φ = np.arctan2(y, x)
  ρφζ = np.zeros(xyω.shape)
  ρφζ[...,0] = ρ
  ρφζ[...,1] = φ
  ρφζ[...,2:5] = rotate_vec(-__ez, φ, ω)
  return ρφζ

def backward_transform(ρφζ : Optional[np.ndarray]) -> np.ndarray:
  ρφζ = np.array(ρφζ)
  ρ = ρφζ[...,0]
  φ = ρφζ[...,1]
  ζ = ρφζ[...,2:5]
  xyω = np.zeros(ρφζ.shape)
  xyω[...,0] = ρ * np.cos(φ)
  xyω[...,1] = ρ * np.sin(φ)
  xyω[...,2:5] = rotate_vec(__ez, φ, ζ)
  return xyω

def transform(xyω : Optional[np.ndarray] = None, ρφζ : Optional[np.ndarray] = None):
  """
    Make transformations
    \[
      x&=&\rho\cos\phi\\y&=&\rho\sin\phi \\
      \zeta&=&\mathbf{R}_{\mathbf{e}_{z},-\phi}\omega
    \]
  """
  if xyω is not None:
    return forward_transform(xyω)

  if ρφζ is not None:
    return backward_transform(ρφζ)

  assert False
