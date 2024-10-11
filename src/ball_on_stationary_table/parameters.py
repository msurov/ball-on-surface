from dataclasses import dataclass
from common.surface import Surface
from typing import Optional


@dataclass
class BallOnSurfaceParameters:
  surface : Surface
  gravity_accel : float
  ball_mass : float
  ball_radius : float
  ball_inertia : Optional[float] = None

  def __post_init__(self):
    if self.ball_inertia is None:
      self.ball_inertia = 2 * self.ball_mass * self.ball_radius**2 / 5
