import ball_on_rotary_table.dynamics as dynamics
import ball_on_rotary_table.dynamics_simplified as dynamics_simpl
from common.surface import ConeSurface
from common.frame_rotation import (
  FrameAccelRot
)
import numpy as np


def test():
  cone_side_coef = -0.1
  ball_radius = 0.06
  angvel_initial = 5.0
  angaccel = 0.624353
  surf = ConeSurface(cone_side_coef, eps=1e-5)
  par = dynamics.SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = ball_radius,
  )
  tablerot = FrameAccelRot([0, 0, 1], angvel_initial, angaccel)
  d1 = dynamics.Dynamics(par, tablerot)
  angvel_fun = lambda t, nder: np.polyval(np.polyder([angaccel, angvel_initial], nder), t)
  d2 = dynamics_simpl.Dynamics(par, angvel_fun)
  t = 12.354
  st = np.random.normal(size=5)
  print(d1(t, st) - d2(t, st))

if __name__ == '__main__':
  np.set_printoptions(suppress=True)
  np.random.seed(0)
  test()
