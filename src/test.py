from scipy.integrate import solve_ivp
from dynamics import Dynamics, SystemParameters
from surface import ParaboloidSurface
import numpy as np
import matplotlib.pyplot as plt


def test():
  surf = ParaboloidSurface(kx = 0.05, ky = 0)
  par = SystemParameters(
    surf, 9.81, 0.05, 0.1
  )
  d = Dynamics(par)
  st0 = np.array([1, 0, 0, 0, 0], float)
  sol = solve_ivp(lambda _,st: d(st), [0, 5], st0, max_step=1e-3)
  t = sol.t
  x = sol.y[0]
  y = sol.y[1]
  w = sol.y[2:5].T
  plt.plot(t, x)
  plt.show()


test()
