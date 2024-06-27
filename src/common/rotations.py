import numpy as np
from typing import Literal
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from common.quat import quat_vec_mul, quat_mul_vec
from common.linalg import wedge
from typing import Callable


FrameName = Literal['fixed', 'self']

def solve_poisson_kinematics(
    t : np.ndarray,
    w : np.ndarray | Callable[[float], float],
    q0 : np.ndarray,
    angvel_frame : FrameName = 'self',
    **integ_par
  ) -> np.ndarray:
  R"""
    Compute quaternion trajectory for a given angular velocity trajectory
  """

  if callable(w):
    wfun = w
  elif isinstance(w, np.ndarray):
    wfun = make_interp_spline(t, w)
  else:
    assert False

  match angvel_frame:
    case 'self': rhs = lambda t,q: 0.5 * quat_mul_vec(q, wfun(t))
    case 'fixed': rhs = lambda t,q: 0.5 * quat_vec_mul(wfun(t), q)
    case _: assert False

  q0 = np.array(q0) / np.linalg.norm(q0)
  sol = solve_ivp(rhs, [t[0], t[-1]], q0, t_eval=t, **integ_par)
  return sol.y.T

def angleaxis(axis : np.ndarray, angle : float):
  l = axis / np.linalg.norm(axis)
  l_x = wedge(l)
  R = I + l_x * np.sin(angle) + (1 - np.cos(angle)) * l_x @ l_x
  return R
