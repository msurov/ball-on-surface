import numpy as np
from typing import Literal
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from .quat import quat_vec_mul, quat_mul_vec

FrameName = Literal['fixed', 'self']

def solve_poisson_kinematics(t : np.ndarray, w : np.ndarray, q0 : np.ndarray, angvel_frame : FrameName = 'self') -> np.ndarray:
  wsp = make_interp_spline(t, w)
  match angvel_frame:
    case 'self': rhs = lambda t,q: 0.5 * quat_mul_vec(q, wsp(t))
    case 'fixed': rhs = lambda t,q: 0.5 * quat_vec_mul(wsp(t), q)

  q0 = np.array(q0) / np.linalg.norm(q0)
  sol = solve_ivp(rhs, [t[0], t[-1]], q0, t_eval=t)
  return sol.y.T
