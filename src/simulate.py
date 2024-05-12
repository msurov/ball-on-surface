import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from quat import quat_vec_mul
from dynamics import SystemParameters, Dynamics, compute_velocity, compute_ball_position
from trajectory import Trajectory


def compute_ball_orietnation(par : SystemParameters, t : np.ndarray, w : np.ndarray):
  q0 = np.array([1., 0., 0., 0.])
  wsp = make_interp_spline(t, w)
  def rhs(t, q):
    dq = 0.5 * quat_vec_mul(wsp(t), q)
    return dq
  sol = solve_ivp(rhs, [t[0], t[-1]], q0, t_eval=t)
  return sol.y.T

def simulate(par : SystemParameters, x0 : float, y0 : float, w0 : np.ndarray, sim_interval : float):
  d = Dynamics(par)
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(lambda _,st: d(st), [0, sim_interval], st0, max_step=1e-2)
  st = sol.y.T
  v = compute_velocity(par, st)
  p = compute_ball_position(par, st)
  w = st[:,2:5]
  q = compute_ball_orietnation(par, sol.t, w)
  energy = np.array([d.full_energy(e) for e in st])
  return Trajectory(sol.t, p, v, q, w), energy
