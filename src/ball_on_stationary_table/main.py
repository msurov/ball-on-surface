import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt # type: ignore
from scipy.integrate import solve_ivp # type: ignore
from scipy.interpolate import make_interp_spline # type: ignore
from common.quat import quat_vec_mul
from common.surface import ParaboloidSurface
from .dynamics import (
  BallOnSurfaceParameters,
  Dynamics,
  compute_velocity,
  compute_ball_position
)
from common.trajectory import RigidBodyTrajectory

@dataclass
class SimulationResults:
  t : np.ndarray
  p_ball : np.ndarray
  q_ball : np.ndarray
  v_ball : np.ndarray
  w_ball : np.ndarray
  energy : np.ndarray

def compute_ball_orietnation(t : np.ndarray, w : np.ndarray) -> np.ndarray:
  q0 = np.array([1., 0., 0., 0.])
  wsp = make_interp_spline(t, w)
  def rhs(t, q):
    dq = 0.5 * quat_vec_mul(wsp(t), q)
    return dq
  sol = solve_ivp(rhs, [t[0], t[-1]], q0, t_eval=t)
  return sol.y.T

def simulate(par : BallOnSurfaceParameters, x0 : float, y0 : float, w0 : np.ndarray, sim_interval : float) -> SimulationResults:
  d = Dynamics(par)
  st0 = np.concatenate(((x0, y0), w0))
  sol = solve_ivp(lambda _,st: d(st), [0, sim_interval], st0, max_step=1e-2)
  st = sol.y.T
  v = compute_velocity(par, st)
  p = compute_ball_position(par, st)
  w = st[:,2:5]
  q = compute_ball_orietnation(sol.t, w)
  energy = np.array([d.full_energy(e) for e in st])
  return SimulationResults(t=sol.t, p_ball=p, q_ball=q, v_ball=v, w_ball=w, energy=energy)

def main():
  surf = ParaboloidSurface(0.4, -0.13)
  par = BallOnSurfaceParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.1,
    ball_radius = 0.18
  )

  y = 0.5327249411435333
  simres = simulate(par, 0.8, y, [0, 5, 8], 20)

  ball_traj = RigidBodyTrajectory(t=simres.t, p=simres.p_ball, q=simres.q_ball)

  np.save('./data/ball_trajectory.npy', ball_traj)
  np.save('./data/parameters.npy', par)

  plt.subplot(221)
  plt.plot(simres.p_ball[:,0], simres.p_ball[:,1])
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.grid(True)
  plt.subplot(222)
  plt.plot(simres.t, simres.w_ball)
  plt.legend([R'$\omega_x$', R'$\omega_y$', R'$\omega_z$'])
  plt.grid(True)
  plt.subplot(223)
  plt.plot(simres.t, simres.energy)
  plt.ylabel('full energy')
  plt.grid(True)
  plt.subplot(224)
  plt.plot(simres.t, simres.v_ball)
  plt.legend(['$v_x$', '$v_y$', '$v_z$'])
  plt.grid(True)
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
