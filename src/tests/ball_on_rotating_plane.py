import numpy as np
import matplotlib.pyplot as plt
from common.math import wedge, integrate_table
from scipy.integrate import solve_ivp


cross = np.cross
dot = np.dot
norm_sq = lambda v: dot(v, v)
norm_sq_cross = lambda a,b: norm_sq(cross(a, b))

class Dynamics:
  def __init__(self) -> None:
    pass

  def get_plane_rot(self, t):
    angaccel = 0.05
    angvel0 = 1.
    theta = angvel0 * t + angaccel * t**2 / 2
    dtheta = angvel0 + angaccel * t
    ddtheta = angaccel
    return theta, dtheta, ddtheta

  def __call__(self, t, st):
    x,y = st[0:2]
    w = st[2:5]
    p = np.array([x, y, 0])
    I = np.eye(3)
    ez = I[:,2]
    ez_x = wedge(ez)
    _,dtheta,ddtheta = self.get_plane_rot(t)
    v = cross(w, ez)
    gam = ez * ddtheta - cross(w, ez) * dtheta
    a = -2*dtheta*cross(ez, v) - (dtheta * ez_x @ ez_x + ddtheta * ez_x) @ p
    lam = np.diag([0.5, 0.5, 1.]) @ (ez + cross(gam, ez) - a)
    dw = gam + cross(lam, ez)
    return np.concatenate((v[0:2], dw))
  
  def hamiltonian(self, t, st):
    x,y = st[0:2]
    w = st[2:5]
    p = np.array([x, y, 0])
    I = np.eye(3)
    ez = I[:,2]
    _,dtheta,_ = self.get_plane_rot(t)
    H = norm_sq(w) / 2 + norm_sq_cross(w, ez) / 2 \
      - dtheta**2 / 2 - norm_sq(p) * dtheta**2 / 2
    return H
  
  def hamiltonian_time_deriv(self, t, st):
    x,y = st[0:2]
    p = np.array([x, y, 0])
    _,dtheta,ddtheta = self.get_plane_rot(t)
    dH = -(1 + norm_sq(p)) * dtheta * ddtheta
    return dH

def compute_auxiliary(dynamics : Dynamics, t : float, st : np.ndarray):
  n = len(t)
  ham = np.zeros(n)
  dham = np.zeros(n)
  inv1 = np.zeros(n)
  for i in range(n):
    ham[i] = dynamics.hamiltonian(t[i], st[i,:])
    dham[i] = dynamics.hamiltonian_time_deriv(t[i], st[i,:])
  return ham, dham, inv1

def test():
  d = Dynamics()
  st0 = np.array([0.1, 0.1, 0., 0., 0.])
  sol = solve_ivp(d, [0, 5], st0, max_step=1e-3)
  h,dh,i1 = compute_auxiliary(d, sol.t, sol.y.T)
  energy_change = integrate_table(sol.t, dh)

  plt.subplot(221)
  plt.plot(sol.y[0], sol.y[1])
  plt.xlabel('x')
  plt.ylabel('y')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(sol.t, h)
  plt.plot(sol.t, energy_change)
  plt.ylabel('energy')
  plt.xlabel('t')
  plt.grid(True)

  plt.subplot(223)
  plt.plot(sol.t, i1)
  plt.ylabel('invariants')
  plt.xlabel('t')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  test()
