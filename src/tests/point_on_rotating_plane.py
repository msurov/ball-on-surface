import numpy as np
import matplotlib.pyplot as plt
from common.math import integrate_table
from scipy.integrate import solve_ivp

dot = np.dot
cross = np.cross

def rotmat(a):
  return np.array([
    [np.cos(a), -np.sin(a)],
    [np.sin(a), np.cos(a)]
  ])

def rotmat_deriv(a):
  return np.array([
    [-np.sin(a), -np.cos(a)],
    [np.cos(a), -np.sin(a)]
  ])

class Dynamics:
  def __init__(self):
    self.k = 0.1

  def get_plane_rot(self, t):
    theta0 = -1.251
    dtheta0 = 0.354
    ddtheta = 0.0752
    theta = theta0 + dtheta0 * t + ddtheta * t**2 / 2
    dtheta = dtheta0 + ddtheta * t
    return theta, dtheta, ddtheta
  
  def compute_a(self, t : float, st : np.ndarray):
    q = st[0:2]
    a = rotmat(np.pi/6) @ q
    da = rotmat(np.pi/6)
    return a, da
  
  def potential_energy(self, t : float, st : np.ndarray):
    q = st[0:2]
    theta, *_ = self.get_plane_rot(t)
    U = np.array([[0, 1]]) @ rotmat(theta) @ q
    return U

  def potential_energy_time_deriv(self, t : float, st : np.ndarray):
    q = st[0:2]
    theta,dtheta,_ = self.get_plane_rot(t)
    dU = dot(np.array([0, 1]), rotmat_deriv(theta) @ q) * dtheta
    return dU

  def potential_force(self, t : float, st : np.ndarray):
    q = st[0:2]
    theta, *_ = self.get_plane_rot(t)
    f = -rotmat(theta).T @ np.array([0, 1])
    return f

  def __call__(self, t : float, st : np.ndarray):
    _, dtheta, _ = self.get_plane_rot(t)
    q = st[0:2]
    a, da = self.compute_a(t, st)
    dq = st[2:4]
    f = self.potential_force(t, st)
    ddq = dtheta**2 * q + f + ( dot(a, - f - dtheta**2 * q) - dot(dq, da @ dq) ) / dot(a, a) * a
    return np.concatenate((dq, ddq))

  def hamiltonian(self, t, st):
    _, dtheta, _ = self.get_plane_rot(t)
    q = st[0:2]
    dq = st[2:4]
    H = (dot(dq, dq) - dot(q, q) * dtheta**2) / 2 + self.potential_energy(t, st)
    return H

  def hamiltonian_time_deriv(self, t, st):
    _, dtheta, ddtheta = self.get_plane_rot(t)
    q = st[0:2]
    dq = st[2:4]
    return -dot(q, q) * dtheta * ddtheta + self.potential_energy_time_deriv(t, st)

  def invariants(self, t, st):
    _, dtheta, _ = self.get_plane_rot(t)
    dq = st[2:4]
    a, da = self.compute_a(t, st)
    return dot(a, dq)

def test():
  d = Dynamics()
  sol = solve_ivp(d, [0, 10], [1., 2., 0, 0.], max_step=1e-2)
  q = sol.y[0:2].T
  energy = np.zeros(sol.t.shape)
  energy_change = np.zeros(sol.t.shape)
  z = np.zeros(sol.t.shape)

  for i in range(len(sol.t)):
    energy[i] = d.hamiltonian(sol.t[i], sol.y[:,i])
    energy_change[i] = d.hamiltonian_time_deriv(sol.t[i], sol.y[:,i])
    z[i] = d.invariants(sol.t[i], sol.y[:,i])

  energy_change = integrate_table(sol.t, energy_change)

  print(sol.y)

  plt.subplot(221)
  plt.plot(sol.y[0], sol.y[1])
  plt.ylabel('y')
  plt.xlabel('x')
  plt.grid(True)

  plt.subplot(222)
  plt.plot(sol.t, energy - energy_change)
  plt.axhline(energy[0], color='red', ls='--')
  plt.ylabel('energy')
  plt.grid(True)

  plt.subplot(223)
  plt.plot(sol.t, z)
  plt.ylabel('constraint')
  plt.grid(True)

  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  test()
