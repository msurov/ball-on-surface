from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from common.surface import Surface
from common.quat import quat_rot, quat_conj
from common.math import wedge
from common.frame_rotation import FrameRotation


cross = np.cross
dot = np.dot

def norm_sq(v):
  return dot(v, v)

def norm_sq_cross(a, b):
  return norm_sq(cross(a, b))

@dataclass
class SystemParameters:
  surface : Surface
  gravity_accel : float
  ball_mass : float
  ball_radius : float
  ball_inertia : float = None

  def __post_init__(self):
    if self.ball_inertia is None:
      self.ball_inertia = 2 * self.ball_mass * self.ball_radius**2 / 3

class Dynamics:
  def __init__(self, par : SystemParameters, tabrot : FrameRotation) -> None:
    self.par = par
    self.tabrot = tabrot

  def __call__(self, t, st):
    return self.dynamics(t, st)
  
  def reaction_force(self, t, st):
    g = self.par.gravity_accel
    r = self.par.ball_radius
    M = self.par.ball_inertia
    m = self.par.ball_mass
    I = np.eye(3)
    ez = I[:,2]

    x,y = st[0:2]
    normal = self.par.surface.normal(x, y)
    J = self.par.surface.normal_jac(x, y)
    p = np.array([x, y, self.par.surface(x, y)])
    w = st[2:5]
    v = r * cross(w, normal)

    q_table = self.tabrot.rot(t)
    mu = self.tabrot.angvel(t)
    mu_x = wedge(mu)
    dmu = self.tabrot.angaccel(t)
    dmu_x = wedge(dmu)
    angacc = -mu_x @ w - dmu
    acc = -2 * mu_x @ v - (mu_x @ mu_x + dmu_x) @ p

    k = m * r**2 / M
    n_x = wedge(normal)
    A = (I - k * n_x @ n_x) / m
    B = g * quat_rot(quat_conj(q_table), ez) + \
      r * cross(w, J @ v[0:2]) + \
      r * cross(angacc, normal) - acc
    lam = np.linalg.solve(A, B)
    return lam
  
  def normal_force(self, t, st):
    lam = self.reaction_force(t, st)
    x,y = st[0:2]
    normal = self.par.surface.normal(x, y)
    return lam @ normal
  
  def friction_force(self, t, st):
    lam = self.reaction_force(t, st)
    x,y = st[0:2]
    normal = self.par.surface.normal(x, y)
    return cross(lam, normal)

  def dynamics(self, t, st):
    g = self.par.gravity_accel
    r = self.par.ball_radius
    M = self.par.ball_inertia
    m = self.par.ball_mass
    I = np.eye(3)
    ez = I[:,2]

    x,y = st[0:2]
    normal = self.par.surface.normal(x, y)
    J = self.par.surface.normal_jac(x, y)
    p = np.array([x, y, self.par.surface(x, y)])
    w = st[2:5]
    v = r * cross(w, normal)

    q_table = self.tabrot.rot(t)
    mu = self.tabrot.angvel(t)
    mu_x = wedge(mu)
    dmu = self.tabrot.angaccel(t)
    dmu_x = wedge(dmu)
    angacc = -mu_x @ w - dmu
    acc = -2 * mu_x @ v - (mu_x @ mu_x + dmu_x) @ p

    k = m * r**2 / M
    n_x = wedge(normal)
    A = (I - k * n_x @ n_x) / m
    B = g * quat_rot(quat_conj(q_table), ez) + \
      r * cross(w, J @ v[0:2]) + \
      r * cross(angacc, normal) - acc
    lam = np.linalg.solve(A, B)
    dw = angacc + r / M * cross(lam, normal)
    dst = np.concatenate((v[0:2], dw))
    return dst
  
  def hamiltonian(self, t, st):
    g = self.par.gravity_accel
    r = self.par.ball_radius
    M = self.par.ball_inertia
    m = self.par.ball_mass
    I = np.eye(3)
    ez = I[:,2]

    x,y = st[0:2]
    normal = self.par.surface.normal(x, y)
    p = np.array([x, y, self.par.surface(x, y)])
    w = st[2:5]

    q_table = self.tabrot.rot(t)
    mu = self.tabrot.angvel(t)

    H = (m * r**2 * norm_sq_cross(w, normal)
         + M * norm_sq(w)
         - M * norm_sq(mu)
         - m * norm_sq_cross(mu, p)) / 2 + \
        m * g * dot(ez, quat_rot(q_table, p))
  
    return H

  def hamiltonian_time_deriv(self, t, st):
    g = self.par.gravity_accel
    M = self.par.ball_inertia
    m = self.par.ball_mass
    I = np.eye(3)
    ez = I[:,2]

    x,y = st[0:2]
    p = np.array([x, y, self.par.surface(x, y)])

    q_table = self.tabrot.rot(t)
    mu = self.tabrot.angvel(t)
    dmu = self.tabrot.angaccel(t)
    dH = - m * dot(cross(p, mu), cross(p, dmu)) \
         - M * dot(mu, dmu) \
         + m * g * dot(ez, quat_rot(q_table, cross(mu, p)))
    return dH
