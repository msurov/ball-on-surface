import numpy as np
from common.quat import quat_vec_mul
from common.linalg import wedge
from .parameters import BallOnSurfaceParameters


class Dynamics:
  def __init__(self, par : BallOnSurfaceParameters):
    self.mass = par.ball_mass
    self.radius = par.ball_radius
    self.inertia = par.ball_inertia
    self.surface = par.surface
    self.gravity_accel = par.gravity_accel

  def compute_force(self, st) -> np.ndarray:
    x,y = st[0:2]
    w = st[2:5]

    I = np.eye(3)
    ez = I[:,2]
    normal = self.surface.normal(x, y)
    v = self.radius * np.cross(w, normal)

    k = self.mass * self.radius**2 / self.inertia
    Wn = wedge(normal)
    A = (I - k * Wn @ Wn) / self.mass
    J = self.surface.normal_jac(x, y)
    B = -self.radius * np.cross(J @ v[0:2], w) + ez * self.gravity_accel
    force = np.linalg.solve(A, B)

    return force

  def __call__(self, st) -> np.ndarray:
    x,y = st[0:2]
    w = st[2:5]

    I = np.eye(3)
    ez = I[:,2]
    normal = self.surface.normal(x, y)
    v = self.radius * np.cross(w, normal)

    k = self.mass * self.radius**2 / self.inertia
    Wn = wedge(normal)
    A = (I - k * Wn @ Wn) / self.mass
    J = self.surface.normal_jac(x, y)
    B = -self.radius * np.cross(J @ v[0:2], w) + ez * self.gravity_accel
    force = np.linalg.solve(A, B)

    dx = v[0]
    dy = v[1]
    dw = self.radius/self.inertia * np.cross(force, normal)

    return np.concatenate(((dx, dy), dw))
  
  def kinetic_energy(self, st):
    x,y = st[0:2]
    w = st[2:5]
    normal = self.surface.normal(x, y)
    v = self.radius * np.cross(w, normal)
    K1 = np.dot(v, v) * self.mass / 2
    K2 = np.dot(w, w) * self.inertia / 2
    return K1 + K2

  def potential_energy(self, st):
    x,y = st[0:2]
    z = self.surface(x, y)
    return self.gravity_accel * self.mass * z
  
  def full_energy(self, st):
    return self.kinetic_energy(st) + self.potential_energy(st)


def compute_velocity(par : BallOnSurfaceParameters, st : np.ndarray):
  n,d = np.shape(st)
  assert d == 5
  v = np.zeros((n, 3))
  for i,e in enumerate(st):
    n = par.surface.normal(e[0], e[1])
    w = e[2:5]
    v[i,:] = par.ball_radius * np.cross(n, w)
  return v

def compute_ball_position(par : BallOnSurfaceParameters, st : np.ndarray):
  n,d = np.shape(st)
  assert d == 5
  p = np.zeros((n, 3))
  for i,e in enumerate(st):
    p[i,0:2] = e[0:2]
    p[i,2] = par.surface(e[0], e[1])
  return p
