import numpy as np
from common.quat import quat_rot, quat_conj
from common.linalg import wedge
from common.frame_rotation import FrameRotation
from .parameters import BallOnRotarySurfaceParameters


cross = np.cross
dot = np.dot
outer = np.outer

def norm_sq(v):
  return dot(v, v)

def norm_sq_cross(a, b):
  return norm_sq(cross(a, b))

class BallOnRotarySurfaceDynamics:
  def __init__(self, par : BallOnRotarySurfaceParameters, tabrot : FrameRotation) -> None:
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
