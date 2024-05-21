import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.integrate import solve_ivp
from quat import quat_vec_mul, quat_mul_vec, quat_rot, quat_conj, quat_mul, wedge


def find_orientation(t: np.ndarray, q0: np.ndarray, self_angvel: callable):
  q0 = np.array(q0, float) / np.linalg.norm(q0)
  def rhs(t, q):
    dq = 0.5 * quat_mul_vec(q, self_angvel(t))
    return dq
  sol = solve_ivp(rhs, [t[0], t[-1]], q0, t_eval=t, max_step=1e-3)
  return sol.y.T

class RotatingCoordinateSystem:
  def __init__(self):
    t = np.linspace(0, 10, 6)
    w = np.random.normal(size=(len(t), 3))
    self.angvel_sp = make_interp_spline(t, w, k=3)
    self.angvel_sp.extrapolate = None
    step = 1e-2
    t = np.linspace(t[0], t[-1], 1 + int((t[-1] - t[0]) / step))
    q = find_orientation(t, np.array([0., 0., 1., 0.]), self.angvel_sp)
    self.q_sp = make_interp_spline(t, q, k=3)

  def self_angular_velocity(self, t):
    return self.angvel_sp(t)

  def self_angular_acceleration(self, t):
    return self.angvel_sp(t, 1)

  def orientation(self, t):
    return self.q_sp(t)
  
def get_state_baseframe(coordsys: RotatingCoordinateSystem, st: np.ndarray, t: np.ndarray):
  n = len(t)
  assert np.shape(st) == (n,3 + 3 +3 + 4)
  st_base = np.zeros(st.shape)

  for i in range(n):
    p = st[i,0:3]
    v = st[i,3:6]
    w = st[i,6:9]
    q = st[i,9:13]

    q_frame = coordsys.orientation(t[i])
    w_frame = coordsys.self_angular_velocity(t[i])

    p_base = quat_rot(q_frame, p)
    v_base = quat_rot(q_frame, 
                      np.cross(w_frame, p) + v
                      )
    w_base = quat_rot(quat_conj(q), w_frame + w)
    st_base[i,0:3] = p_base
    st_base[i,3:6] = v_base
    st_base[i,6:9] = w_base
    st_base[i,9:13] = quat_mul(q_frame, q)

  return st_base

def get_dynamics(coordsys: RotatingCoordinateSystem, gravity_accel: float):
  g = gravity_accel
  ez = np.array([0., 0., 1.])

  def rhs(t, st):
    p = st[0:3]
    v = st[3:6]
    w = st[6:9]
    q = st[9:13]

    mu = coordsys.self_angular_velocity(t)
    dmu = coordsys.self_angular_acceleration(t)
    q_table = coordsys.orientation(t)

    dw = np.cross(w, mu) - dmu
    dv = -g * quat_rot(quat_conj(q_table), ez) - 2 * np.cross(mu, v) - (wedge(mu) @ wedge(mu) + wedge(dmu)) @ p
    dp = v
    dq = 0.5 * quat_vec_mul(w, q)

    return np.concatenate((dp, dv, dw, dq))

  return rhs

def test():
  g = 9.81
  coordsys = RotatingCoordinateSystem()
  rhs = get_dynamics(coordsys, g)
  p0 = np.random.normal(size=3)
  v0 = np.random.normal(size=3)
  q0 = np.random.normal(size=4)
  q0 /= np.linalg.norm(q0)
  w0 = np.random.normal(size=3)
  st0 = np.concatenate((p0, v0, w0, q0))
  sol = solve_ivp(rhs, [0, 10], st0, max_step=1e-2)

  st_base = get_state_baseframe(coordsys, sol.y.T, sol.t)
  p = st_base[:,0:3]
  v = st_base[:,3:6]
  w = st_base[:,6:9]
  q = st_base[:,9:13]

  p0 = st_base[0,0:3]
  v0 = st_base[0,3:6]
  w0 = st_base[0,6:9]
  q0 = st_base[0,9:13]
  n = len(sol.t)

  for i in range(n):
    t = sol.t[i]
    z = p0[2] + v0[2] * t - g * t**2/2
    x = p0[0] + v0[0] * t
    y = p0[1] + v0[1] * t
    assert np.allclose(p[i], [x, y, z])
    assert np.allclose(v[i], [v0[0], v0[1], v0[2] - g * t])
    assert np.allclose(w[i], w0)


if __name__ == '__main__':
  test()
