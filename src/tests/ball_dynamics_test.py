from ball_on_rotary_table.dynamics import Dynamics, SystemParameters
from tests.ball_on_rotary_plane_autocomp import calc_gen_acc_expression
from common.surface import Plane, ConeSurface
from common.frame_rotation import FrameAccelRot
import casadi as ca
import numpy as np
from common.quat import (
  quat_mul_vec, 
  quat_conj_mul
)

def calc_ang_acc(q, ddq):
  dw = 2 * quat_conj_mul(q, ddq)[1:4]
  return dw

def test():
  surf = Plane()
  par = SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = 0.18
  )
  table_angvel = 0.3
  table_angacc = -0.07
  tablerot = FrameAccelRot([0., 0., 1.], table_angvel, table_angacc)
  dyn = Dynamics(par, tablerot)
  
  t = 3.25
  p = np.array([1., 3.])
  w = np.array([6., -2., 9.])
  st = np.concatenate((p, w))
  dst = dyn(t, st)
  dx = dst[0]
  dy = dst[1]
  dw1 = dst[2:5]

  ddq_fun = calc_gen_acc_expression(table_angvel, table_angacc, 
              par.ball_inertia * ca.DM.eye(3), par.ball_mass, 
              par.ball_radius)
  phi = np.array([1., 0., 0., 0.])
  dphi = 0.5 * quat_mul_vec(phi, w)
  ez = np.array([0., 0., 1.])
  v = par.ball_radius * np.cross(w, ez)
  v = v[0:2]
  gencoords = np.concatenate((phi, p))
  genvels = np.concatenate((dphi, v))
  genacc = ddq_fun(t, gencoords, genvels)
  genacc = np.reshape(genacc, (6,))
  ddphi = genacc[0:4]
  dw2 = calc_ang_acc(phi, ddphi)
  # assert np.allclose(dw1, dw2)
  print(dw1)
  print(dw2)

if __name__ == '__main__':
  test()
