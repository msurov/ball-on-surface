import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from common.quat_casadi import (
  get_quat_mat,
  get_quat_right_mat,
  quat_mul_vec,
  quat_vec_part,
  quat_conj, 
  quat_conj_mul
)
from scipy.integrate import solve_ivp


def cross(a, b):
  ax,ay,az = a.elements()
  bx,by,bz = b.elements()
  return ca.vertcat(
    ay*bz - az*by,
    -ax*bz + az*bx,
    ax*by - ay*bx
  )

def compute_W(q):
  Q = 2 * get_quat_mat(quat_conj(q))
  return Q

def compute_coriolis_mat(M, q, dq):
  DM = ca.jacobian(M @ dq, q)
  C = DM - DM.T / 2
  return C @ dq

'''
def lagrange_eqs_with_constr(M, U, A, t, q, dq):
  R"""
    Lagrangian
      L(q,dq,t) = dq' M(q,t) dq - U(q,t)
    constraint
      A'(q) dq = 0
    will find expression for ddq
  """
  n,_ = q.shape
  I = ca.DM.eye(n)
  P = I - A @ ca.pinv(A.T @ A) @ A.T
  t1 = ca.jtimes(M @ dq, q, dq)
  t2 = ca.jacobian(M @ dq, t)
  t3 = ca.jacobian(dq.T @ M @ dq / 2 - U, q).T
  t4 = P @ (t1 + t2 - t3)
  t5 = ca.jtimes(A.T, q, dq) @ dq
  X = ca.vertcat(P @ M, A.T)
  Y = ca.vertcat(t4, t5)
  ddq = -ca.pinv(X) @ Y
  return ddq
'''

def lagrange_eqs_with_constr(L, A, t, q, dq):
  R"""
    Lagrangian
      L(q,dq,t) = dq' M(q,t) dq / 2 + l'(q,t) dq - U(q,t)
    constraint
      A'(q) dq = 0
    will find expression for ddq
  """
  n,_ = q.shape
  I = ca.DM.eye(n)
  P = I - A @ ca.pinv(A.T @ A) @ A.T
  Ldq = ca.jacobian(L, dq).T
  Ldqdq = ca.jacobian(Ldq, dq)
  X = ca.vertcat(P @ Ldqdq, A.T)
  Y = ca.vertcat(
    P @ (ca.jtimes(Ldq, q, dq) + ca.jacobian(Ldq, t) - ca.jacobian(L, q).T),
    ca.jtimes(A.T, q, dq) @ dq
  )
  ddq = -ca.pinv(X) @ Y
  return ddq

def dynamics_1(J):
  q = ca.SX.sym('psi', 4)
  dq = ca.SX.sym('dpsi', 4)
  W = compute_W(q)
  J_ext = ca.SX.eye(4)
  J_ext[1:4,1:4] = J
  M = W.T @ J_ext @ W
  Cdq = compute_coriolis_mat(M, q, dq)
  Minv = ca.pinv(M)
  lam = 1 / (q.T @ Minv @ q) * (q.T @ Minv @ Cdq - dq.T @ dq)
  ddq = Minv @ (-Cdq + q * lam)
  ddq_fun = ca.Function('rhs', [q, dq], [ddq])

  def rhs(_, st):
    q = st[0:4]
    dq = st[4:8]
    ddq = ddq_fun(q, dq)
    dst = ca.vertcat(dq, ddq)
    return np.reshape(dst, (8,))
  
  return rhs

def dynamics_2(J):
  q = ca.SX.sym('psi', 4)
  dq = ca.SX.sym('dpsi', 4)
  W = compute_W(q)
  J_ext = ca.SX.eye(4)
  J_ext[1:4,1:4] = J
  M = W.T @ J_ext @ W
  L = dq.T @ M @ dq / 2
  A = q
  t = ca.SX.sym('t')
  ddq = lagrange_eqs_with_constr(L, A, t, q, dq)
  st =  ca.vertcat(q, dq)
  dst = ca.vertcat(dq, ddq)
  rhs_fun = ca.Function('rhs', [st], [dst])
  rhs = lambda _, st: np.reshape(rhs_fun(st), (8,))
  return rhs

def dynamics_orig(J):
  Jinv = ca.pinv(J)
  q = ca.SX.sym('psi', 4)
  dq = ca.SX.sym('dpsi', 4)
  w = 2 * quat_conj_mul(q, dq)[1:4]
  dw = -Jinv @ (cross(w, J @ w))
  ddq = -0.25 * w.T @ w * q + 0.5 * quat_mul_vec(q, dw)
  rhs_fun = ca.Function('rhs', [q, dq], [dq, ddq])

  def rhs(_, st):
    q = st[0:4]
    dq = st[4:8]
    dq,ddq = rhs_fun(q, dq)
    dst = ca.vertcat(dq, ddq)
    return np.reshape(dst, (8,))
  
  return rhs

def test():
  np.random.seed(0)
  J = np.random.normal(size=(3,3))
  J = J @ J.T
  rhs1 = dynamics_1(J)
  q0 = ca.DM([0., 1., 0., 0.])
  w0 = ca.DM([1., 2., 5.])
  dq0 = 0.5 * quat_mul_vec(q0, w0)
  st0 = ca.vertcat(q0, dq0)
  st0 = np.reshape(st0, (8,))
  sol1 = solve_ivp(rhs1, [0., 1.], st0, max_step=1e-3)
  q1 = sol1.y[0:4].T
  dq1 = sol1.y[4:8].T

  rhs2 = dynamics_2(J)
  sol2 = solve_ivp(rhs2, [0., 1.], st0, max_step=1e-3, t_eval=sol1.t)
  q2 = sol2.y[0:4].T
  dq2 = sol2.y[4:8].T

  rhs3 = dynamics_orig(J)
  sol3 = solve_ivp(rhs2, [0., 1.], st0, max_step=1e-3, t_eval=sol1.t)
  q3 = sol2.y[0:4].T
  dq3 = sol2.y[4:8].T

  plt.gca().set_prop_cycle(None)
  plt.plot(sol1.t, q1, lw=0.5)
  plt.gca().set_prop_cycle(None)
  plt.plot(sol2.t, q2, '--', lw=1)
  plt.gca().set_prop_cycle(None)
  plt.plot(sol3.t, q3, '--', lw=2)
  plt.grid(True)
  plt.show()

if __name__ == '__main__':
  test()
