import casadi as ca

def lagrange_1st_kind(L, A, t, q, dq):
  R"""
    Find equations of motion in the normal form using lagrange equations of the 1st kind
    where
        L is casadi SX expression defining lagrangian L(q,dq,t)
        A is casadi SX expression constraint A'(q) dq = 0
    results  casadi SX expression for ddq
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
