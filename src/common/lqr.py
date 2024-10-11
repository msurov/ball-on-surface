from scipy.linalg import solve_continuous_are # type: ignore
import numpy as np

def lqr_lti(A, B, Q, R):
  R'''
      :param A, B: linear system matrices of dimensions nxn, and nxm \
      :param Q, R: weighted matrices of dimensions nxn, mxm \
      :result: K,P
  '''
  P = solve_continuous_are(A, B, Q, R)
  K = -np.linalg.inv(R) @ (B.T @ P)
  return K, P
