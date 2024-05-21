from common.quat import quat_mul, quat_conj
import numpy as np

def get_quat_mat(q):
  w,x,y,z = q
  return np.array([
    [w, -x, -y, -z],
    [x, w, -z, y],
    [y, z, w, -x],
    [z, -y, x, w]
  ])

def get_quat_right_mat(q):
  w,x,y,z = q
  return np.array([
    [w, -x, -y, -z],
    [x, w, z, -y],
    [y, -z, w, x],
    [z, y, -x, w]
  ])

a = np.random.normal(size=4)
b = np.random.normal(size=4)
c1 = quat_mul(a, b)
c2 = get_quat_mat(a) @ b
c3 = get_quat_right_mat(b) @ a
assert np.allclose(get_quat_mat(a).T, get_quat_mat(quat_conj(a)))
assert np.allclose(get_quat_right_mat(a).T, get_quat_right_mat(quat_conj(a)))

assert np.allclose(get_quat_mat(a).T @ get_quat_mat(a), np.dot(a, a) * np.eye(4))
assert np.allclose(get_quat_mat(a) @ get_quat_mat(a).T, np.dot(a, a) * np.eye(4))
assert np.allclose(get_quat_right_mat(a).T @ get_quat_right_mat(a), np.dot(a, a) * np.eye(4))

# np.set_printoptions(suppress=True)
# print(Q1.T @ Q1)
# print(np.dot(a, a))
