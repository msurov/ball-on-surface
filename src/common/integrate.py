import numpy as np
from scipy.interpolate import make_interp_spline
from common.linalg import wedge


def integrate_table_v1(x, y, I0=0):
  dx = np.diff(x)
  ndim = y.ndim
  I = np.zeros(y.shape)
  I[0,...] = I0
  I[1:,...] = np.multiply(y[1:,...] + y[:-1,...], np.reshape(0.5 * dx, (-1,) + (1,) * (ndim - 1)))
  I = np.cumsum(I, axis=0)
  return I

def integrate_table_v2(x : np.ndarray, y : np.ndarray, I0=0):
  R"""
    Integrate table-valued function
    :param x: function arguments
    :param y: function velues
    :param I0: integration constant
  """
  sp = make_interp_spline(x, y, k=3)
  Isp = sp.antiderivative()
  return Isp(x) + I0

integrate_table = integrate_table_v2

def test_integrate():
  x = np.linspace(-1, 2, 1543)
  y = np.array([
    np.sin(x),
    np.exp(x)
  ]).T
  I_expected = np.array([
    -np.cos(x),
    np.exp(x)
  ]).T
  I1 = integrate_table_v1(x, y, I_expected[0])
  I2 = integrate_table_v2(x, y, I_expected[0])
  assert np.allclose(I1, I_expected, atol=1e-5)
  assert np.allclose(I2, I_expected, atol=1e-5)

if __name__ == '__main__':
  test_integrate()
