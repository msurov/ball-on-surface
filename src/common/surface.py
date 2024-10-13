import numpy as np
from abc import ABC, abstractmethod

class Surface(ABC):
  @abstractmethod
  def derivative(self, x : float, y : float, xder : int = 0, yder : int = 0) -> float:
    R"""
      computes z(x,y) and partial derivatives ∂z/∂x, ∂z/∂y, ∂2z/∂x∂y, etc. up to order 2
    """
    assert False

  def __call__(self, x, y) -> float:
    R"""
      computes z(x,y) and partial derivatives ∂z/∂x, ∂z/∂y, ∂2z/∂x∂y, etc. up to order 2
    """
    return self.derivative(x, y, 0, 0)

  def coords(self, x, y) -> np.ndarray:
    return np.array([x, y, self.derivative(x, y, 0, 0)])

  def normal(self, x, y) -> np.ndarray:
    hx = self.derivative(x, y, 1, 0)
    hy = self.derivative(x, y, 0, 1)
    k = np.sqrt(1 + hx**2 + hy**2)
    return np.array([-hx, -hy, 1]) / k

  def normal_jac(self, x, y) -> np.ndarray:
    hx = self.derivative(x, y, 1, 0)
    hy = self.derivative(x, y, 0, 1)
    hxx = self.derivative(x, y, 2, 0)
    hyy = self.derivative(x, y, 0, 2)
    hxy = self.derivative(x, y, 1, 1)
    k = np.sqrt(1 + hx**2 + hy**2)
    n = np.array([-hx, -hy, 1]) / k
    I = np.eye(3)
    Q = (np.outer(n, n) - I) / k
    J = Q @ np.array([
      [hxx, hxy],
      [hxy, hyy],
      [0, 0]
    ])
    return J

class ParaboloidSurface(Surface):
  def __init__(self, kx : float, ky : float) -> None:
    self.kx = kx
    self.ky = ky

  def derivative(self, x: float, y: float, xder: int = 0, yder: int = 0) -> float:
    match (xder, yder):
      case (0,0): return self.kx * x**2 + self.ky * y**2
      case (1,0): return 2 * self.kx * x
      case (0,1): return 2 * self.ky * y
      case (2,0): return 2 * self.kx
      case (0,2): return 2 * self.ky
      case (1,1): return 0
      case _: assert False, 'Not implemented'

class ConeSurface(Surface):
  def __init__(self, k, eps=1e-3):
    assert eps > 0
    self.k = k
    self.eps = eps

  @property
  def cone_side_angle(self) -> float:
    return np.arctan(self.k)

  def derivative(self, x: float, y: float, xder: int = 0, yder: int = 0) -> float:
    k = self.k
    eps = self.eps
    h = k * np.sqrt(eps + x**2 + y**2)
    match (xder, yder):
      case (0,0): return h
      case (1,0): return k**2 * x / h
      case (0,1): return k**2 * y / h
      case (2,0): return k**2 / h - k**4 * x**2 / h**3
      case (0,2): return k**2 / h - k**4 * y**2 / h**3
      case (1,1): return -k**4 * x * y / h**3
      case _: assert False, 'Not implemented'

class Plane(Surface):
  def __init__(self):
    pass

  def derivative(self, x: float, y: float, xder: int = 0, yder: int = 0) -> float:
    return 0.

class Sphere(Surface):
  def __init__(self, radius : float):
    self.radius = radius

  def derivative(self, x: float, y: float, xder: int = 0, yder: int = 0) -> float:
    r = self.radius
    q = np.sqrt(r**2 - x**2 - y**2)
    match (xder, yder):
      case (0,0): return -r + q
      case (1,0): return -x/q
      case (0,1): return -y/q
      case (2,0): return -x**2/q**3 - 1/q
      case (0,2): return -y**2/q**3 - 1/q
      case (1,1): return -x*y/q**3
      case _: assert False, 'Not implemented'

def shifted_surface(surf, dist, x, y, eps=1e-5, max_iter=100):
  u = x
  v = y

  for i in range(max_iter):
    n = surf.normal(u, v)
    x2 = u + dist * n[0]
    y2 = v + dist * n[1]
    err = max(abs(x2 - x), abs(y2 - y))
    if err <= eps:
      break
    u += x - x2
    v += y - y2

  if i == max_iter - 1:
    print(f"warn: didn't converge after {i + 1} iterations, error is {err}")
  
  z1 = surf(u, v)
  n = surf.normal(u, v)
  z2 = z1 + dist * n[2]
  return z2, u, v, err

def test_shifted():
  surf_to_test = [
    ParaboloidSurface(0.3, -0.2),
    ConeSurface(-0.7)
  ]
  for surf in surf_to_test:
    dist = 0.1
    x2 = 1.
    y2 = 2.
    z2,x1,y1,err = shifted_surface(surf, dist, x2, y2)
    z1 = surf(x1, y1)
    n = surf.normal(x1, y1)
    p1 = np.array([x1, y1, z1])
    p2 = p1 + n * dist
    assert np.allclose(p2, [x2, y2, z2])

def test_normal():
  surf_to_test = [
    ParaboloidSurface(0.3, -0.2),
    ConeSurface(-0.7)
  ]
  for surf in surf_to_test:
    x = 0.8653
    y = 2.543
    eps = 1e-5
    dx,dy = np.random.normal(size=2)
    p1 = surf.coords(x, y)
    p2 = surf.coords(x + eps * dx, y + eps * dy)
    n = surf.normal(x, y)
    assert np.allclose(np.dot(n, p2 - p1), 0)
    assert np.allclose(np.dot(n, n), 1)

def test_derivatives():
  surf_to_test = [
    ParaboloidSurface(0.3, -0.2),
    ConeSurface(-0.7)
  ]
  for surf in surf_to_test:
    x = -0.2635433
    y = 1.125
    eps = 1e-8
    hx = (surf(x + eps, y) - surf(x - eps, y)) * 0.5 / eps
    hy = (surf(x, y + eps) - surf(x, y - eps)) * 0.5 / eps
    assert np.allclose(hx, surf.derivative(x, y, 1, 0))
    assert np.allclose(hy, surf.derivative(x, y, 0, 1))

  for surf in surf_to_test:
    x = -0.2635433
    y = 1.125
    eps = 1e-8
    hxx = (surf.derivative(x + eps, y, 1, 0) - surf.derivative(x - eps, y, 1, 0)) * 0.5 / eps
    hyy = (surf.derivative(x, y + eps, 0, 1) - surf.derivative(x, y - eps, 0, 1)) * 0.5 / eps
    hxy = (surf.derivative(x + eps, y, 0, 1) - surf.derivative(x - eps, y, 0, 1)) * 0.5 / eps
    hyx = (surf.derivative(x, y + eps, 1, 0) - surf.derivative(x, y - eps, 1, 0)) * 0.5 / eps
    assert np.allclose(hxx, surf.derivative(x, y, 2, 0))
    assert np.allclose(hyy, surf.derivative(x, y, 0, 2))
    assert np.allclose(hxy, hyx)
    assert np.allclose(hxy, surf.derivative(x, y, 1, 1))

def test_normal_jac():
  surf_to_test = [
    ParaboloidSurface(0.3, 0.2),
    ConeSurface(-0.7)
  ]
  for surf in surf_to_test:
    x = 0.5213
    y = -0.9873
    eps = 1e-5
    dx = (surf.normal(x + eps, y) - surf.normal(x - eps, y)) * 0.5 / eps
    dy = (surf.normal(x, y + eps) - surf.normal(x, y - eps)) * 0.5 / eps
    J1 = np.array([dx, dy]).T
    J2 = surf.normal_jac(x, y)
    assert np.allclose(J1, J2)
    n = surf.normal(x, y)
    assert np.allclose(np.linalg.norm(n), 1)

def test_cone():
  surf = ConeSurface(-1.)
  x2 = 5
  y2 = 0
  dist = 0.3
  z2,x1,y1,err = shifted_surface(surf, dist, x2, y2)
  z1 = surf(x1, y1)
  assert np.allclose(x1, x2 - np.sqrt(2) * dist / 2, atol=1e-5)
  assert np.allclose(z2 - z1, np.sqrt(2) * dist / 2, atol=1e-5)

if __name__ == '__main__':
  test_shifted()
  test_normal_jac()
  test_cone()
  test_normal()
  test_derivatives()
