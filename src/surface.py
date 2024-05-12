import numpy as np
from abc import ABC, abstractmethod


class Surface(ABC):
  @abstractmethod
  def __call__(self, x : float, y : float) -> float:
    pass

  @abstractmethod
  def normal(self, x : float, y : float) -> np.ndarray:
    pass

  @abstractmethod
  def normal_jac(self, x, y) -> np.ndarray:
    pass

class ParaboloidSurface(Surface):
  def __init__(self, kx : float, ky : float) -> None:
    self.kx = kx
    self.ky = ky

  def __call__(self, x, y) -> float:
    z = self.kx * x**2 + self.ky * y**2
    return z
  
  def normal(self, x, y) -> np.ndarray:
    px = np.array([
      1., 0., 2 * self.kx * x
    ])
    py = np.array([
      0., 1., 2 * self.ky * y
    ])
    k = np.cross(px, py)
    return k / np.linalg.norm(k)
  
  def normal_jac(self, x, y) -> np.ndarray:
    px = np.array([
      1., 0., 2 * self.kx * x
    ])
    py = np.array([
      0., 1., 2 * self.ky * y
    ])
    pxx = np.array([
      0., 0., 2 * self.kx
    ])
    pyy = np.array([
      0., 0., 2 * self.ky
    ])
    pxy = np.zeros(3)

    k = np.cross(px, py)
    norm_k = np.linalg.norm(k)
    I = np.eye(3)

    n = self.normal(x, y)
    J = np.zeros((3, 2))
    J[:,0] = (I - np.outer(n, n)) @ (np.cross(pxx, py) + np.cross(px, pxy)) / norm_k
    J[:,1] = (I - np.outer(n, n)) @ (np.cross(pxy, py) + np.cross(px, pyy)) / norm_k
    return J

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
  surf = ParaboloidSurface(0.3, 0.2)
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
  surf = ParaboloidSurface(0.3, 0.2)
  x = 0.5213
  y = -0.9873
  eps = 1e-5
  dx = (surf.normal(x + eps, y) - surf.normal(x, y)) / eps
  dy = (surf.normal(x, y + eps) - surf.normal(x, y)) / eps
  J1 = np.array([dx, dy]).T
  J2 = surf.normal_jac(x, y)
  assert np.allclose(J1, J2)

  n = surf.normal(x, y)
  assert np.allclose(np.linalg.norm(n), 1)

if __name__ == '__main__':
  test_shifted()
  test_normal()
