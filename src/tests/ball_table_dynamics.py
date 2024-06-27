from sympy import *
from sympy.physics.mechanics import dynamicsymbols


def angleaxis(l, a):
  I = eye(3)
  l_x = wedge(l)
  R = I * cos(a) + l_x * sin(a) + l @ l.T * (1 - cos(a))
  return R

def wedge(a):
  x,y,z = a
  return Matrix([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ])

def test():
  x,y,z,wx,wy,wz = dynamicsymbols('x y z w_x w_y w_z', real=True)
  r = symbols('r', real=True, positive=True)
  g = symbols('g', real=True, positive=True)
  m = symbols('m', real=True, positive=True)
  I = eye(3)
  Jxx,Jyy,Jzz = symbols('Jxx Jyy Jzz', real=True, positive=True)
  J = Matrix([
    [Jxx, 0, 0],
    [0, Jyy, 0],
    [0, 0, Jzz],
  ])
  Fx,Fy,Fz = symbols('Fx Fy Fz', real=True)
  F = Matrix([Fx, Fy, Fz])
  ez = Matrix([0, 0, 1])

  w = Matrix([wx, wy, wz])
  p = Matrix([x, y, z])
  t, = x.args
  v = p.diff(t)
  dw = -wedge(w) @ J @ w - r * wedge(ez) @ F
  -(dw + wedge(w) @ wedge(w)) @ p - 2 * wedge(w) @ dp - g * R @ ez + F / m

if __name__ == '__main__':
  test()
