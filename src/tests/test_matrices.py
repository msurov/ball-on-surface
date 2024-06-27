import sympy as sy

def wedge(a):
  x,y,z = a
  return sy.Matrix([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ])

a_ = sy.symbols('a_(x:z)', real=True)
a = sy.Matrix(a_)
b_ = sy.symbols('b_(x:z)', real=True)
b = sy.Matrix(b_)

I = sy.eye(3)

M = (wedge(a) @ wedge(b) + I)
H = wedge(b) @ M.inv()
H.simplify()
sy.pprint(H)
