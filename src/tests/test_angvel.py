from sympy import symbols, Matrix, pprint, latex

def wedge(a):
  x,y,z = a
  return Matrix([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ])

def vee(A):
  return Matrix([A[2,1], A[0,2], A[1,0]])

R_ = symbols('R_(1:4)(1:4)', real=True)
w = symbols('w_(1:4)', real=True)
w_x = wedge(w)
R = Matrix(3, 3, R_)
dS = (R @ w_x + w_x @ R.T) / 2
dr = vee(dS)
A = dr.jacobian(w)
print(latex(A))
