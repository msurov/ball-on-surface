import numpy as np

def normalized(v):
  return v / np.linalg.norm(v)

def wedge(a):
  x,y,z = a
  return np.array([
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0]
  ])
