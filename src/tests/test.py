from quat import wedge
import numpy as np

a = np.random.normal(size=3)
I = np.eye(3)
a_x = wedge(a)
A1 = a_x @ a_x
A2 = np.outer(a, a) - I * np.dot(a, a)
print(A1 - A2)