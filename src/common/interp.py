import numpy as np
from bisect import bisect

def linear_interp(xx, yy, x):
    i = bisect(xx, x)
    if i == 0:
        return yy[0]
    if i == len(xx):
        return yy[-1]

    y1 = yy[i - 1]
    y2 = yy[i]
    x1 = xx[i - 1]
    x2 = xx[i]
    y = y1 * (x2 - x) / (x2 - x1) + y2 * (x - x1) / (x2 - x1)
    return y
