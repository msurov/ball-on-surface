import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass


def curve(x, nder=0):
  kx = 0.05
  poly = np.array([kx, 0, 0])
  dpoly = np.polyder(poly, nder)
  return np.polyval(dpoly, x)

def tangent(x):
  dh = curve(x, 1)
  return np.array([1, dh]) / np.sqrt(1 + dh**2)

def tangent_deriv(x):
  dh = curve(x, 1)
  ddh = curve(x, 2)
  n = np.sqrt(1 + dh**2)**3
  return np.array([-dh * ddh, ddh]) / n


@dataclass
class Parameters:
  g : float
  m : float
  r : float
  M : float

def get_dynamics(par : Parameters):
  g = par.g
  m = par.m
  r = par.r
  M = par.M
  
  def rhs(_, st):
    x,v = st
    dh = curve(x, 1)
    ddh = curve(x, 2)
    a = 1 + dh**2
    b = dh * ddh
    c = g * dh / (1 + M / (m * r**2))
    dv = (-b * v**2 - c) / a
    return [v, dv]

  return rhs

def get_energy_fun(par : Parameters):
  g = par.g
  m = par.m
  r = par.r
  M = par.M
  
  def energy(st):
    x,vx = st
    h = curve(x)
    dh = curve(x, 1)
    vz = dh * vx
    w_sq = (1 + dh**2) * vx**2 / r**2
    K1 = m * (vx**2 + vz**2) / 2
    K2 = M * w_sq / 2
    U = m * g * h
    return K1 + K2 + U

  return energy

def get_dynamics_2(par : Parameters):
  g = par.g
  m = par.m
  r = par.r
  M = par.M
  I = np.eye(2)
  k = m * r**2 / M
  ez = np.array([0, 1])
  
  def rhs(_, st):
    x,w = st
    dh = curve(x, 1)
    ddh = curve(x, 2)
    t = tangent(x)
    dt = tangent_deriv(x)
    a = 1 / m * (I + k * np.outer(t, t))
    vx,vz = r * t * w
    b = r * dt * vx * w + ez * g
    lam = np.linalg.solve(a, b)
    print('tt', np.outer(t, t))
    print('A', a)
    print('B', b)
    print('force', lam)
    dw = -r / M * np.dot(lam, t)
    return [vx, dw]

  return rhs

def get_energy_fun_2(par : Parameters):
  g = par.g
  m = par.m
  r = par.r
  M = par.M
  
  def energy(st):
    x,w = st
    h = curve(x)
    dh = curve(x, 1)
    t = tangent(x)
    vx,vz = r * t * w
    K1 = m * (vx**2 + vz**2) / 2
    K2 = M * w**2 / 2
    U = m * g * h
    return K1 + K2 + U

  return energy

def test():
  g = 9.81
  m = 0.05
  r = 0.1
  M = 2 * m * r**2 / 5
  par = Parameters(g = g, m = m, r = r, M = M)
  rhs = get_dynamics(par)
  sol = solve_ivp(rhs, [0, 5], [1, 0], max_step=1e-2)
  energy = get_energy_fun(par)
  E = [energy(e) for e in sol.y.T]

  plt.plot(sol.t, E)
  plt.show()

def test_extended():
  g = 9.81
  m = 0.05
  r = 0.1
  M = 2 * m * r**2 / 5
  par = Parameters(g = g, m = m, r = r, M = M)
  rhs = get_dynamics_2(par)
  # sol = solve_ivp(rhs, [0, 5], [1, 0], max_step=1e-2)
  # x = sol.y[0]
  # w = sol.y[1]

  dx, dw = rhs(0, [1, 0])
  print(dx, dw)

  # plt.plot(sol.t, x)
  # plt.show()

test_extended()
