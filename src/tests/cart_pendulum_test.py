import numpy as np
import casadi as ca
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class CartPendPar:
  mass : float
  gravity_accel : float
  rod_length : float

def cart_pend_dynamics(par : CartPendPar):
  m = par.mass
  g = par.gravity_accel
  l = par.rod_length

  u = ca.SX.sym('u')
  θ = ca.SX.sym('theta')
  dθ = ca.SX.sym('dtheta')
  rhs = ca.vertcat(
    dθ,
    (g * ca.sin(θ) - u * ca.cos(θ)) / l
  )
  state = ca.vertcat(θ, dθ)
  fun = ca.Function('rhs', [state, u], [rhs])
  return fun

def energy(par : CartPendPar):
  m = par.mass
  g = par.gravity_accel
  l = par.rod_length

  dx = ca.SX.sym('dx')
  θ = ca.SX.sym('theta')
  dθ = ca.SX.sym('dtheta')

  expr = m*(l*dθ + dx*ca.cos(θ))**2 / 2 + m*g*l*ca.cos(θ)
  fun = ca.Function('energy', [θ, dθ, dx], [expr])
  return fun

def test():
  par = CartPendPar(
    mass = 0.3, rod_length = 0.5, gravity_accel = 9.81
  )
  rhs = cart_pend_dynamics(par)

  def pivot_speed(t):
    return np.sin(t)

  def pivot_accel(t):
    return np.cos(t)

  def sys(t, st):
    u = pivot_accel(t)
    dst = rhs(st, u)
    return np.reshape(dst, (-1,))

  st0 = np.array([1e-3, 0.])
  sol = solve_ivp(sys, [0, 5], st0, max_step=1e-2)

  efun = energy(par)
  evals = [float(efun(sol.y[0,i], sol.y[1,i], pivot_speed(sol.t[i])))
            for i in range(len(sol.t))]

  plt.plot(sol.t, evals)
  plt.show()

test()
