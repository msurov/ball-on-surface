from dynamics import SystemParameters, Dynamics
from surface import ParaboloidSurface
from simulate import simulate
import numpy as np
from dataclasses import asdict
import matplotlib.pyplot as plt


def main():
  surf = ParaboloidSurface(0.2, 0.13)
  par = SystemParameters(
    surface = surf,
    gravity_accel = 9.81,
    ball_mass = 0.05,
    ball_radius = 0.1
  )
  traj, energy = simulate(par, 0.7, 0.7, [0, 0, 13], 20)

  np.save('data/trajectory.npy', traj)
  np.save('data/parameters.npy', par)

  plt.subplot(221)
  plt.plot(traj.p[:,0], traj.p[:,1])
  plt.xlabel('$x$')
  plt.ylabel('$y$')
  plt.grid(True)
  plt.subplot(222)
  plt.plot(traj.t, traj.w)
  plt.legend([R'$\omega_x$', R'$\omega_y$', R'$\omega_z$'])
  plt.grid(True)
  plt.subplot(223)
  plt.plot(traj.t, energy)
  plt.ylabel('full energy')
  plt.grid(True)
  plt.subplot(224)
  plt.plot(traj.t, traj.v)
  plt.legend(['$v_x$', '$v_y$', '$v_z$'])
  plt.grid(True)
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  main()
