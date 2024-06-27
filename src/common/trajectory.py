import numpy as np
from dataclasses import dataclass

@dataclass
class RigidBodyTrajectory:
  t : np.ndarray # time
  p : np.ndarray # position vector
  q : np.ndarray # quaternion
  w : np.ndarray = None # angular velocity
  v : np.ndarray = None # velocity
