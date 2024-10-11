import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class RigidBodyTrajectory:
  t : np.ndarray # time
  p : np.ndarray # position vector
  q : np.ndarray # quaternion
  w : Optional[np.ndarray] = None # angular velocity
  v : Optional[np.ndarray] = None # velocity
