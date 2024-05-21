import numpy as np
from dataclasses import dataclass

@dataclass
class RigidBodyTrajectory:
  t : np.ndarray
  p : np.ndarray
  q : np.ndarray
