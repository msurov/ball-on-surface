import numpy as np
from dataclasses import dataclass

@dataclass
class Trajectory:
  t : np.ndarray
  p : np.ndarray
  v : np.ndarray
  q : np.ndarray
  w : np.ndarray
