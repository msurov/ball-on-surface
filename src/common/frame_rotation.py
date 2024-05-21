from abc import ABC, abstractmethod
import numpy as np


class FrameRotation(ABC):
  @abstractmethod
  def rot(self, t):
    R"""
      get quaternion describing frame orientation
    """
    pass

  @abstractmethod
  def angvel(self, t):
    R"""
      get self angular velocity
    """
    pass

  @abstractmethod
  def angaccel(self, t):
    R"""
      get self angular velocity time derivative
    """
    pass

class FrameAccelRot(FrameRotation):
  def __init__(self, axis, angvel, angaccel):
    self._angvel = float(angvel)
    self._angaccel = float(angaccel)
    self._axis = np.array(axis) / np.linalg.norm(axis)

  def rot(self, t):
    theta = self._angvel * t + self._angaccel * t**2 / 2
    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:4] = self._axis * np.sin(theta / 2)
    return q

  def angvel(self, t):
    return self._axis * (self._angvel + self._angaccel * t)

  def angaccel(self, t):
    return self._axis * self._angaccel

class FrameConstRot(FrameRotation):
  def __init__(self, axis, angvel):
    self._angvel = float(angvel)
    self._axis = np.array(axis) / np.linalg.norm(axis)

  def rot(self, t):
    theta = self._angvel * t
    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:4] = self._axis * np.sin(theta / 2)
    return q

  def angvel(self, t):
    return self._axis * self._angvel

  def angaccel(self, t):
    return self._axis * 0

class OscillatingFrame(FrameRotation):
  def __init__(self, axis, freq, magnitude):
    self._axis = np.array(axis) / np.linalg.norm(axis)
    self._a = magnitude
    self._w = freq

  def rot(self, t):
    theta = self._a * np.sin(self._w * t)
    q = np.zeros(4)
    q[0] = np.cos(theta/2)
    q[1:4] = self._axis * np.sin(theta/2)
    return q

  def angvel(self, t):
    dtheta = self._w * self._a * np.cos(self._w * t)
    return self._axis * dtheta

  def angaccel(self, t):
    ddtheta = -self._w * self._a**2 * np.sin(self._w * t)
    return self._axis * ddtheta
