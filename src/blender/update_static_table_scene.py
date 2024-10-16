from os.path import split, join
import bpy
import numpy as np
from common.surface import shifted_surface
from common.trajectory import RigidBodyTrajectory
from common.interp import linear_interp
from ball_on_rotary_surface.parameters import BallOnRotarySurfaceParameters


def load_sim_data():
  basepath,_ = split(bpy.data.filepath)
  datapath = join(basepath, '../data')
  par = np.load(join(datapath, 'parameters.npy'), allow_pickle=True).item()
  traj = np.load(join(datapath, 'ball_trajectory.npy'), allow_pickle=True).item()
  return {
    'trajectory': traj,
    'parameters': par,
  }

def find_object(name):
  for obj in bpy.data.objects:
    if obj.name == name:
      return obj
  return None

def get_object_bounding_box(obj):
  mesh = obj.data
  assert isinstance(mesh, bpy.types.Mesh)
  verts = mesh.vertices
  points = np.array([(elem.co.x, elem.co.y, elem.co.z) for elem in verts])
  min_xyz = np.min(points, axis=0)
  max_xyz = np.max(points, axis=0)
  return min_xyz, max_xyz

def get_object_size(obj):
  min_xyz, max_xyz = get_object_bounding_box(obj)
  dx, dy, dz = max_xyz - min_xyz
  return dx, dy, dz

def fixup_table(par : BallOnRotarySurfaceParameters):
  def f(x, y) -> float:
    z,_,_,err = shifted_surface(par.surface, -par.ball_radius, x, y)
    assert err < 1e-5
    return z

  obj = find_object('Table')
  pmin,_ = get_object_bounding_box(obj)
  table_bottom = pmin[2] + 1e-2

  mesh = obj.data
  assert isinstance(mesh, bpy.types.Mesh)
  verts = mesh.vertices

  for v in verts:
    if v.co.z > table_bottom:
      v.co.z = f(v.co.x, v.co.y)

def fixup_ball(par : BallOnRotarySurfaceParameters):
  obj = find_object('football/soccer ball')
  actual_diameter,*_ = get_object_size(obj)
  obj.scale.x = \
  obj.scale.y = \
  obj.scale.z = par.ball_radius * 2 / actual_diameter

def insert_frames(traj : RigidBodyTrajectory):
  scene = bpy.data.scenes['Scene']
  fps = scene.render.fps
  tstart = traj.t[0]
  tend = traj.t[-1]
  duration = tend - tstart
  nframes = int(duration * fps + 0.5)
  ball = find_object('football/soccer ball')
  for i in range(1, nframes + 1):
    t = (i - 1) / fps
    p = linear_interp(traj.t, traj.p, t)
    q = linear_interp(traj.t, traj.q, t)
    ball.location = p
    ball.keyframe_insert(data_path="location", frame=i)
    ball.rotation_quaternion = q
    ball.keyframe_insert(data_path="rotation_quaternion", frame=i)

def setup_anim(traj : RigidBodyTrajectory):
  scene = bpy.data.scenes['Scene']
  fps = scene.render.fps
  duration = traj.t[-1] - traj.t[0]
  scene.frame_start = 1
  scene.frame_end = int(duration * fps + 0.5)

def cleanup():
  find_object('football/soccer ball').animation_data_clear()

def main():
  data = load_sim_data()
  par = data['parameters']
  traj = data['trajectory']
  fixup_table(par)
  fixup_ball(par)
  setup_anim(traj)
  insert_frames(traj)
