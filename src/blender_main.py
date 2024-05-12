from os.path import split, join
import bpy
import numpy as np
from blender_print import print
from surface import shifted_surface
from trajectory import Trajectory
from dynamics import SystemParameters
from interp import linear_interp


def load_sim_data():
  basepath,_ = split(bpy.data.filepath)
  datapath = join(basepath, '../data')
  par = np.load(join(datapath, 'parameters.npy'), allow_pickle=True).item()
  traj = np.load(join(datapath, 'trajectory.npy'), allow_pickle=True).item()
  return {
    'trajectory': traj,
    'parameters': par
  }

def find_object(name):
  for obj in bpy.data.objects:
    if obj.name == name:
      return obj
  return None

def set_grid_heaight_map(obj, z_fun):
  mesh = obj.data
  assert isinstance(mesh, bpy.types.Mesh)
  verts = mesh.vertices
  for v in verts:
    v.co.z = z_fun(v.co.x, v.co.y)
    print(v.co.x, v.co.y, v.co.z)

def fixup_surface(par : SystemParameters):
  def f(x, y) -> float:
    z,_,_,err = shifted_surface(par.surface, -par.ball_radius, x, y)
    assert err < 1e-5
    return z
  obj = find_object('Grid')
  set_grid_heaight_map(obj, f)

def fixup_ball(par : SystemParameters):
  obj = find_object('football/soccer ball')
  obj.scale.x = \
  obj.scale.y = \
  obj.scale.z = par.ball_radius * 10

def insert_frames(traj : Trajectory):
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

def setup_anim(traj : Trajectory):
  scene = bpy.data.scenes['Scene']
  fps = scene.render.fps
  duration = traj.t[-1] - traj.t[0]
  scene.frame_start = 1
  scene.frame_end = int(duration * fps + 0.5)

def cleanup():
  find_object('football/soccer ball').animation_data_clear()
  find_object('Sphere').animation_data_clear()

def main():
  data = load_sim_data()
  fixup_surface(data['parameters'])
  fixup_ball(data['parameters'])
  traj = data['trajectory']
  setup_anim(traj)
  insert_frames(traj)

cleanup()
main()
