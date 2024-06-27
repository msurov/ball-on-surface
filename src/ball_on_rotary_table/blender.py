from os.path import split, join
import bpy
import numpy as np
from common.surface import shifted_surface
from common.trajectory import RigidBodyTrajectory
from common.interp import linear_interp
from .dynamics import SystemParameters
from blender.digital_display import set_display_number, animate_display
from blender.helpers import find_object, get_object_bounding_box, get_object_size


def load_sim_data():
  basepath,_ = split(bpy.data.filepath)
  datapath = join(basepath, '../data')
  par = np.load(join(datapath, 'parameters.npy'), allow_pickle=True).item()
  ball_traj = np.load(join(datapath, 'ball_trajectory.npy'), allow_pickle=True).item()
  table_traj = np.load(join(datapath, 'table_trajectory.npy'), allow_pickle=True).item()
  return {
    'ball_trajectory': ball_traj,
    'table_trajectory': table_traj,
    'parameters': par,
  }

def fixup_table(par : SystemParameters):
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

def fixup_ball(par : SystemParameters):
  obj = find_object('football/soccer ball')
  actual_diameter,*_ = get_object_size(obj)
  obj.scale.x = \
  obj.scale.y = \
  obj.scale.z = par.ball_radius * 2 / actual_diameter

def animate_ball_motion(traj : RigidBodyTrajectory):
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

def animate_time(traj : RigidBodyTrajectory):
  display = find_object('display-4')
  t = traj.t
  animate_display(display, t, t)

def animate_table_speed(traj : RigidBodyTrajectory):
  display = find_object('display-1')
  t = traj.t
  w = traj.w[:,2]
  w[::11] += 1e-2 * np.random.normal(size=w[::11].shape)
  animate_display(display, t, w)

def animate_ball_x(traj : RigidBodyTrajectory):
  display = find_object('display-2')
  t = traj.t
  x = traj.p[:,0]
  x[::17] += 1e-2 * np.random.normal(size=x[::17].shape)
  animate_display(display, t, x)

def animate_ball_y(traj : RigidBodyTrajectory):
  display = find_object('display-3')
  t = traj.t
  y = traj.p[:,1]
  y[::23] += 1e-2 * np.random.normal(size=y[::23].shape)
  animate_display(display, t, y)

def animate_table_motion(traj : RigidBodyTrajectory):
  scene = bpy.data.scenes['Scene']
  fps = scene.render.fps
  tstart = traj.t[0]
  tend = traj.t[-1]
  duration = tend - tstart
  nframes = int(duration * fps + 0.5)
  table = find_object('Table')
  for i in range(1, nframes + 1):
    t = (i - 1) / fps
    p = linear_interp(traj.t, traj.p, t)
    q = linear_interp(traj.t, traj.q, t)
    table.location = p
    table.keyframe_insert(data_path="location", frame=i)
    table.rotation_quaternion = q
    table.keyframe_insert(data_path="rotation_quaternion", frame=i)

def setup_anim(traj : RigidBodyTrajectory):
  scene = bpy.data.scenes['Scene']
  fps = scene.render.fps
  duration = traj.t[-1] - traj.t[0]
  scene.frame_start = 1
  scene.frame_end = int(duration * fps + 0.5)

def cleanup():
  find_object('football/soccer ball').animation_data_clear()
  find_object('Table').animation_data_clear()

def main():
  data = load_sim_data()
  par = data['parameters']
  ball_traj = data['ball_trajectory']
  table_traj = data['table_trajectory']
  fixup_table(par)
  fixup_ball(par)
  setup_anim(ball_traj)
  animate_ball_motion(ball_traj)
  animate_table_motion(table_traj)
  animate_time(table_traj)
  animate_table_speed(table_traj)
  animate_ball_x(ball_traj)
  animate_ball_y(ball_traj)
