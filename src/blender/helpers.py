import bpy
import numpy as np

def find_object(name : str):
    #   for obj in bpy.data.objects:
    #     if obj.name == name:
    #       return obj
    #   return None
    return bpy.data.objects[name]

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
