'''
  @ Date: 2022-04-24 15:45:49
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-04-24 16:53:33
  @ FilePath: /EasyMocapPublic/easymocap/blender/objs/generate_objs.py
'''
import open3d as o3d
import os
from os.path import join
import numpy as np

current_dir = os.path.abspath(os.path.dirname(__file__))

Vector3dVector = o3d.utility.Vector3dVector
Vector3iVector = o3d.utility.Vector3iVector
Vector2iVector = o3d.utility.Vector2iVector
TriangleMesh = o3d.geometry.TriangleMesh

def create_mesh(vertices, faces):
    mesh = TriangleMesh()
    mesh.vertices = Vector3dVector(vertices)
    mesh.triangles = Vector3iVector(faces)
    return mesh

def make_cylinder(res=100):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0, resolution=res, split=4, create_uv_map=False)
    mesh.paint_uniform_color([1, 0, 0])
    o3d.io.write_triangle_mesh(join(current_dir, 'cylinder_{}.obj'.format(res)), mesh)

def make_cube():
    mesh = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0, create_uv_map=False, map_texture_to_each_face=False)
    mesh.paint_uniform_color([1, 0, 0])
    o3d.io.write_triangle_mesh(join(current_dir, 'cube.obj'), mesh)

def make_cone(res=100, split=4):
    mesh = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=2.0, resolution=res, split=split, create_uv_map=False)
    mesh.paint_uniform_color([1, 0, 0])
    o3d.io.write_triangle_mesh(join(current_dir, f'cone_{res}.obj'), mesh)


def make_halfcylinder(res=100):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=2.0, resolution=res, split=4, create_uv_map=False)
    mesh.paint_uniform_color([1, 0, 0])
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    valid_faces = []
    for face in faces:
        # 判断是否是顶面或地面
        vert = vertices[face]
        if (vert[:, 2] > 0.999).all():
            continue
        if (vert[:, 2] < -0.999).all():
            continue
        if (vert[:, 0] <= 0.0).any():
            continue
        valid_faces.append(face)
    valid_faces = np.stack(valid_faces)
    mesh = create_mesh(vertices, valid_faces)
    o3d.io.write_triangle_mesh(join(current_dir, 'halfcylinder_{}.obj'.format(res)), mesh)
    
def make_sphere(res=8):
    mesh = o3d.geometry.TriangleMesh.create_sphere(
        radius=1.0, resolution=res)
    mesh.paint_uniform_color([1, 0, 0])
    o3d.io.write_triangle_mesh(join(current_dir, f'sphere_{res}.obj'), mesh)

if __name__ == '__main__':
    # make_cylinder()
    # make_halfcylinder()
    # make_cube()
    # make_cone()
    make_sphere(res=8)
