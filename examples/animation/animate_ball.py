import bpy
import random
random.seed(0)
from myblender.geometry import (
    set_camera,
    build_plane
)

from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.geometry import add_material_to_blender_primitive

def create_plane_for_animation():
    # 创建平面
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    plane = bpy.context.scene.objects['Plane']
    bpy.ops.rigidbody.object_add({'object': plane})
    plane.rigid_body.type = 'PASSIVE'
    plane.rigid_body.restitution = 0.3  # (X, Y, Z)

def create_animation_example():
    bpy.context.scene.gravity[0] = -2
    bpy.context.scene.gravity[2] = -5
    # 创建球体
    N = 3
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                color = (random.random(), random.random(), random.random())
                bpy.ops.mesh.primitive_uv_sphere_add(radius=1/N/3, location=(i/N, j/N, 5 + k/N))
                sphere = bpy.context.object
                add_material_to_blender_primitive(sphere, color)
                # 设置刚体物理
                bpy.ops.rigidbody.object_add({'object': sphere})
                sphere.rigid_body.type = 'ACTIVE'
                sphere.rigid_body.mass = 1
                sphere.rigid_body.collision_margin = 0
                # 设置球体的初始速度
                # sphere.rigid_body.linear_velocity = (5, 0, 0)  # (X, Y, Z)
                sphere.rigid_body.restitution = 0.3  # (X, Y, Z)
                sphere.rigid_body.collision_shape = 'MESH'
    # 设置动画参数
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 100

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()
    set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., 3.14/12, 0))
    create_plane_for_animation()
    create_animation_example()