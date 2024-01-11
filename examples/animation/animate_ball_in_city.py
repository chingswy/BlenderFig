import bpy
import random
random.seed(0)
from myblender.geometry import (
    set_camera,
    build_plane
)
from myblender.camera import get_3x4_RT_matrix_from_blender, get_calibration_matrix_K_from_blender
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.geometry import add_material_to_blender_primitive

def set_rigid_body(plane):
    bpy.ops.rigidbody.object_add({'object': plane})
    plane.rigid_body.type = 'PASSIVE'
    plane.rigid_body.restitution = 0.3  # (X, Y, Z)
    plane.rigid_body.collision_shape = 'MESH'

def create_plane_for_animation():
    # 创建平面
    bpy.ops.mesh.primitive_plane_add(size=100, location=(0, 0, 0))
    plane = bpy.context.scene.objects['Plane']
    bpy.ops.rigidbody.object_add({'object': plane})
    plane.rigid_body.type = 'PASSIVE'
    plane.rigid_body.restitution = 0.3  # (X, Y, Z)

def set_gravity(gx=0, gy=0, gz=-9.8):
    bpy.context.scene.gravity[0] = gx
    bpy.context.scene.gravity[1] = gy
    bpy.context.scene.gravity[2] = gz

def create_animation_example(N = 3, init_height=5):
    # 创建球体
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                color = (random.random(), random.random(), random.random())
                bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=8, radius=1/N/3, location=(i/N, j/N, init_height + k/N))
                sphere = bpy.context.object
                add_material_to_blender_primitive(sphere, color)
                # 设置刚体物理
                bpy.ops.rigidbody.object_add({'object': sphere})
                sphere.rigid_body.type = 'ACTIVE'
                sphere.rigid_body.mass = 1
                sphere.rigid_body.collision_margin = 0
                sphere.rigid_body.restitution = 0.3  # (X, Y, Z)
                sphere.rigid_body.collision_shape = 'MESH'
    # 设置动画参数
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 100

def deg2rad(deg):
    return deg / 180 * 3.14

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()

    camera = set_camera(location=(3, 6.2, -2.), rotation=(deg2rad(70), deg2rad(-180), deg2rad(140)), focal=30)
    add_sunlight(name='Light', location=(0., 0., -5.), rotation=(180., 3.14/12, 0))

    from myblender.geometry import myimport
    scene_obj, scene_obj_name, scene_obj_mat = myimport(args.path)
    scene_obj.scale = (0.05, 0.05, 0.05)
    scene_obj.location = (-10, -10, 0.4)
    set_rigid_body(scene_obj)
    set_gravity(0, 0, 9.8)
    create_animation_example(N=5, init_height=-2.)
    K = get_calibration_matrix_K_from_blender()
    RT = get_3x4_RT_matrix_from_blender(camera)
    print(K)
    print(RT)