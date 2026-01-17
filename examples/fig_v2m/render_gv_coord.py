import os
import bpy
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myblender.setup import setup, set_cycles_renderer, set_output_properties
from myblender.geometry import create_ray, set_camera, build_plane
from myblender.material import set_material_i

if __name__ == '__main__':
    # ${blender} -noaudio --python examples/render_gv_coord.py
    setup()
    fbx_path = '/Users/shuaiqing/Desktop/t2m/00000000_00.fbx'

    assert os.path.exists(fbx_path), fbx_path
    
    # 记录导入前的对象和材质
    objects_before = set(bpy.data.objects.keys())
    
    # 导入FBX文件
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    # 获取导入后新添加的对象
    objects_after = set(bpy.data.objects.keys())
    imported_objects = objects_after - objects_before
    
    # 为导入的mesh对象设置材质
    vid = 0  # 颜色ID，可以根据需要修改
    for obj_name in imported_objects:
        obj = bpy.data.objects[obj_name]
        # 只为mesh对象设置材质，跳过armature等其他类型
        if obj.type == 'MESH':
            print(f'Setting material for mesh object: {obj_name}')
            set_material_i(obj, 5, use_plastic=True)

    camera_position = (2., -2., 2.)
    camera_lookat = (0, 0, 1.)
    camera_vec = np.array(camera_lookat) - np.array(camera_position)
    camera_vec = camera_vec / np.linalg.norm(camera_vec)

    arrow_cfg = {
        'cylinder_radius': 0.05,
        'cone_radius': 0.1,
        'cone_height': 0.2,
    }
    create_ray(vid=0, start=camera_position, end=camera_lookat, length_scale=0.5, **arrow_cfg)
    # camera = set_camera(location=camera_position, center=camera_lookat, focal=20)
    camera = set_camera(location=(0., -6., 3.), center=camera_lookat, focal=40)

    gravity_direction = (0, 0, 2.)

    gravity_vec = np.array(gravity_direction) - np.array(camera_lookat)
    gravity_vec = gravity_vec / np.linalg.norm(gravity_vec)

    print('gravity_vec', gravity_vec)
    create_ray(vid=1, start=camera_lookat, end=(gravity_vec*1+np.array(camera_lookat)).tolist(), **arrow_cfg)

    x_direction = np.cross(camera_vec, gravity_vec)
    x_direction = x_direction / np.linalg.norm(x_direction)
    z_direction = np.cross(x_direction, gravity_vec)
    z_direction = z_direction / np.linalg.norm(z_direction)
    print('x_direction', x_direction)
    print('z_direction', z_direction)

    create_ray(vid=2, start=camera_lookat, end=(x_direction*1+np.array(camera_lookat)).tolist(), **arrow_cfg)
    create_ray(vid=3, start=camera_lookat, end=(z_direction*1+np.array(camera_lookat)).tolist(), **arrow_cfg)

    build_plane(translation=(0, 0, 0), plane_size=5)

    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=128,
        use_transparent_bg=False,
        use_denoising=True,
    )


    set_output_properties(bpy.context.scene, 
        res_x=1024, res_y=1024, 
        tile_x=1024//1, tile_y=1024, resolution_percentage=100,
        format='JPEG')