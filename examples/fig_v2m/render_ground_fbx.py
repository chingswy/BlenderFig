import bpy
import time
from myblender.geometry import build_plane
from myblender.material import set_material_i, setup_mist_fog
from myblender.setup import setup, set_output_properties, parse_args
from myblender.setup import set_cycles_renderer, add_sunlight, add_area_light
from myblender.geometry import set_camera
from myblender.fbxtools import load_fbx_at_frame
import os



if __name__ == "__main__":
    # ${blender} --python examples/fig_v2m/render_ground_fbx.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--frame", default=95, type=int)
    parser.add_argument("--body", default=[0.05, 0.326, 1.], nargs=3, type=float)
    parser.add_argument("--ground", default=[0.78, 0.78, 0.78, 1.], nargs=4, type=float)
    args = parse_args(parser)
    
    setup(rgb=(1,1,1,0))

    # 读入fbx
    fbxname = "assets/2026V2M/00000001_keling2_pred_seed42.fbx"

    frame_index = args.frame

    assert os.path.exists(fbxname), fbxname

    armature, mesh_object_list = load_fbx_at_frame(fbxname, frame=frame_index, x_offset=-0.5, y_offset=0)

    for mesh_obj in mesh_object_list:
        set_material_i(mesh_obj, args.body, use_plastic=True)

    armature, mesh_object_list = load_fbx_at_frame(fbxname, frame=frame_index, x_offset=0.5, y_offset=0, z_rotation=-90)

    for mesh_obj in mesh_object_list:
        set_material_i(mesh_obj, args.body, use_plastic=True)


    # 添加带有镜面反射效果的棋盘格地面
    # roughness: 0.0 = 完美镜子, 0.1 = 轻微模糊的反射, 0.5 = 默认
    # metallic: 0.0 = 非金属反射, 1.0 = 金属反射
    # specular: 反射强度
    build_plane(translation=(0, 0, 0), plane_size=100,
                white=(1,1,1,1), black=args.ground,
                roughness=0.1, metallic=0.8, specular=0.8)

    setup_mist_fog(
        bpy.context.scene,
        start=12,
        depth=20,
        fog_color=(1, 1, 1) # 稍微蓝一点，证明雾存在
    )
    add_sunlight(
        location=(0, 0, 5),
        lookat=(0, 0, 0),
        strength=10,
    )

    add_area_light(
        location=(0, 5, 2),
        lookat=(0, 0, 1),
        strength=30,
    )

    set_camera(
        location=(0, 10, 2),
        center=(0, 0, 1),
        focal=85  # Portrait lens focal length
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f'output/v2m_teaser_alt_view_{frame_index}_{timestamp}.jpg'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    set_output_properties(
        bpy.context.scene,
        output_file_path=output_path,
        format='JPEG',
        res_x=1024+512,
        res_y=2048,
        tile_x=1024+512,
        tile_y=2048,
        resolution_percentage=100,
    )

    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=512,
        use_transparent_bg=False,
        use_denoising=True,
    )

    bpy.ops.render.render(write_still=True, animation=False)