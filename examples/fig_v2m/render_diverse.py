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
    # ${blender} --python examples/fig_v2m/render_diverse.py -- --frame 22 --debug
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=110, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--body", default=[0.05, 0.326, 1.], nargs=3, type=float)
    parser.add_argument("--ground", default=[0.78, 0.78, 0.78, 1.], nargs=4, type=float)
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))

    # 读入fbx
    config = {
        22: {"fbxnames": [
        "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed42.fbx",
        "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed44.fbx",
        # "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed51.fbx",
        "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed55.fbx"],
        "camera_height": 10,
        "offset": 1.4,
        "use_ground": True,
        },
        62: {
            "fbxnames": [
                "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed42.fbx",
                "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed51.fbx",
                "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed44.fbx",
                "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed55.fbx"],
            "camera_height": 3,
            "offset": 0.85,
            "use_ground": True,
        }
    }[args.frame]

    frame_index = args.frame
    offset = config["offset"]

    for index,fbxname in enumerate(config["fbxnames"]):
        assert os.path.exists(fbxname), fbxname

        armature, mesh_object_list = load_fbx_at_frame(fbxname, frame=frame_index, x_offset=index*offset, y_offset=0, z_rotation=0)

        for mesh_obj in mesh_object_list:
            set_material_i(mesh_obj, tuple(args.body), use_plastic=False)

    center_x = offset * (len(config["fbxnames"]) - 1) / 2

    # 添加带有镜面反射效果的棋盘格地面
    # roughness: 0.0 = 完美镜子, 0.1 = 轻微模糊的反射, 0.5 = 默认
    # metallic: 0.0 = 非金属反射, 1.0 = 金属反射
    # specular: 反射强度
    if config["use_ground"]:
        build_plane(translation=(center_x, 0, 0), plane_size=100,
            white=(1,1,1,1), black=args.ground,
            roughness=0.1, metallic=0.8, specular=0.8)
    add_sunlight(
        location=(center_x, 0, 5),
        lookat=(center_x, 0, 0),
        strength=10,
    )

    add_area_light(
        location=(center_x, 5, 2),
        lookat=(center_x, 0, 1),
        strength=30,
    )

    set_camera(
        location=(center_x, 10, config["camera_height"]),
        center=(center_x, 0, 1),
        focal=85  # Portrait lens focal length
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f'output/v2m_teaser_diverse_{timestamp}.png'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.debug:
        res_x = 512
        res_y = 512
        num_samples = 16
    else:
        res_x = 2048
        res_y = 2048
        num_samples = 512

    set_output_properties(
        bpy.context.scene,
        output_file_path=output_path,
        format='PNG',
        res_x=res_x,
        res_y=res_y,
        tile_x=res_x,
        tile_y=res_y,
        resolution_percentage=100,
    )

    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=num_samples,
        use_transparent_bg=True,
        use_denoising=True,
    )

    if not args.debug:
        bpy.ops.render.render(write_still=True, animation=False)