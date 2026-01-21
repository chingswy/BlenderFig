'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''
import os
import sys
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
    load_smpl_npz,
    load_fbx,
    export_smpl_npz_to_fbx
)
from myblender.material import set_material_i, setup_mist_fog
from myblender.setup import set_cycles_renderer, add_sunlight, add_area_light, add_spot_light, build_rgb_background
from myblender.fbxtools import load_fbx_at_frame

from myblender.setup import (
    setLight_sun,
    setLight_ambient,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

from myblender.material import colorObj, setMat_plastic
from myblender.video3d import setup_video_in_3d

if __name__ == '__main__':
    # ${blender} -noaudio --python examples/fig_v2m/render_fbx_video3d.py -- /Users/shuaiqing/Desktop/t2m/00000000_00.fbx /Users/shuaiqing/Desktop/t2m/00000000_00.mp4
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--video_down', type=int, default=1)
    # parser.add_argument("--body", default=[0.05, 0.326, 1.], nargs=3, type=float)
    parser.add_argument("--body", default=[0.14, 0.211, 0.554], nargs=3, type=float)
    parser.add_argument("--ground", default=(200/255, 200/255, 200/255, 1.0), nargs=4, type=float,
    help="Ground color: (94/255, 124/255, 226/255, 1.0) 青色")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parse_args(parser)
    setup()

    configs = {
        "baichuyu": {
            "motion": "examples/fig_v2m/assets/00000005_baichuyu_pred_seed42.fbx",
            "video": "examples/fig_v2m/assets/baichuyu_30fps.mp4",
            "out": "results/baichuyu",
            "layout": "none",
            "width": 1920,
            "height": 1080,
            "camera_distance": 10,
            "camera_height": 4,
        },
        "baichuyu_wovideo": {
            "motion": "examples/fig_v2m/assets/baichuyu_ground.fbx",
            "out": "results/baichuyu_wovideo",
            "layout": "none",
            "width": 3840,
            "height": 2160,
            "camera_distance": 8,
            "camera_height": 2,
            "body": [0.14, 0.211, 0.554],
            "theme": "dark",
            "z_offset": -0.05,
        },
        "xixiyu_wovideo": {
            "motion": "examples/fig_v2m/assets/00000003_leisaixixiyucrop_pred_seed42.fbx",
            "out": "results/xixiyu_wovideo",
            "layout": "none",
            "width": 1920,
            "height": 1080,
            "camera_distance": 8,
            "camera_height": 2,
            "body": [0.14, 0.211, 0.554],
            "theme": "dark",
            "z_offset": -0.1,
        },
        "baichuyu_compare": {
            "motion": [
                {
                    "filename": "examples/fig_v2m/assets/00000005_baichuyu_pred_seed42.fbx",
                    "x_offset": -1,
                    "z_rotation": 0,
                    "z_offset": 0.1,
                },
                {
                    "filename": "examples/fig_v2m/assets/baichuyu_phmr.fbx",
                    "x_offset": 1,
                    "y_offset": 3.5,
                    "z_rotation": 180,
                }
            ],
            "video": "examples/fig_v2m/assets/baichuyu_30fps.mp4",
            "video_args": {
                "crop": [0.1, 0.1, 0.9, 0.75],
                "scale": 0.5,
            },
            "out": "results/baichuyu_compare",
            "layout": "none",
            "width": 1920,
            "height": 1080,
            "camera_distance": 8,
            "camera_height": 2,
        },
        "xixiyu": {
            "motion": "examples/fig_v2m/assets/00000003_leisaixixiyucrop_pred_seed42.fbx",
            "video": "examples/fig_v2m/assets/leisaixixiyucrop.mp4",
            "out": "results/xixiyu",
            "layout": "none",
            "width": 1024,
            "height": 1024,
            "camera_distance": 5,
            "camera_height": 2,
        },
        "xixiyu_compare": {
            "motion": [
                {
                    "filename": "examples/fig_v2m/assets/00000003_leisaixixiyucrop_pred_seed42.fbx",
                    "x_offset": -1,
                    "z_rotation": 0,
                },
                {
                    "filename": "examples/fig_v2m/assets/leisaixixiyucrop_phmr.fbx",
                    "x_offset": 1,
                    "z_rotation": 180,
                }
            ],
            "video": "examples/fig_v2m/assets/leisaixixiyucrop.mp4",
            "out": "results/xixiyu_compare",
            "layout": "none",
            "width": 1024,
            "height": 1024,
            "camera_distance": 5,
            "camera_height": 2,
        },
        "budian1": {
            "motion": "examples/fig_v2m/assets/budian1.fbx",
            "video": "examples/fig_v2m/assets/budian1_30fps.mp4",
            "out": "results/budian1",
            "layout": "none",
            "width": 1024,
            "height": 1024,
            "camera_distance": 5,
            "camera_height": 2,
        },
        "budian2": {
            "motion": "examples/fig_v2m/assets/budian2.fbx",
            "video": "examples/fig_v2m/assets/budian2_30fps.mp4",
            "out": "results/budian2",
            "layout": "none",
            "width": 1024,
            "height": 1024,
            "camera_distance": 5,
            "camera_height": 2,
        }
    }

    config = configs[args.name]

    focal = 60
    set_camera(
        location=(0, config["camera_distance"], config["camera_height"]),
        center=(0, 0, 1),
        focal=focal  # Portrait lens focal length
    )

    theme = config.get("theme", "light")
    if theme == "light":
        setup_mist_fog(
            bpy.context.scene,
            start=17,
            depth=20,
            fog_color=(1, 1, 1) # 稍微蓝一点，证明雾存在
        )
        add_sunlight(
            location=(-3, 3, 5),
            lookat=(0, 0, 1),
            strength=5,
        )

        add_area_light(
            location=(-5, 5, 2),
            lookat=(0, 0, 1),
            strength=30,
            size=10,
        )

        build_plane(translation=(0, 0, 0), plane_size=100,
                    white=(1,1,1,1), black=args.ground,
                    roughness=0.1, metallic=0.8, specular=0.8)
    elif theme == "dark":
        # 舞台聚光灯效果：纯黑背景 + 聚光灯照亮人物
        
        # 纯黑背景
        build_rgb_background(bpy.context.scene.world, rgb=(0.2, 0.2, 0.2, 1), strength=1.)
        setup_mist_fog(
            bpy.context.scene,
            start=config["camera_distance"] + 1,
            depth=config["camera_distance"] + 5,
            fog_color=(0.02, 0.02, 0.02) # 稍微蓝一点，证明雾存在
        )

        # 深色地面，略有反射以显示光圈
        build_plane(translation=(0, 0, 0), plane_size=100,
                    white=(0.02, 0.02, 0.02, 1), black=(0, 0, 0, 1),
                    roughness=0.7, metallic=0.1, specular=0.2)
        
        # 舞台光圈 - 从正上方打，在地面形成明显的圆形光圈
        add_spot_light(
            name='StageSpot',
            location=(0, 0, 8),
            lookat=(0, 0, 0),
            strength=600,
            spot_size=np.pi/8,  # 约22度，形成约2-3米直径的光圈
            spot_blend=0.5,     # 边缘柔和虚化
            shadow_soft_size=0.05,
            cast_shadow=False   # 不投射阴影，避免顶光产生难看的阴影
        )
        
        # 主聚光灯 - 从前方偏左上方打光，照亮人物
        add_spot_light(
            name='KeySpot',
            location=(-2, 5, 4),
            lookat=(0, 0, 1),
            strength=500,
            spot_size=np.pi/6,  # 30度
            spot_blend=0.3,
            shadow_soft_size=0.05
        )
        
        # 补光 - 从前方偏右侧，较弱，填充阴影
        add_spot_light(
            name='FillSpot',
            location=(2, 4, 3),
            lookat=(0, 0, 1),
            strength=200,
            spot_size=np.pi/5,
            spot_blend=0.5,
            shadow_soft_size=0.3
        )

    # Import the SMPLX animation
    smplx_name = config["motion"]

    if "video" in config:
        if config["layout"] == "leftright":
            width, height, fps = setup_video_in_3d(config["video"], down=args.video_down, position=[2.5, 0, 0], **config.get("video_args", {}))
        else:
            width, height, fps = setup_video_in_3d(config["video"], down=args.video_down, **config.get("video_args", {}))
    else:
        fps = 30

    width = config["width"]
    height = config["height"]

    if isinstance(smplx_name, str) and smplx_name.endswith('.npz'):
        smplx_obj, key, mat = load_smpl_npz(smplx_name, default_rotation=(0., 0., 0.), speedup=args.speedup)
        export_smpl_npz_to_fbx(smplx_name)
        meshColor = colorObj((153/255.,  216/255.,  201/255., 1.), 0.5, 1.0, 1.0, 0.0, 2.0)
        setMat_plastic(smplx_obj, meshColor, roughness=0.9, metallic=0.5, specular=0.5)
    elif isinstance(smplx_name, list) or smplx_name.endswith('.fbx') or smplx_name.endswith('.FBX'):
        if config["layout"] == "leftright":
            armature, mesh_object_list = load_fbx_at_frame(
                smplx_name,
                0,
                x_offset=0,
                target_frame=1
            )
            armature.location.z += 0.1

            for mesh_obj in mesh_object_list:
                set_material_i(mesh_obj, tuple(args.body), use_plastic=False)
            
            armature_side, mesh_object_list_side = load_fbx_at_frame(
                smplx_name,
                0,
                x_offset=-2.5,
                target_frame=1,
                z_rotation=90
            )
            armature_side.location.z += 0.1

            for mesh_obj in mesh_object_list_side:
                set_material_i(mesh_obj, tuple(args.body), use_plastic=False)
            
        elif config["layout"] == "none":
            if isinstance(smplx_name, list):
                for motion in smplx_name:
                    armature, mesh_object_list = load_fbx_at_frame(
                        motion["filename"],
                        0,
                        x_offset=motion["x_offset"],
                        y_offset=motion.get("y_offset", 0),
                        z_offset=motion.get("z_offset", 0),
                        target_frame=1,
                        z_rotation=motion.get("z_rotation", 0)
                    )

                    for mesh_obj in mesh_object_list:
                        set_material_i(mesh_obj, tuple(args.body), use_plastic=False)
            else:
                armature, mesh_object_list = load_fbx_at_frame(
                    smplx_name,
                    0,
                    0,
                    target_frame=1
                )
                armature.location.z += 0

                for mesh_obj in mesh_object_list:
                    set_material_i(mesh_obj, tuple(args.body), use_plastic=False)

    else:
        raise ValueError(f"Unsupported file format: {smplx_name}. Supported formats: .npz, .fbx")


    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=128,
        use_transparent_bg=False,
        use_denoising=True,
    )

    n_parallel = 1

    outdir = config["out"]
    if not outdir.endswith(os.path.sep):
        outdir = outdir + os.path.sep
   
    if args.debug:
        set_output_properties(bpy.context.scene, output_file_path=outdir,
            res_x=width//2, res_y=height//2,
            tile_x=width//2//n_parallel, tile_y=height//2, resolution_percentage=100,
            format='JPEG')
    else:
        set_output_properties(bpy.context.scene, output_file_path=outdir,
            res_x=width, res_y=height,
            tile_x=width//n_parallel, tile_y=height, resolution_percentage=100,
            format='JPEG')
    
    if not args.debug or args.render:
        bpy.ops.render.render(animation=True)
        # 使用ffmpeg合成视频
        ffmpeg_command = f"ffmpeg -r {fps} -i {outdir}/%04d.jpg -c:v libx264 -pix_fmt yuv420p {outdir}/compose.mp4"
        print(ffmpeg_command)
        os.system(ffmpeg_command)
