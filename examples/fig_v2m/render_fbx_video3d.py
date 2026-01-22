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
    build_solid_plane,
    create_volume_cube,
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


def get_pelvis_position(armature, frame):
    """Get the Pelvis bone world position at a specific frame.
    
    Args:
        armature: The Blender armature object
        frame: Frame number
    
    Returns:
        numpy array of shape (3,) containing xyz position
    """
    pelvis_names = ['Pelvis', 'pelvis', 'Hips', 'hips', 'Root', 'root', 'mixamorig:Hips']
    
    bpy.context.scene.frame_set(frame)
    
    for name in pelvis_names:
        if name in armature.pose.bones:
            bone_world_pos = armature.matrix_world @ armature.pose.bones[name].head
            return np.array([bone_world_pos.x, bone_world_pos.y, bone_world_pos.z])
    
    # Fallback: use armature location
    return np.array([armature.location.x, armature.location.y, armature.location.z])


def look_at(camera, target):
    """Make camera look at target point."""
    from mathutils import Vector
    direction = Vector(target) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def setup_following_camera(armature, camera, camera_distance, camera_height, 
                           frame_start, frame_end, keyframe_interval=30):
    """Set up a camera that follows the character's movement.
    
    Args:
        armature: The Blender armature object
        camera: The Blender camera object
        camera_distance: Distance from camera to character (Y offset)
        camera_height: Height of the camera
        frame_start: Starting frame
        frame_end: Ending frame
        keyframe_interval: Sample every N frames (default: 30)
    """
    # Generate keyframe frames (every N frames)
    keyframes = list(range(frame_start, frame_end + 1, keyframe_interval))
    # Always include the last frame
    if keyframes[-1] != frame_end:
        keyframes.append(frame_end)
    
    # Set keyframes
    for frame in keyframes:
        # 1. Get Pelvis position at this frame
        pelvis_pos = get_pelvis_position(armature, frame)
        
        # 2. Calculate camera position: same X as pelvis, Y + distance, fixed height
        camera.location.x = pelvis_pos[0]
        camera.location.y = pelvis_pos[1] + camera_distance
        camera.location.z = camera_height
        
        # 3. Look at the pelvis (or slightly above for chest)
        target = pelvis_pos.copy()
        target[2] = 1.0  # Look at chest height
        look_at(camera, target)
        
        # 4. Insert keyframes
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    
    # 5. Set Bezier interpolation for smooth camera movement
    if camera.animation_data and camera.animation_data.action:
        for fcurve in camera.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'BEZIER'
                keyframe.handle_left_type = 'AUTO'
                keyframe.handle_right_type = 'AUTO'
    
    print(f"Following camera: {len(keyframes)} keyframes from frame {frame_start} to {frame_end}")


def load_config_from_json(config_path, task_name):
    """Load task configuration from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        task_name: Name of the task to load
    
    Returns:
        dict: Merged configuration (defaults + task-specific)
    """
    import json
    with open(config_path, 'r') as f:
        full_config = json.load(f)
    
    defaults = full_config.get('defaults', {})
    tasks = full_config.get('tasks', {})
    
    if task_name not in tasks:
        raise ValueError(f"Task '{task_name}' not found in config. Available: {list(tasks.keys())}")
    
    # Merge defaults with task-specific config
    config = defaults.copy()
    config.update(tasks[task_name])
    return config


if __name__ == '__main__':
    # ${blender} -noaudio --python examples/fig_v2m/render_fbx_video3d.py -- /Users/shuaiqing/Desktop/t2m/00000000_00.fbx /Users/shuaiqing/Desktop/t2m/00000000_00.mp4
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--config', type=str, default='examples/fig_v2m/render_config.json', help="Path to JSON config file")
    parser.add_argument('--video_down', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=256)
    # parser.add_argument("--body", default=[0.05, 0.326, 1.], nargs=3, type=float)
    parser.add_argument("--body", default=[0.14, 0.211, 0.554], nargs=3, type=float)
    parser.add_argument("--ground", default=(200/255, 200/255, 200/255, 1.0), nargs=4, type=float,
    help="Ground color: (94/255, 124/255, 226/255, 1.0) 青色")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--denoising", action="store_true")
    args = parse_args(parser)
    setup()

    # Load config from JSON
    config = load_config_from_json(args.config, args.name)
    # Override args with config values
    if 'body' in config:
        args.body = config['body']
    if 'ground' in config and isinstance(config['ground'], list):
        args.ground = tuple(config['ground'])
    if 'video_down' in config:
        args.video_down = config['video_down']
    if 'num_samples' in config:
        args.num_samples = config['num_samples']

    focal = 60
    set_camera(
        location=(0, config["camera_distance"], config["camera_height"]),
        center=(0, 0, 1),
        focal=focal  # Portrait lens focal length
    )

    theme = config.get("theme", "light")
    if theme == "light":
        if '_over' in args.name:
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
        ground_mode = config.get("ground", "checkerboard")
        if ground_mode == "checkerboard":
            build_plane(translation=(0, 0, 0), plane_size=100,
                        white=(0.02, 0.02, 0.02, 1), black=(0, 0, 0, 1),
                        roughness=0.7, metallic=0.1, specular=0.2)
        elif ground_mode == "plane":
            build_solid_plane(translation=(0, 0, 0), plane_size=100,
                              color=(0.02, 0.02, 0.02, 1.0),  # Dark gray/black
                              roughness=0.1, metallic=0.5, specular=0.5)
        
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
        
        # # 体积光 (Volumetric Lighting / 丁达尔效应)
        # # 创建一个大的体积立方体，包裹整个场景
        # create_volume_cube(
        #     location=(0, 0, 5),      # 中心位置稍高，覆盖人物上方空间
        #     size=20,                  # 足够大以覆盖整个场景
        #     density=0.05,             # 低密度，产生若隐若现的光束效果
        #     anisotropy=0.5,           # 正向散射，让光束更明显
        #     name="VolumetricFog"
        # )
        
        # # 体积光专用聚光灯 - 从头顶打下来，产生明显的光柱效果
        # add_spot_light(
        #     name='VolumetricSpot',
        #     location=(0, 2, 10),       # 从正上方稍偏后打下来
        #     lookat=(0, 0, 0),
        #     strength=800,              # 较强的光，穿透体积产生光柱
        #     spot_size=np.pi/10,        # 约18度，形成集中的光柱
        #     spot_blend=0.2,            # 边缘较锐利
        #     shadow_soft_size=0.02,
        #     cast_shadow=True           # 投射阴影，增强光柱效果
        # )

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
                z_offset = config.get("z_offset", 0)
                armature, mesh_object_list = load_fbx_at_frame(
                    smplx_name,
                    0,
                    0,
                    target_frame=1,
                    z_offset=z_offset
                )
                armature.location.z += 0

                for mesh_obj in mesh_object_list:
                    set_material_i(mesh_obj, tuple(args.body), use_plastic=False)

    else:
        raise ValueError(f"Unsupported file format: {smplx_name}. Supported formats: .npz, .fbx")

    # Setup following camera if specified
    camera_mode = config.get("camera", "static")
    if camera_mode == "following":
        camera = bpy.data.objects["Camera"]
        frame_start = bpy.context.scene.frame_start
        frame_end = bpy.context.scene.frame_end
        keyframe_interval = config.get("camera_keyframe_interval", 30)
        
        setup_following_camera(
            armature=armature,
            camera=camera,
            camera_distance=config["camera_distance"],
            camera_height=config["camera_height"],
            frame_start=frame_start,
            frame_end=frame_end,
            keyframe_interval=keyframe_interval
        )

    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=False,
        use_denoising=args.denoising,
    )

    n_parallel = 1

    outdir = config["out"]
    if not outdir.endswith(os.path.sep):
        outdir = outdir + os.path.sep
   
    if args.debug:
        set_output_properties(bpy.context.scene, output_file_path=outdir,
            res_x=width//4, res_y=height//4,
            tile_x=width//4//n_parallel, tile_y=height//4, resolution_percentage=100,
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
