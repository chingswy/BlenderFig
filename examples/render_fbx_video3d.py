'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_fbx_video3d.py
'''
import os
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
)
from myblender.material import set_material_i
from myblender.setup import add_sunlight, add_area_light
from myblender.fbxtools import load_fbx_at_frame

from myblender.setup import (
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

from myblender.video3d import setup_video_in_3d


if __name__ == '__main__':
    # Usage: ${blender} -noaudio --python examples/render_fbx_video3d.py -- --fbx path/to/motion.fbx --video path/to/video.mp4
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fbx', type=str, required=True, help="Path to FBX file")
    parser.add_argument('--video', type=str, required=True, help="Path to video file")
    parser.add_argument('--out', type=str, default=None, help="Output directory (default: same as fbx)")
    parser.add_argument('--width', type=int, default=1920, help="Output width")
    parser.add_argument('--height', type=int, default=1080, help="Output height")
    parser.add_argument('--num_samples', type=int, default=128, help="Render samples")
    parser.add_argument("--body", default=[0.14, 0.211, 0.554], nargs=3, type=float, help="Body color")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parse_args(parser)
    setup()

    # Camera settings
    camera_distance = 12
    camera_height = 1.5
    focal = 60
    set_camera(
        location=(0, camera_distance, camera_height),
        center=(0, 0, 1),
        focal=focal
    )

    # Lighting
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

    # Checkerboard ground
    ground_color = (200/255, 200/255, 200/255, 1.0)
    build_plane(
        translation=(0, 0, 0), 
        plane_size=100,
        white=(1, 1, 1, 1), 
        black=ground_color,
        roughness=0.1, 
        metallic=0.8, 
        specular=0.8
    )

    # Load video on the left side
    width, height, fps = setup_video_in_3d(
        args.video, 
        down=1, 
        position=[-2.5, 0, 0]  # Video on the left
    )

    # Override with user-specified dimensions if provided
    if args.width:
        width = args.width
    if args.height:
        height = args.height

    # Load FBX animation (motion on the right, at origin)
    armature, mesh_object_list = load_fbx_at_frame(
        args.fbx,
        0,
        x_offset=0,
        target_frame=1
    )
    
    # Set body material
    for mesh_obj in mesh_object_list:
        set_material_i(mesh_obj, tuple(args.body), use_plastic=False)

    # Setup renderer
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=False,
        use_denoising=True,
    )

    # Output directory (use video basename as folder name)
    if args.out is None:
        video_basename = os.path.splitext(os.path.basename(args.video))[0]
        outdir = os.path.dirname(args.fbx)
        outdir = os.path.join(outdir, video_basename)
    else:
        outdir = args.out
    
    if not outdir.endswith(os.path.sep):
        outdir = outdir + os.path.sep

    # Set output properties
    if args.debug:
        set_output_properties(
            bpy.context.scene, 
            output_file_path=outdir,
            res_x=width // 4, 
            res_y=height // 4,
            tile_x=width // 4, 
            tile_y=height // 4, 
            resolution_percentage=100,
            format='JPEG'
        )
    else:
        set_output_properties(
            bpy.context.scene, 
            output_file_path=outdir,
            res_x=width, 
            res_y=height,
            tile_x=width, 
            tile_y=height, 
            resolution_percentage=100,
            format='JPEG'
        )

    # Render
    if not args.debug or args.render:
        bpy.ops.render.render(animation=True)
        # Compose video with ffmpeg
        output_video = os.path.join(outdir, 'compose.mp4')
        ffmpeg_command = f"ffmpeg -y -r {fps} -i {outdir}/%04d.jpg -c:v libx264 -pix_fmt yuv420p {output_video}"
        print(ffmpeg_command)
        os.system(ffmpeg_command)
        print(f"Video saved to: {output_video}")

