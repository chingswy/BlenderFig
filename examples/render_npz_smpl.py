'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''
import os
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
    load_smpl_npz,
    load_fbx,
    export_smpl_npz_to_fbx
)

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

    parser = get_parser()
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--down', type=int, default=1)
    parser.add_argument('--speedup', type=float, default=1.0,
                        help='Speedup factor for animation (e.g., 2.0 means 2x faster playback by downsampling keyframes)')
    args = parse_args(parser)

    setup()
    camera = set_camera(location=(0, 2.5, 2.5), center=(0, 0, 1), focal=50)

    camera.location = [0, 5, 2.3]
    camera.rotation_euler = [75 * 3.14159 / 180, 0, -180 * 3.14159 / 180]

    lightAngle = [-45, -45, 0]
    strength = 2
    shadowSoftness = 0.3
    sun = setLight_sun(lightAngle, strength, shadowSoftness)
    setLight_ambient(color=(0.1,0.1,0.1,1))

    build_plane(translation=(0, 0, 0), plane_size=6)

    # Import the SMPLX animation
    smplx_name = args.path

    width, height, fps = setup_video_in_3d(args.video, down=args.down)
    if smplx_name.endswith('.npz'):
        smplx_obj, key, mat = load_smpl_npz(smplx_name, default_rotation=(0., 0., 0.), speedup=args.speedup)
        export_smpl_npz_to_fbx(smplx_name)
    elif smplx_name.endswith('.fbx') or smplx_name.endswith('.FBX'):
        smplx_obj, key, mat = load_fbx(smplx_name, default_rotation=(0., 0., 0.), speedup=args.speedup)
    else:
        raise ValueError(f"Unsupported file format: {smplx_name}. Supported formats: .npz, .fbx")

    meshColor = colorObj((153/255.,  216/255.,  201/255., 1.), 0.5, 1.0, 1.0, 0.0, 2.0)
    setMat_plastic(smplx_obj, meshColor, roughness=0.9, metallic=0.5, specular=0.5)

    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=64,
        use_transparent_bg=False,
        use_denoising=True,
    )

    n_parallel = 1

    outdir = args.out
    if not outdir.endswith(os.path.sep):
        outdir = outdir + os.path.sep

    set_output_properties(bpy.context.scene, output_file_path=outdir,
        res_x=width, res_y=height,
        tile_x=width//n_parallel, tile_y=height, resolution_percentage=100,
        format='JPEG')