'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''
# TODO: This scripts show how to use blender to render a cube
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
    load_humanmesh
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
from myblender.geometry import create_plane, create_points
from myblender.material import colorObj

if __name__ == '__main__':

    parser = get_parser()
    args = parse_args(parser)

    setup()
    set_camera(location=(0, 2.5, 2.5), center=(0, 0, 1), focal=50)
    lightAngle = [0, 0, 0]
    strength = 2
    shadowSoftness = 0.3
    sun = setLight_sun(lightAngle, strength, shadowSoftness)
    setLight_ambient(color=(0.1,0.1,0.1,1))

    build_plane(translation=(0, 0, 0), plane_size=5)

    RGB = [	30/255, 144/255, 255/255]
    RGBA = (RGB[0], RGB[1], RGB[2], 1)
    meshColor = colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
    load_humanmesh(args.path, pid=1, meshColor=meshColor)

    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=False,
        use_denoising=args.denoising,
    )

    n_parallel = 1
    set_output_properties(bpy.context.scene, output_file_path=args.out, 
        res_x=args.res_x, res_y=args.res_y, 
        tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
        format='JPEG')
    bpy.ops.render.render(write_still=True, animation=False)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)