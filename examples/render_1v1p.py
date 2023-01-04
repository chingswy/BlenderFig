'''
  @ Date: 2022-11-15 14:37:25
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-11-15 15:16:54
  @ FilePath: /EasyMocapPublic/apps/blender/render_1v1p.py
'''
import os
import bpy
import numpy as np
from os.path import join
from myblender.geometry import (
    set_camera,
    build_plane,
    create_any_mesh,
    create_image_euler_translation
)
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--obj", type=str, help="obj names", nargs='+')
    args = parse_args(parser)
    print(args)

    setup()

    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))
    set_camera(location=(0, -3, 2.5), center=(0, 2, 1), focal=30)
    build_plane(translation=(0, 0, 0), plane_size=8)
    create_image_euler_translation(args.path, [-np.pi/2, 0, 0], [0., 1.5, 1.25],
        scale=(2.5,2.5,2.5),)
    for objname in args.obj:
        create_any_mesh(objname, vid=0, rotation=(np.pi/2, 0, 0), location=(0,0,0.8))

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

    if not args.debug:
        bpy.ops.render.render(write_still=True, animation=True)
