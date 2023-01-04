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
    build_plane
)

from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.geometry import create_plane, create_points

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()
    set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))
    create_plane(vid=0)
    create_points(vid=1, center=(0,0,0.5), alpha=0.5)
    create_points(vid=2, center=(0,1,0.5), alpha=1)

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
    