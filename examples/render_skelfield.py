'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''
# TODO: This scripts show how to use blender to render a skeleton
import numpy as np
import os
from os.path import join
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
    create_image_corners
)

from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.skeleton import build_skel, read_skeleton

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='body25')
    args = parse_args(parser)

    setup()
    # set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=30)
    set_camera(location=(0, 0.5, -2), rotation=(-np.deg2rad(20), np.deg2rad(180), 0), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    datas = read_skeleton(args.path)
    for data in datas:
        if data['type'] == 'skeleton':
            build_skel(data, data['skeltype'])
        elif data['type'] == 'image':
            if False:
                x_min, y_min, z_min = data['corners'][0]
                x_max, y_max, z_max = data['corners'][1]
                corners = np.array([
                    [x_min, y_min, z_min], 
                    [x_max, y_min, z_min], 
                    [x_max, y_max, z_min], 
                    [x_min, y_max, z_min]
                ])
            else:
                corners = np.array(data['corners'])
            print(corners)
            imgname = join(os.path.dirname(args.path), os.path.basename(data['imgname']))
            # imgname = imgname.replace('heatmap_', 'heatmap_gt_')
            create_image_corners(imgname, corners)

    # build_plane(translation=(0, 0, 0))
    # build_skeleton(args.path, args.skel)

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
    # bpy.ops.render.render(write_still=True, animation=False)
    # if args.out_blend is not None:
    #     bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)