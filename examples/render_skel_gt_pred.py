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
from myblender.skeleton import read_skeleton, add_skeleton


if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='panoptic15')
    parser.add_argument('--grid', type=str, default=None)
    parser.add_argument('--offset', type=float, default=[0., 0., 0.], nargs=3)
    parser.add_argument('--ground', action='store_true')
    parser.add_argument('--no_pred', action='store_true')
    parser.add_argument('--field', type=str, default=None)
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))
    set_camera(location=(0, -2, 2.5), center=(0, 0, 1), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    if args.ground:
        build_plane(translation=(0, 0, 0), plane_size=3)
    record = read_skeleton(args.path)

    gt = np.array(record['gt'])
    pred = np.array(record['pred_by_gt'])
    offset = np.array(args.offset + [0.]).reshape(1, -1)
    for i in range(gt.shape[0]):
        gt_ = gt[i]
        add_skeleton(gt_, i, skeltype=args.skel)
    if not args.no_pred:
        for i in range(pred.shape[0]):
            pred_ = pred[i]
            pred_ = pred_ + offset
            add_skeleton(pred_, i, skeltype=args.skel)
    if args.field is not None:
        bpy.ops.object.volume_import(
            filepath=args.field, align='WORLD', 
            location=(-1, -1, 0.5), scale=(1, -1, 1), rotation=(0, 0, np.pi/2))
        bpy.context.object.scale[1] = -1

    if args.grid is not None:
        from myblender.grid import plot_grids
        grids = np.loadtxt(args.grid)
        print(grids.shape)
        grids, confs = grids[:, :3], grids[:, 3]
        print(grids[0], grids[1])
        radius = np.linalg.norm(grids[0] - grids[1]) / 2
        plot_grids(grids, confs, radius=radius, res=4, MIN_THRES=-1, gamma=0.15)
        
    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=True,
        use_denoising=args.denoising,
    )

    n_parallel = 1
    if args.out is not None:
        set_output_properties(bpy.context.scene, output_file_path=args.out, 
            res_x=args.res_x, res_y=args.res_y, 
            tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
            format=args.format)
        bpy.ops.render.render(write_still=True, animation=False)
    # if args.out_blend is not None:
    #     bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)