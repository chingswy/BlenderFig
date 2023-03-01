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
from myblender.skeleton import read_skeleton, build_skel

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='body25')
    parser.add_argument('--phi', type=float, default=30,
                        help='the angle of camera in the vertical direction')
    parser.add_argument('--theta', type=float, default=0,
                        help='the angle of camera')
    
    args = parse_args(parser)

    setup()
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    skels = read_skeleton(args.path)
    gt = np.array(skels['gt'])
    # 中心设置在人的中心
    center = gt[..., :3].mean(axis=1).mean(axis=0)
    build_plane(translation=(center[0], center[1], 0), plane_size=16)
    # 计算相机角度
    radius = 2
    phi = args.phi
    theta = args.theta
    z = center[2] + radius * np.sin(np.deg2rad(phi))
    x = center[0] + radius * np.cos(np.deg2rad(theta))
    y = center[1] + radius * np.sin(np.deg2rad(theta))

    set_camera(location=(x, y, z), center=center, focal=30)
    preds = np.array(skels['pred'])
    for pid, pred in enumerate(preds[:2]):
        print('[Vis] add skeleton {}'.format(pid))
        build_skel({'id': pid, 'keypoints3d': pred}, skeltype=args.skel)


    n_parallel = 1
    if args.out is not None:
        if args.out.endswith('.png'):
            format = 'PNG'
            use_transparent = True
        else:
            format = 'JPEG'
            use_transparent = False
        # setup render
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=args.num_samples,
            use_transparent_bg=use_transparent,
            use_denoising=args.denoising,
        )
        set_output_properties(bpy.context.scene, output_file_path=args.out, 
            res_x=args.res_x, res_y=args.res_y, 
            tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
            format=format)
        bpy.ops.render.render(write_still=True, animation=False)
        # if args.out_blend is not None:
        #     bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)