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
from myblender.geometry import create_plane, create_points, create_plane_blender
from myblender.skeleton import read_skeleton, add_skeleton, update_skeleton

color_table = [
    (94/255, 124/255, 226/255), # 青色
    (255/255, 200/255, 87/255), # yellow
    (74/255.,  189/255.,  172/255.), # green
    (8/255, 76/255, 97/255), # blue
    (219/255, 58/255, 52/255), # red
    (77/255, 40/255, 49/255), # brown
    '114B5F',
    'D89D6A',
    'A0A4B8',
    '2EC0F9',
    '30332E', # 淡黑色
    'F2D1C9', # 淡粉色
]

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()

    data = np.load(args.path)
    data = data - data[:, [8]]
    # (nFrames, 15, 3)
    FRAME_STEP = 6
    FRAME_DIST = 1.4
    SEQ_DIST = 1.4
    indices = np.random.randint(0, data.shape[0], FRAME_STEP**2)
    center = (FRAME_STEP*FRAME_DIST/2, FRAME_STEP*FRAME_DIST/2, 0)
    set_camera(location=(35, ((FRAME_STEP-1)/2 * FRAME_DIST), 25), rotation=(50/180*np.pi, 0., np.pi/2), focal=150)
    add_sunlight(name='Light', location=(center[0], center[1], 2), rotation=(0., np.pi/4, 0))
    plane = create_plane_blender([center[0], center[1], data[indices, ..., 2].min()], size=FRAME_STEP*FRAME_DIST*1.2)
    plane.hide_viewport = True
    plane.is_shadow_catcher = True

    for order, i in enumerate(indices):
        kpts = data[i, :15]
        kpts[:, 0] += FRAME_DIST * (order // FRAME_STEP)
        kpts[:, 1] += SEQ_DIST * (order % FRAME_STEP)
        add_skeleton(kpts, pid=order//FRAME_STEP, skeltype='body15', mode='ellips')
    
    args.res_x = 1024
    args.res_y = 1024

    if False:
        starts = [0, 100000, 200000, 300000, 400000, 500000]
        for seq, start in enumerate(starts):
            for i in range(0, 3):
                kpts = data[start+i*FRAME_STEP * 10, :15]
                kpts[:, 0] += FRAME_DIST * i
                kpts[:, 1] += SEQ_DIST * seq
                add_skeleton(kpts, pid='000000', skeltype='body15', mode='ellips')

    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=True,
        use_denoising=True,
    )

    n_parallel = 1
    set_output_properties(bpy.context.scene, output_file_path=args.out, 
        res_x=args.res_x, res_y=args.res_y, 
        tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
        format='PNG')
    bpy.ops.render.render(write_still=True, animation=False)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)