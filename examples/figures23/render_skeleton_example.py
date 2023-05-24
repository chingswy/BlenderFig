'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''
# TODO: This scripts show how to use blender to render a skeleton
import os
from os.path import join
import math
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
    create_plane_blender,
    create_bbox3d
)
from myblender.camera import set_extrinsic, set_intrinsic
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.skeleton import read_skeleton, add_skeleton, update_skeleton
from myblender.material import setMat_plastic, colorObj

color_table = [
    (0/255, 0/255, 255/255), # 青色
    (255/255, 240/255, 0/255), # yellow
    (0/255.,  240/255.,  0/255.), # green
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

for i, color in enumerate(color_table):
    if isinstance(color, str):
        color_table[i] = tuple(map(lambda x: int(x, 16)/255., [color[:2], color[2:4], color[4:]]))
    color_table[i] = (*color_table[i], 1.0)

CONFIG = {
    'demo558':{
        'cams': ['07'],
        'keyframe': [304],
        'camera_root': '/Users/shuaiqing/nas/ZJUMoCap/DeepMocap/230511/558-balance',
        'res': [1024, 1024],
        'light': {'location': [0, -1, 1], 'rotation': [0., np.pi/8, 0], 'strength': 4.0},
        'add_ground': False,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ]
    }
}

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))
    camera = bpy.data.objects['Camera']
    bpy.context.scene.render.engine = 'CYCLES'

    res_x, res_y = 1024, 1024
    skeletons = np.load(args.path)
    build_plane(plane_size=4, translation=(0, 0, 0.1))
    add_sunlight(name='Light0', location=(0, 0, 5), rotation=(0, 0, 0), strength=4)
    for i, skel in enumerate(skeletons):
        add_skeleton(skel, color_table[i], skeltype='panoptic15', mode='ellips')
    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=True,
        use_denoising=True,
    )
    camera = (0, 4.5, 2)
    set_camera(location=camera, center=(0, 0, 1))

    n_parallel = 1
    if args.out is not None and not args.debug:
        outdir = args.out + '_view{}.png'.format(i)
        set_output_properties(bpy.context.scene, output_file_path=outdir, 
            res_x=res_x, res_y=res_y, 
            tile_x=res_x//n_parallel, tile_y=res_y, resolution_percentage=100,
            format='PNG')
        bpy.ops.render.render(write_still=True)
