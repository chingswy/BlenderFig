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
import numpy as np
from myblender.camera_file import read_camera
import bpy
from myblender.geometry import (
    set_camera,
    build_plane
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

for i, color in enumerate(color_table):
    if isinstance(color, str):
        color_table[i] = tuple(map(lambda x: int(x, 16)/255., [color[:2], color[2:4], color[4:]]))
    color_table[i] = (*color_table[i], 1.0)

CONFIG = {
    'pair10_dance10': {
        'cams': ['52', '64'],
        'camera_root': '/Users/shuaiqing/nas/home/shuaiqing/datasets/HI4D_easymocap/pair10_dance10',
        'res': [940, 1280],
        'light': {'location': [0, -1, 1], 'rotation': [0., np.pi/4, 0], 'strength': 4.0},
        'add_ground': True,
    }
}

def load_skeletons_from_dir(path, skeltype):
    filenames = sorted(os.listdir(path))
    bpy.context.scene.frame_end = len(filenames) - 1
    caches = {}
    for frame, filename in enumerate(filenames):
        print('Loading frame', frame)
        bpy.context.scene.frame_set(frame)
        record = read_skeleton(join(args.path, filename))
        pred = np.array(record['pred'])
        if 'pids' in record.keys():
            pids = record['pids']
        else:
            pids = [i for i in range(pred.shape[0])]
        for i in range(pred.shape[0]):
            pid = pids[i]
            if pid >= len(color_table):
                continue
            pred_ = pred[i]
            if pid not in caches:
                if frame > 1:continue
                points, limbs = add_skeleton(pred_, pid=color_table[i], skeltype=skeltype, mode='ellipsold')
                caches[pids[i]] = (points, limbs)
            points, limbs = caches[pid]
            update_skeleton(pred_, skeltype, points, limbs, frame)

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='panoptic15')
    parser.add_argument('--grid', type=str, default=None)
    parser.add_argument('--offset', type=float, default=[0., 0., 0.], nargs=3)
    parser.add_argument('--ground', action='store_true')
    parser.add_argument('--no_gt', action='store_true')
    parser.add_argument('--no_pred', action='store_true')
    parser.add_argument('--mode', type=str, default=None)
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))
    camera = bpy.data.objects['Camera']

    if args.mode not in CONFIG.keys():
        raise NotImplementedError
        exit()
    config = CONFIG[args.mode]
    if config['add_ground']:
        build_plane(translation=(0, 0, 0), plane_size=5)
    if 'light' in config.keys():
        add_sunlight(name='Light', **config['light'])
    load_skeletons_from_dir(args.path, args.skel)
    cameras = read_camera(join(config['camera_root'], 'intri.yml'), join(config['camera_root'], 'extri.yml'))
    for cam in config['cams']:
        K = cameras[cam]['K']
        R = cameras[cam]['R']
        T = cameras[cam]['T']
        set_extrinsic(R, T, camera)
        res_x, res_y = config['res']
        set_intrinsic(K, camera, res_x, res_y)
        # setup render
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=args.num_samples,
            use_transparent_bg=False,
            use_denoising=args.denoising,
        )

        n_parallel = 1
        if args.out is not None and not args.debug:
            outdir = join(args.out, cam + '_')
            set_output_properties(bpy.context.scene, output_file_path=outdir, 
                res_x=res_x, res_y=res_y, 
                tile_x=res_x//n_parallel, tile_y=res_y, resolution_percentage=100,
                format=args.format)
            bpy.ops.render.render(write_still=True, animation=True)
    # if args.out_blend is not None:
    #     bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)