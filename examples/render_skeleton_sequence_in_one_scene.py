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
import glob
from os.path import join
import numpy as np
from myblender.camera_file import read_camera
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
    create_plane_blender
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
        # 'cams': ['52', '64'],
        'cams': ['4'],
        'keyframe': [100],
        'camera_root': '/Users/shuaiqing/nas/home/shuaiqing/datasets/HI4D_easymocap/pair10_dance10',
        'res': [940, 1280],
        'light': {'location': [0, -1, 1], 'rotation': [0., np.pi/4, 0], 'strength': 4.0},
        'add_ground': True,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ]
    },
    'pair32_pose32':{
        'cams': ['40'],
        'keyframe': [40],
        'camera_root': '/Users/shuaiqing/nas/home/shuaiqing/datasets/HI4D_easymocap/pair32_pose32',
        'res': [940, 1280],
        'light': {'location': [0, -1, 1], 'rotation': [0., np.pi/8, 0], 'strength': 4.0},
        'add_ground': True,
        'add_ground': True,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ]
    },
    'demo558':{
        'cams': ['09'],
        'keyframe': [310],
        'camera_root': '/Users/shuaiqing/nas/ZJUMoCap/DeepMocap/230511/558-balance',
        'res': [1024, 1024],
        'light': {'location': [0, -1, 1], 'rotation': [0., np.pi/8, 0], 'strength': 4.0},
        'add_ground': False,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ]
    },
    'green1':{
        'cams': ['19502328'],
        'keyframe': [44],
        'camera_root': '/Users/shuaiqing/nas/home/shuaiqing/extern/green-baseketball-seq/green-baseketball+000300+000450',
        'res': [2448, 2048],
        'light': [
            {'location': (2., 5., 5.), 'rotation': (-np.pi/4, 0, 0), 'strength': 1.},
            {'location': (2., -5., 5.), 'rotation': (np.pi/4, 0, 0), 'strength': 1.}],
        'add_ground': {
            'filepath': 'assets/Basketball_15x28.fbx',
            'name': 'Pole_basketball 15x28',
            'location': (14, 0, 0)
        },
        'color_table': [color_table[i] for i in [2, 2, 0, 0, 0, 0, 3, 1, 4, 0, 0]]
    },
    'chi3d':{
        'res': [3840, 540],
        'light': {'location': [0, 0, 5], 'rotation': [0., np.pi/4, 0], 'strength': 4.0},
        'add_ground': True,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ]
    }
}

CONFIG['pair12_hug12'] = CONFIG['pair10_dance10']
CONFIG['demo511'] = CONFIG['demo558']

def load_skeletons_from_dir(path, skeltype, color_table, key='gt'):
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
    else:
        filenames = glob.glob(path + '*')
        filenames = sorted([os.path.basename(x) for x in filenames])
        path = os.path.dirname(path)
    # bpy.context.scene.frame_end = len(filenames) - 1
    # caches = {}
    interval = 10
    filenames = filenames[::interval]
    num_frames = len(filenames)
    xrange = 2 * num_frames
    x_offset = np.linspace(xrange / 2, -xrange / 2, num_frames)
    bbox_thr = 2
    for frame, filename in enumerate(filenames):
        print('Loading frame', frame)
        # bpy.context.scene.frame_set(frame)
        record = read_skeleton(join(path, filename))
        pred = np.array(record[key])
        if pred.shape[-1] == 4:
            pred[..., -1] = 1
        if 'pids' in record.keys():
            pids = record['pids']
        else:
            pids = [i for i in range(pred.shape[0])]
        for i in range(pred.shape[0]):
            pid = pids[i]
            if pid >= len(color_table):
                continue
            pred_ = pred[i].copy()
            pred_[..., :1] += x_offset[frame]
            bbox = pred_[:, :3].max(axis=0) - pred_[:, :3].min(axis=0)
            if np.prod(bbox) > bbox_thr:
                continue
            # if pid not in caches:
            #     if frame > 1:continue
            points, limbs = add_skeleton(pred_, pid=list(color_table[i]), skeltype=skeltype, mode='ellipsold')
            #     caches[pids[i]] = (points, limbs)
            # points, limbs = caches[pid]
            # update_skeleton(pred_, skeltype, points, limbs, frame)

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='panoptic15')
    parser.add_argument('--grid', type=str, default=None)
    parser.add_argument('--offset', type=float, default=[0., 0., 0.], nargs=3)
    parser.add_argument('--ground', action='store_true')
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--key', choices=['pred', 'gt'], type=str, default='gt')
    parser.add_argument('--mode', type=str, default=None)
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))
    camera = bpy.data.objects['Camera']
    bpy.context.scene.render.engine = 'CYCLES'

    if args.mode not in CONFIG.keys():
        raise NotImplementedError
        exit()
    config = CONFIG[args.mode]
    if config['add_ground'] == True:
        size = 20
        build_plane(translation=(0, 0, 0), plane_size=size*2)
        # for loc, rot in zip([[size, 0, size], [-size, 0, size], [0, size, size], [0, -size, size]], 
        #     [[0, np.pi/2, 0], [0, -np.pi/2, 0], [np.pi/2, 0, 0], [-np.pi/2, 0, 0]]):
        #     obj = create_plane_blender(size=size*2, location=loc, rotation=rot, shadow=True)
        #     setMat_plastic(obj, colorObj([1, 1, 1, 1]), metallic=0.3, specular=0.2)
    elif isinstance(config['add_ground'], dict):
        bpy.ops.import_scene.fbx(filepath=config['add_ground']['filepath'])
        bpy.data.objects['Pole_basketball 15x28'].location = config['add_ground']['location']

    if 'light' in config.keys():
        if isinstance(config['light'], dict):
            add_sunlight(name='Light', **config['light'])
        elif isinstance(config['light'], list):
            for cfg in config['light']:
                add_sunlight(**cfg)

    load_skeletons_from_dir(args.path, args.skel, config.get('color_table', color_table), args.key)
    res_x, res_y = config['res']
    K = np.array([
        [4000., 0, res_x / 2],
        [0, 4000., res_y / 2],
        [0, 0, 1]
    ]).reshape(3, 3)
    # R = np.array([
    #     1, 0, 0,
    #     0, 1, 0,
    #     0, 0, 1
    # ]).reshape(3, 3)
    # T = np.array([0, 0, 0]).reshape(3, 1)
    # set_extrinsic(R, T, camera)
    camera.location = [0, -20, 1]
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = (np.pi/2, 0, 0)
    set_intrinsic(K, camera, res_x, res_y)
    # setup render
    if not args.nocycle:
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=args.num_samples,
            use_transparent_bg=False,
            use_denoising=args.denoising,
        )
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'


    n_parallel = 1
    if args.out is not None and not args.debug:
        outdir = join(args.out, 'output')
        set_output_properties(bpy.context.scene, output_file_path=outdir, 
            res_x=res_x, res_y=res_y, 
            tile_x=res_x//n_parallel, tile_y=res_y, resolution_percentage=100,
            format=args.format)
        # if args.animation:
        #     bpy.ops.render.render(write_still=True, animation=True)
        # else:
        #     for frame in config['keyframe']:
        #         bpy.context.scene.frame_set(frame)
        bpy.ops.render.render(write_still=True)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)