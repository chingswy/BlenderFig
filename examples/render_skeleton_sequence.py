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
        'add_ground': True,
        'add_wall': False,
        'format': 'PNG',
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
        'cams': ['58860488'],
        'cams_custom': {
            'normal': [[-3.76,0.41,1.96],[np.deg2rad(73.6),np.deg2rad(-1.62),np.deg2rad(-94.2)]],
            'bird': [[-1.61,0.30,3.16],[np.deg2rad(35),np.deg2rad(-3.25),np.deg2rad(-94.6)]]
        },
        'keyframe': [40],
        'camera_root': './data/chi3d_easymocap/s03_Hug 8',
        'res': [900, 900],
        'light': {'location': [0, -1, 5], 'rotation': [0., np.pi/4, 0], 'strength': 4.0},
        'add_ground': True,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ],
        'format': 'JPEG'
    },
     'hi4d':{
        'cams': ['58860488'],
        'cams_custom': {
            'normal': [[-0.14,-3.45,2.31],[np.deg2rad(63.7),np.deg2rad(-0.05),np.deg2rad(0.313)]],
            'bird': [[-0.27,-1.80,3.43],[np.deg2rad(33.2),np.deg2rad(-0.5),np.deg2rad(-0.68)]]
        },
        'keyframe': [40],
        'camera_root': './data/chi3d_easymocap/s03_Hug 8',
        'res': [900, 900],
        'light': {'location': [0, -1, 5], 'rotation': [0., np.pi/4, 0], 'strength': 4.0},
        'add_ground': True,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ],
        'format': 'JPEG'
    }
}

CONFIG['pair12_hug12'] = CONFIG['pair10_dance10']
CONFIG['demo511'] = CONFIG['demo558']

def load_skeletons_from_dir(path, skeltype, color_table, interval=1):
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
    else:
        filenames = glob.glob(path + '*')
        filenames = sorted([os.path.basename(x) for x in filenames])
        path = os.path.dirname(path)
    filenames = filenames[::interval]
    bpy.context.scene.frame_end = len(filenames) - 1
    caches = {}
    for frame, filename in enumerate(filenames):
        print('Loading frame', frame)
        bpy.context.scene.frame_set(frame)
        record = read_skeleton(join(path, filename))
        preds = np.array(record['pred'])
        gts = np.array(record['gt'])
        preds = np.concatenate([preds[..., :3] , preds[..., -1:]], axis=-1)
        if preds.shape[0] == 0:
            distance = np.zeros((gts.shape[0], 0)) + 1000
        else:
            distance = np.linalg.norm(gts[:, None, :, :3] - preds[None, :, :, :3], axis=-1)
            distance = distance.mean(axis=-1) * 1000
        nJoints = gts.shape[-2]
        ii, jj = np.arange(distance.shape[0]), np.arange(distance.shape[1])
        ii, jj = np.meshgrid(ii, jj)
        ii_flat, jj_flat = ii.reshape(-1), jj.reshape(-1)
        distance_flat = distance[ii_flat, jj_flat]
        i_used, j_used = set(), set()
        valid_pair = []
        for idx in distance_flat.argsort():
            i_, j_ = ii_flat[idx], jj_flat[idx]
            if i_ in i_used or j_ in j_used:
                continue
            valid_pair.append((i_, j_))
            i_used.add(i_)
            j_used.add(j_)
        records_pred, records_gt = [], []
        for i, j in valid_pair:
            dist = np.linalg.norm(gts[i, ..., :3] - preds[j, ..., :3], axis=-1) * 1000
            records_pred.append({
                "mpjpe": float(distance[i, j]),
                'joints': dist,
                "score": float(preds[j, 0, 3]),
                "gt_id": int(i),
                'gt': gts[i],
                'pred': preds[j],
            })
            records_gt.append({
                "mpjpe": float(distance[i, j]),
                'joints': dist,
                "score": float(preds[j, 0, 3]),
                "gt_id": int(i),
                'gt': gts[i],
                'pred': preds[j],
            })
        for i in range(distance.shape[0]):
            if i in i_used:
                continue
            records_gt.append({
                "mpjpe": 1e8,
                'joints': np.zeros(nJoints) + 1e8,
                "score": 0.,
                "gt_id": int(i),
                'gt': gts[i]
            })
        for j in range(distance.shape[1]):
            if j in j_used:
                continue
            records_pred.append({
                "mpjpe": 1e8,
                'joints': np.zeros(nJoints) + 1e8,
                "score": float(preds[j, 0, 3]),
                "gt_id": 100000,
                'pred': preds[j]
            })
        records_pred.sort(key=lambda x:x['gt_id'])
        for i in range(len(records_pred)):
            pid = records_pred[i]['gt_id']
            if pid >= len(color_table):
                continue
            pred_ = records_pred[i]['pred']
            if pid not in caches:
                if frame > 1:continue
                points, limbs = add_skeleton(pred_, pid=list(color_table[i]), skeltype=skeltype, mode='ellipsold')
                caches[pid] = (points, limbs)
            points, limbs = caches[pid]
            update_skeleton(pred_, skeltype, points, limbs, frame)

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='panoptic15')
    parser.add_argument('--grid', type=str, default=None)
    parser.add_argument('--offset', type=float, default=[0., 0., 0.], nargs=3)
    parser.add_argument('--ground', action='store_true')
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--interval', type=int, default=1)
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))
    camera = bpy.data.objects['Camera']
    bpy.context.scene.render.engine = 'CYCLES'

    if args.mode not in CONFIG.keys():
        print('Please specify the mode: {}'.format(CONFIG.keys()))
        raise NotImplementedError
        exit()
    config = CONFIG[args.mode]
    if config['add_ground'] == True:
        size = 3
        build_plane(translation=(0, 0, 0), plane_size=size*2)
    if config.get('add_wall', False):
        for loc, rot in zip([[size, 0, size], [-size, 0, size], [0, size, size], [0, -size, size]], 
            [[0, np.pi/2, 0], [0, -np.pi/2, 0], [np.pi/2, 0, 0], [-np.pi/2, 0, 0]]):
            obj = create_plane_blender(size=size*2, location=loc, rotation=rot, shadow=False)
            setMat_plastic(obj, colorObj([1, 1, 1, 1]), metallic=0.3, specular=0.2)
    if isinstance(config['add_ground'], dict):
        bpy.ops.import_scene.fbx(filepath=config['add_ground']['filepath'])
        bpy.data.objects['Pole_basketball 15x28'].location = config['add_ground']['location']

    if 'light' in config.keys():
        if isinstance(config['light'], dict):
            add_sunlight(name='Light', **config['light'])
        elif isinstance(config['light'], list):
            for cfg in config['light']:
                add_sunlight(**cfg)

    load_skeletons_from_dir(args.path, args.skel, config.get('color_table', color_table), interval=args.interval)
    if not os.path.exists(config['camera_root']):
        config['camera_root'] = config['camera_root'].replace('/Users/shuaiqing', '')
    cameras = read_camera(join(config['camera_root'], 'intri.yml'), join(config['camera_root'], 'extri.yml'))
    format = config.get('format', 'JPEG')
    if 'cams_custom' in config.keys():
        cam = config['cams_custom']
        for k, v in cam.items():
            location = v[0]
            rotation = v[1]
            res_x, res_y = config['res']
            set_camera(location=location, rotation=rotation)
                        # setup render
            if not args.nocycle:
                set_cycles_renderer(
                    bpy.context.scene,
                    bpy.data.objects["Camera"],
                    num_samples=args.num_samples,
                    use_transparent_bg=format == 'PNG',
                    use_denoising=True,
                )
            else:
                bpy.context.scene.render.engine = 'BLENDER_EEVEE'

            n_parallel = 1
            if not args.animation:
                bpy.context.scene.frame_set(config['keyframe'][0])
            if args.out is not None and not args.debug:
                outdir = join(args.out, k + '_')
                set_output_properties(bpy.context.scene, output_file_path=outdir, 
                    res_x=res_x, res_y=res_y, 
                    tile_x=res_x//n_parallel, tile_y=res_y, resolution_percentage=100,
                    format=format)
                if args.animation:
                    bpy.ops.render.render(write_still=True, animation=True)
                else:
                    for frame in config['keyframe']:
                        bpy.context.scene.frame_set(frame)
                        # bpy.ops.render.render(write_still=True)    
    else:
        for cam in config['cams']:
            K = cameras[cam]['K']
            R = cameras[cam]['R']
            T = cameras[cam]['T']
            set_extrinsic(R, T, camera)
            res_x, res_y = config['res']
            set_intrinsic(K, camera, res_x, res_y)
            # setup render
            if not args.nocycle:
                set_cycles_renderer(
                    bpy.context.scene,
                    bpy.data.objects["Camera"],
                    num_samples=args.num_samples,
                    use_transparent_bg=format == 'PNG',
                    use_denoising=True,
                )
            else:
                bpy.context.scene.render.engine = 'BLENDER_EEVEE'

            n_parallel = 1
            if not args.animation:
                bpy.context.scene.frame_set(config['keyframe'][0])
            if args.out is not None and not args.debug:
                outdir = join(args.out, cam + '_')
                set_output_properties(bpy.context.scene, output_file_path=outdir, 
                    res_x=res_x, res_y=res_y, 
                    tile_x=res_x//n_parallel, tile_y=res_y, resolution_percentage=100,
                    format=format)
                if args.animation:
                    bpy.ops.render.render(write_still=True, animation=True)
                else:
                    for frame in config['keyframe']:
                        bpy.context.scene.frame_set(frame)
                        # bpy.ops.render.render(write_still=True)

    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)