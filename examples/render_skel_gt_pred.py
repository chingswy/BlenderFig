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
import glob
from os.path import join
import json
import bpy
import cv2
from myblender.camera_file import read_camera
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
from myblender.skeleton import read_skeleton, add_skeleton
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
        'cams': ['50591643', '58860488', '60457274', '65906101'],
        'keyframe': [40],
        'data_root': './data/chi3d_easymocap',
        'camera_root': './data/chi3d_easymocap/s03_Hug 8',
        'res': [900, 900],
        'light': {'location': [0, -1, 10], 'rotation': [0., np.pi/4, 0], 'strength': 3.0},
        'add_ground': True,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ],
        'bird_focal': 100,
    },
    'hi4d':{
        # 'cams': ['4', '16', '28', '40', '52', '64', '76', '88'],
        # 'cams': ['16', '40', '64', '88'],
        'cams': ['16'],
        'keyframe': [40],
        'data_root': './data/HI4D_easymocap',
        'camera_root': './data/HI4D_easymocap/pair00_fight00',
        'res': [940, 1280],
        'light': {'location': [0, -1, 10], 'rotation': [0., np.pi/4, 0], 'strength': 3.0},
        'add_ground': True,
        'color_table': [
            (8/255, 76/255, 97/255, 1.), # blue
            (219/255, 58/255, 52/255, 1.), # red
        ],
        'bird_focal': 40,
    }
}

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='panoptic15')
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--key', type=str, default='pred', choices=['gt', 'pred'])
    parser.add_argument('--crop', action='store_true')
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
        size = 2
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
    
    if not os.path.exists(config['camera_root']):
        config['camera_root'] = config['camera_root'].replace('/Users/shuaiqing', '')
    cameras = read_camera(join(config['camera_root'], 'intri.yml'), join(config['camera_root'], 'extri.yml'))

    color_table = config.get('color_table', color_table)

    record = read_skeleton(args.path) 
    if args.gt is not None:
        gt = read_skeleton(args.gt)
        gt = np.array(gt['gt'])
    else:
        gt = np.array(record['gt'])
    pred = np.array(record['pred'])
    indices0 = np.argmin(np.linalg.norm(pred[:1, 2, :3] - gt[:, 2, :3], axis=-1))
    if len(pred) > 1:
        indices = [int(indices0), int(list((set([0, 1]) - set([indices0])))[0])]
    else:
        indices = [int(indices0)]
    if args.key == 'gt':
        for i in range(gt.shape[0]):
            gt_ = gt[i]
            add_skeleton(gt_, pid=list(color_table[i]), skeltype=args.skel, mode='ellipsold')
    elif args.key == 'pred':
        for i in range(pred.shape[0]):
            pred_ = pred[i]
            add_skeleton(pred_, pid=list(color_table[indices[i]]), skeltype=args.skel, mode='ellipsold')
    
    for cam in config['cams']:
        K = cameras[cam]['K']
        R = cameras[cam]['R']
        T = cameras[cam]['T']
        dist = cameras[cam]['dist']
        set_extrinsic(R, T, camera)
        res_x, res_y = config['res']
        set_intrinsic(K, camera, res_x, res_y)
        # setup render
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=args.num_samples,
            use_transparent_bg=True,
            use_denoising=args.denoising,
        )

        n_parallel = 1
        if args.out is not None and not args.debug:
            basename = os.path.basename(args.path)
            seq, act, frame = basename.split('_')
            frame = frame.replace('.json', '')
            outname = '_'.join([cam, seq, act, frame])
            outpath = join(args.out, outname)
            set_output_properties(bpy.context.scene, 
                                  output_file_path=outpath, 
                                  res_x=res_x, 
                                  res_y=res_y, 
                                  tile_x=res_x//n_parallel, 
                                  tile_y=res_y, 
                                  resolution_percentage=100,
                                  format='PNG')
            bpy.ops.render.render(write_still=True, animation=False)

        if args.crop: 
            img = cv2.imread(outpath + '.png', cv2.IMREAD_UNCHANGED)
            raw_img_dir = '_'.join([seq, act])
            raw_img_path = os.path.join(config['data_root'], raw_img_dir, 'images', cam, frame + '.jpg')
            raw_img = cv2.imread(raw_img_path)
            h, w, _ = raw_img.shape
            kps2d = []
            for i in range(gt.shape[0]):
                kp3d = np.ascontiguousarray(gt[i][..., :3].astype(np.float64))
                kp2d, _ = cv2.projectPoints(kp3d, 
                                            R.astype(np.float64), 
                                            T.astype(np.float64), 
                                            K.astype(np.float64), 
                                            dist.astype(np.float64))
                kps2d.append(kp2d.squeeze())
            kps2d = np.concatenate(kps2d, axis=0)
            bbox_max = kps2d.max(axis=0)
            bbox_min = kps2d.min(axis=0)
            hwidth = int((bbox_max[0] - bbox_min[0]) * 0.65)
            hheight = int((bbox_max[1] - bbox_min[1]) * 0.65)
            hsize = max(hwidth, hheight)
            center = (0.5 * (bbox_max + bbox_min)).astype(int)
            bbox_max = center + hsize
            bbox_min = center - hsize
            delta_max = np.maximum(bbox_max - np.array([w, h]), 0)
            center = center - delta_max
            delta_min = np.maximum(-bbox_min, 0)
            center = center + delta_min
            bbox_max = center + hsize
            bbox_min = center - hsize
            bbox_max = bbox_max.astype(int)
            bbox_min = bbox_min.astype(int)
            raw_img = raw_img[bbox_min[1]:bbox_max[1],bbox_min[0]:bbox_max[0]]
            raw_img = cv2.resize(raw_img, (512, 512))
            img = img[bbox_min[1]:bbox_max[1],bbox_min[0]:bbox_max[0]]
            img = cv2.resize(img, (512, 512))
            cv2.imwrite(outpath + '.png', img)
            raw_out_path = os.path.join('output/raw', os.path.basename(outpath) + '.jpg')
            os.makedirs(os.path.dirname(raw_out_path), exist_ok=True)
            if not os.path.exists(raw_out_path):
                cv2.imwrite(raw_out_path, raw_img)
    
        center = np.mean(gt.reshape(-1, 4), axis=0)[:3]
        # location = [center[0], center[1], 3]
        # rotation = [0, 0, np.deg2rad(-180)]
        location = [camera.location[0], camera.location[1], camera.location[2]]
        location = np.array(location)
        vec = location - center
        vec_ = vec / np.linalg.norm(vec)
        axis = np.array([0., 0., 1.])
        w = 0.3
        axis = vec_ * w + axis * (1 - w)
        axis = axis / np.linalg.norm(axis)
        r = np.cross(vec_, axis)
        c = np.sum(axis * vec_)
        theta = np.arccos(c)
        mat = np.asarray([[0, -r[2], r[1]],
                         [r[2], 0, -r[0]],
                         [-r[1], r[0], 0]])
        mat = np.eye(3) + mat + mat @ mat / (1 + c)
        vec_ = np.linalg.norm(vec) * mat @ vec_
        location = (center + vec_).tolist()
        rotation = camera.rotation_euler
        rotation[0] = rotation[0] * w
        set_camera(focal=config['bird_focal'], location=location, rotation=rotation)

        n_parallel = 1
        if args.out is not None and not args.debug:
            basename = os.path.basename(args.path)
            seq, act, frame = basename.split('_')
            frame = frame.replace('.json', '')
            outname = '_'.join(['bird', cam, seq, act, frame])
            outpath = join(args.out, outname)
            if args.crop:
                set_output_properties(bpy.context.scene, 
                                        output_file_path=outpath, 
                                        res_x=512, 
                                        res_y=512, 
                                        tile_x=512, 
                                        tile_y=512, 
                                        resolution_percentage=100,
                                        format='PNG')
            else:
                set_output_properties(bpy.context.scene, 
                                      output_file_path=outpath, 
                                      res_x=res_x, 
                                      res_y=res_x, 
                                      tile_x=res_x, 
                                      tile_y=res_x, 
                                      resolution_percentage=100,
                                      format='PNG')
            bpy.ops.render.render(write_still=True, animation=False)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)