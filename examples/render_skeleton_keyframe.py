import os
import bpy
import numpy as np
import cv2

from glob import glob
import yaml
from os.path import join
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.camera_file import read_camera
from myblender.camera import set_extrinsic, set_intrinsic
from myblender.skeleton import read_skeleton, add_skeleton, update_skeleton
from myblender.geometry import look_at, build_plane
from math import radians

def load_yaml_cfg(path):
    cfg = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)
    return cfg

def load_skeletons_from_dir(path, interval=1):
    if os.path.isdir(path):
        filenames = sorted(os.listdir(path))
    else:
        filenames = glob(path + '*.json')
        filenames = sorted([os.path.basename(x) for x in filenames])
        path = os.path.dirname(path)
    filenames = filenames[::interval]
    return filenames

def set_camera(cfg):
    camera = bpy.data.objects['Camera']
    if not os.path.exists(cfg['camera_root']) and cfg['camera_root'].startswith('/Users/shuaiqing'):
        cfg['camera_root'] = cfg['camera_root'].replace('/Users/shuaiqing', '')

    cameras = read_camera(join(cfg['camera_root'], 'intri.yml'), join(cfg['camera_root'], 'extri.yml'))
    cam = cfg['cam']
    K = cameras[cam]['K']
    R = cameras[cam]['R']
    T = cameras[cam]['T']
    set_extrinsic(R, T, camera)
    res_x, res_y = cfg['res']
    set_intrinsic(K, camera, res_x, res_y)
    return camera, R, T

def parsing_frames(path, filenames, cfg, skeltype):
    frame = 0
    caches = {}
    camera, R, T = set_camera(cfg)
    for filename in filenames:
        print('Loading frame', frame)
        bpy.context.scene.frame_set(frame)
        record = read_skeleton(join(path, filename))
        if isinstance(record, list):
            pred = np.array([d['keypoints3d'] for d in record])[:, :15]
        else:
            pred = np.array(record[args.key])
        if pred.shape[-1] == 3:
            pred = np.dstack([pred, np.ones_like(pred[..., :1])])
        file_frame = int(filename.replace('.jpg', '').replace('.json', ''))
        for i in range(len(pred)):
            pid = i
            pred_ = np.array(pred[i])
            if pid not in caches:
                if frame > 1:continue
                color = cfg['color_table']
                color = [float(d)/255 for d in color[pid]]
                points, limbs = add_skeleton(pred_, pid=color, skeltype=skeltype, mode='ellipsold')
                caches[pid] = (points, limbs)
            points, limbs = caches[pid]
            update_skeleton(pred_, skeltype, points, limbs, frame)
        if file_frame in cfg['keyframes']:
            # 对于关键帧，增加相机模式
            center = pred[:, :, :3].mean(axis=0).mean(axis=0)
            if cfg['camera_mode'] == 'x': # 绕x轴旋转
                start_x = camera.rotation_euler[0]
                camera.keyframe_insert('location', frame=frame)
                camera.keyframe_insert('rotation_euler', frame=frame)
                start_frame = frame
                for _frame, delta in cfg['camera_keyframe'].items():
                    bpy.context.scene.frame_set(frame + _frame)
                    # 设置相机约束
                    camera.location = delta['location']
                    camera.rotation_euler = [np.deg2rad(d) for d in delta['rotation']]
                    # 设置旋转角度（以弧度为单位）
                    # 在动画时间轴上创建一个关键帧
                    camera.keyframe_insert(data_path="rotation_euler", frame=frame + _frame)
                    camera.keyframe_insert(data_path="location", frame=frame + _frame)
                frame += _frame * 2
                set_extrinsic(R, T, camera)
                camera.keyframe_insert('location', frame=frame)
                camera.keyframe_insert('rotation_euler', frame=frame)
                for _frame in range(start_frame, frame):
                    for pid, (points, limbs) in caches.items():
                        for p in limbs:
                            p.keyframe_insert(data_path='rotation_euler', frame=_frame)

        frame += 1
    bpy.context.scene.frame_end = frame
            

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--key', type=str, default='pred')
    parser.add_argument('--skel', type=str, default='panoptic15')
    parser.add_argument('--cfg', type=str, default='assets/config_skeleton.yml')
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--interval', type=int, default=1)
    args = parse_args(parser)
    cfg = load_yaml_cfg(args.cfg)
    filenames = load_skeletons_from_dir(args.path)
    if args.mode not in cfg.keys():
        print(cfg.keys())
        exit()

    setup(rgb=(1,1,1,0))
    cfg = cfg[args.mode]
    parsing_frames(args.path, filenames, cfg, args.skel)
    res_x, res_y = cfg['res']
    for _cfg in cfg['light']:
        _cfg['rotation'] = [np.deg2rad(d) for d in _cfg['rotation']]
        add_sunlight(**_cfg)
    if cfg['add_ground'] == True:
        size = 2
        build_plane(translation=(0, 0, 0), plane_size=size*2)
    
    if not args.nocycle:
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=args.num_samples,
            use_transparent_bg=True,
            use_denoising=True,
        )
    else:
        bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    format = cfg['format']
        
    if args.out is not None:
        set_output_properties(bpy.context.scene, output_file_path=args.out, 
            res_x=res_x, res_y=res_y, 
            tile_x=res_x, tile_y=res_y, resolution_percentage=100,
            format=format)
        if args.animation:
            bpy.ops.render.render(write_still=True, animation=True)