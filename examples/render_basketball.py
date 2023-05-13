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

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--skel', type=str, default='panoptic15')
    parser.add_argument('--grid', type=str, default=None)
    parser.add_argument('--offset', type=float, default=[0., 0., 0.], nargs=3)
    parser.add_argument('--ground', action='store_true')
    parser.add_argument('--no_gt', action='store_true')
    parser.add_argument('--no_pred', action='store_true')
    parser.add_argument('--field', type=str, default=None)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--gamma_density', type=float, default=1)
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))
    camera = set_camera(location=(0, -2, 2.5), center=(0, 0, 1), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., -np.pi/4, 0))

    K = np.array([1108.131, 0.000, 461.170, 0.000, 1105.366, 449.599, 0.000, 0.000, 1.000]).reshape(3, 3)
    R = np.array([-0.864, 0.501, 0.055, 0.093, 0.266, -0.960, -0.495, -0.824, -0.276]).reshape(3, 3)
    T = np.array([-0.065, 0.398, 3.956]).reshape(3, 1)

    set_intrinsic(K, camera, sensor_width=1.0)
    set_extrinsic(R, T, camera)

    # 载入篮球场
    bpy.ops.import_scene.fbx(filepath='assets/Basketball_15x28.fbx')
    bpy.data.objects['Pole_basketball 15x28'].location = (14, 0, 0)

    if os.path.isfile(args.path):
        record = read_skeleton(args.path)

        if not args.no_pred:
            pred = np.array(record['pred'])
            pids = record['pids']
            for i in range(pred.shape[0]):
                pred_ = pred[i]
                pred_[:, :3] = pred_[:, :3]
                add_skeleton(pred_, pid=color_table[i], skeltype=args.skel, mode='ellipsold')
    elif os.path.isdir(args.path):
        filenames = sorted(os.listdir(args.path))
        bpy.context.scene.frame_end = len(filenames) - 1
        caches = {}
        for frame, filename in enumerate(filenames):
            print('frame', frame)
            bpy.context.scene.frame_set(frame)
            record = read_skeleton(join(args.path, filename))
            pred = np.array(record['pred'])
            pids = record['pids']
            for i in range(pred.shape[0]):
                pid = pids[i]
                if pid >= len(color_table):
                    continue
                pred_ = pred[i]
                if pid not in caches:
                    if frame > 1:continue
                    points, limbs = add_skeleton(pred_, pid=color_table[i], skeltype=args.skel, mode='ellipsold')
                    caches[pids[i]] = (points, limbs)
                points, limbs = caches[pid]
                update_skeleton(pred_, args.skel, points, limbs, frame)
        cameras = {
            0: {'location': [18, -7, 6.8], 'rotation': [71, 0, 63]},
            40: {'location': [4.8, -6, 2], 'rotation': [76, 0, 26]},
            149: {'location': [6, -10, 3], 'rotation': [75, 0, 0]},
        }
        camera_obj = bpy.data.objects['Camera']
        for frame, camera in cameras.items():
            bpy.context.scene.frame_set(frame)
            camera_obj.location = camera['location']
            camera_obj.rotation_euler = np.array(camera['rotation']) / 180. * np.pi
            # obj = set_camera(location=camera['location'], rotation=camera['rotation'], focal=30)
            camera_obj.keyframe_insert('location', frame=frame)
            camera_obj.keyframe_insert('rotation_euler', frame=frame)
            
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
        set_output_properties(bpy.context.scene, output_file_path=args.out, 
            res_x=args.res_x, res_y=args.res_y, 
            tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
            format=args.format)
        bpy.ops.render.render(write_still=True, animation=True)
    # if args.out_blend is not None:
    #     bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)