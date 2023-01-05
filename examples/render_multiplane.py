'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''
# TODO: This scripts show how to use blender to render a cube
from os.path import join
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    create_image_corners
)
from myblender.color import color_jet
from myblender.imageio import imwrite
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()
    set_camera(location=(4, 0, 2.5), center=(0, 0, 0.), focal=20)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))
    x_min = -2.
    y_min = -2.
    x_max = 2.
    y_max = 2.
    xy = np.array([
        [x_min, y_min], 
        [x_max, y_min], 
        [x_min, y_max], 
        [x_max, y_max]
    ])
    sample_step = 0.01
    center = np.array([
        [-0.3, -1.1, 0.9],
        [0.5, 0.1, 1.1]
    ])
    # from PIL import Image
    zs = [0., 1., 2.]
    for z in zs:
        corners = np.hstack([xy, np.ones_like(xy[:, :1])*z])
        x_ = np.linspace(x_min, x_max, num=int((x_max-x_min)/sample_step))
        y_ = np.linspace(y_min, y_max, num=int((y_max-y_min)/sample_step))
        x_, y_ = np.meshgrid(x_, y_)
        xy_ = np.stack([x_, y_], axis=-1)
        xyz = np.concatenate([xy_, np.ones_like(xy_[..., :1])*z], axis=-1)
        distance = np.linalg.norm(xyz[None] - center[:, None, None], axis=-1).min(axis=0)
        print(distance.shape)
        affinity = np.exp(-distance**2/(2*0.5**2))
        affinity = (affinity * 255).astype(np.int)
        rgb = color_jet[affinity]
        outname = join('tmp', '{}.jpg'.format(z))
        imwrite(outname, rgb)
        create_image_corners(outname, corners)
    
    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=False,
        use_denoising=args.denoising,
    )

    n_parallel = 1
    set_output_properties(bpy.context.scene, output_file_path=args.out, 
        res_x=args.res_x, res_y=args.res_y, 
        tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
        format='JPEG')
    
    bpy.ops.render.render(write_still=True, animation=False)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)