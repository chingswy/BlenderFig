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
import os
from myblender.geometry import (
    set_camera,
    build_plane,
    create_any_mesh
)
from myblender.color import color_jet

from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from blender.geometry import create_plane, create_points

def create_grid(bounds, N=64, MIN_VIS_THRES=0.1):
    x = np.arange(N)
    xyz = np.meshgrid(x, x, x)
    xyz = np.stack(xyz, axis=-1).reshape(-1, 3) / N
    xyz = xyz * (bounds[1] - bounds[0]) + bounds[0]
    print(xyz.shape)
    center = np.mean(xyz, axis=0, keepdims=True)
    dist = np.linalg.norm(xyz - center, axis=1)
    sigma2 = 0.5**2
    gauss = np.exp(-dist**2 / sigma2)
    valid = gauss > MIN_VIS_THRES
    return xyz[valid], gauss[valid]
    print(gauss.shape, xyz.shape)
    return xyz, gauss

def plot_grids(grids, confs, radius):
    for grid, conf in zip(grids, confs):
        conf_int = int(conf * 255)
        create_points(vid=color_jet[conf_int], center=grid, radius=radius, alpha=conf,
            basename='sphere_8.obj')

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()
    set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=50)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    bounds = np.array([[-1., -1., 0.], [1., 1., 2.]])
    N = 8
    if args.path == 'debug':
        grids, confs = create_grid(bounds, N=N)
        radius = ((bounds[1][0]-bounds[0][0])/N/4)
    else:
        grids = np.loadtxt(args.path)
        print(grids.shape)
        grids, confs = grids[:, :3], grids[:, 3]
        radius = 0.02
    plot_grids(grids, confs, radius=radius)
    for pid, meshname in enumerate(args.extra_mesh):
        if not os.path.exists(meshname):
            continue
        create_any_mesh(meshname, vid=pid)

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