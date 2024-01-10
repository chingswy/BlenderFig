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
from tqdm import tqdm
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
from myblender.geometry import create_plane, create_points

def create_pointcloud_mesh(points, colors):
    # Create new mesh and object
    mesh = bpy.data.meshes.new(name="PointCloud")
    obj = bpy.data.objects.new("PointCloud", mesh)

    # Link object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # Fill mesh with vertices
    mesh.from_pydata(points, [], [])
    # Enter edit mode to create mesh
    # bpy.context.view_layer.objects.active = obj
    # bpy.ops.object.mode_set(mode='EDIT')


    # Create vertex color layer
    # If colors are found, create a new Attribute 'Col' to hold them (NOT the Vertex_Color block!)
    # TODO: Make this more Pythonic
    attr = mesh.attributes.new(name="Col", type='FLOAT_COLOR', domain='POINT')
    for i, col in enumerate(colors):
        print(col)
        print(attr.data)
        attr.data[i].color[0] = col[0]
        attr.data[i].color[1] = col[1]
        attr.data[i].color[2] = col[2]

    # Update mesh with new data
    mesh.update()
    mesh.validate()

if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()
    set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    data = np.loadtxt(args.path)
    data = data[::10]
    points, alpha, color, radius = data[:, :3], data[:, 3], data[:, 4:7], data[:, 7]
    # create_pointcloud_mesh(points, color)
    for i in tqdm(range(points.shape[0])):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=4, ring_count=4, radius=radius[i], location=points[i], enter_editmode=False, align='WORLD')
        obj = bpy.context.object
        # Create a new material
        mat = bpy.data.materials.new(name="MyMaterial")

        # Enable 'Use Nodes'
        mat.use_nodes = True

        # Get the material's node tree
        nodes = mat.node_tree.nodes

        # Set base color
        nodes["Principled BSDF"].inputs["Base Color"].default_value = (*color[i], alpha[i])  # Red color

        obj.data.materials.append(mat)

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
    # bpy.ops.render.render(write_still=True, animation=False)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)