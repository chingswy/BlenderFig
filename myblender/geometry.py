'''
  @ Date: 2022-04-24 15:39:58
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-11-15 15:13:34
  @ FilePath: /EasyMocapPublic/easymocap/blender/geometry.py
'''
import bpy
import os
from os.path import join
from .material import get_rgb, set_material_i, set_principled_node, add_material
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler

log = print

assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'objs'))



def myimport(filename):
    keys_old = set(bpy.data.objects.keys())
    mat_old = set(bpy.data.materials.keys())
    image_old = set(bpy.data.images.keys())
    if filename.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=filename, axis_forward='X', axis_up='Z')
    keys_new = set(bpy.data.objects.keys())
    mat_new = set(bpy.data.materials.keys())
    image_new = set(bpy.data.images.keys())
    key = list(keys_new - keys_old)[0]
    current_obj = bpy.data.objects[key]
    # set default rotation to 0.
    current_obj.rotation_euler = (0., 0., 0.)
    key_image = list(image_new-image_old)
    if len(key_image) > 0:
        print('>>> Loading image {}'.format(key_image[0]))
        key = (key, key_image[0])
    mat = list(mat_new - mat_old)[0]
    return current_obj, key, mat

def create_any_mesh(filename, vid, scale=(1, 1, 1), 
    rotation=(0., 0., 0.), location=(0, 0, 0), shadow=True, **kwargs):
    cylinder, name, matname = myimport(filename)
    cylinder.scale = scale
    cylinder.rotation_euler = rotation
    cylinder.location = location
    set_material_i(bpy.data.materials[matname], vid, **kwargs)
    if not shadow:
        bpy.data.materials[matname].shadow_method = 'NONE'
        cylinder.cycles_visibility.shadow = False
    return cylinder

def create_cylinder(vid, **kwargs):
    create_any_mesh(join(assets_dir, 'cylinder_100.obj'), vid, **kwargs)

def create_plane(vid, radius=1, center=(0, 0), **kwargs):
    scale = (radius*2, radius*2, 0.02)
    # 注意：方向有点反
    location = (center[0]+radius, center[1]-radius, 0)
    create_any_mesh(join(assets_dir, 'cube.obj'), vid=vid,
        scale=scale, location=location, **kwargs)

def create_points(vid, radius=1, center=(0, 0, 0), basename='sphere.obj', **kwargs):
    scale = (radius, radius, radius)
    create_any_mesh(join(assets_dir, basename), vid=vid,
        scale=scale, location=center, **kwargs)

def create_sample_points(start, end, N_sample, radius=0.01, vid=0, **kwargs):
    start, end = np.array(start), np.array(end)
    dir = end - start
    for i in range(N_sample+1): # create 3 layer
        location = start + dir * i/N_sample
        create_points(vid, center=location, radius=radius, **kwargs)

def create_halfcylinder(vid, radius=1, height=2, **kwargs):
    scale = (radius, radius, height/2)
    location = (0, 0, height/2)
    create_any_mesh(join(assets_dir, 'halfcylinder_100.obj'), vid=vid,
        scale=scale, location=location, **kwargs)

def create_arrow(vid, start, end, 
    cylinder_radius=0.2, cone_radius=0.3,
    cylinder_height=0.6, cone_height=0.4,):
    scale = (cylinder_radius, cylinder_radius, cylinder_height)

def look_at(obj_camera, point):
    loc_camera = obj_camera.location
    direction = Vector(point - loc_camera)
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

def create_line(vid, radius, start=(0., 0., 0.), end=(1., 1., 1.)):
    start, end = np.array(start), np.array(end)
    length = np.linalg.norm(end - start)
    scale = (radius, radius, length/2)
    dir = end - start
    dir /= np.linalg.norm(dir)
    location = start + (end - start) / 2
    cylinder = create_any_mesh(join(assets_dir, 'cylinder_100.obj'), vid,
        scale=scale, location=location, shadow=True)
    look_at(cylinder, end)

def create_ray(vid, start=(0., 0., 0.), end=(1., 1., 1.), 
    cone_radius=0.03, cone_height=0.1,
    cylinder_radius=0.01):
    start, end = np.array(start), np.array(end)
    length = np.linalg.norm(end - start)
    scale = (cylinder_radius, cylinder_radius, length/2)
    dir = end - start
    dir /= np.linalg.norm(dir)
    location = start + (end - start) / 2
    # disable shadow for ray
    cylinder = create_any_mesh(join(assets_dir, 'cylinder_100.obj'), vid,
        scale=scale, location=location, shadow=False)
    cone_scale = (cone_radius, cone_radius, cone_height)
    cone_location = end + dir * cone_height * 0.01
    cone = create_any_mesh(join(assets_dir, 'cone_100.obj'), vid,
        scale=cone_scale, location=cone_location, shadow=False)
    look_at(cylinder, end)
    look_at(cone, end)
    # set_material_rgb(bpy.data.materials[matname], [0, 0, 0])

def _create_image(imgname):
    filename = join(assets_dir, 'background.obj')
    image_mesh, name, matname = myimport(filename)
    key, key_image = name
    bpy.data.images[key_image].filepath = imgname
    return image_mesh

def create_image_corners(imgname, corners):
    image_mesh = _create_image(imgname)
    vertices = image_mesh.data.vertices
    assert len(vertices) == len(corners), len(vertices)
    for i in range(corners.shape[0]):
        vertices[i].co = corners[i]

def create_image_euler_translation(imgname, euler, translation,scale):
    image_mesh = _create_image(imgname)
    image_mesh.scale = scale
    image_mesh.rotation_euler = euler
    image_mesh.location = translation

def create_image_RT(imgname, R=np.eye(3), T=np.zeros((3, 1))):
    image_mesh = _create_image(imgname)
    image_mesh.rotation_euler = Matrix(R.T).to_euler()
    center = - R.T @ T
    image_mesh.location = (center[0, 0], center[1, 0], center[2, 0])

def set_camera(height=5., radius = 9, focal=40, center=(0., 0., 0.),
    location=None, rotation=None):
    camera = bpy.data.objects['Camera']
    # theta = np.pi / 8
    if location is None:
        theta = 0.
        camera.location = (radius * np.sin(theta), -radius * np.cos(theta), height)
    else:
        camera.location = location
    if rotation is None:
        look_at(camera, Vector(center))
    else:
        camera.rotation_euler = Euler(rotation, 'XYZ')
    print(camera.location)
    print(camera.rotation_euler)
    camera.data.lens = focal
    return camera

def build_checker_board_nodes(node_tree: bpy.types.NodeTree, size: float, alpha: float=1.) -> None:
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    checker_texture_node = node_tree.nodes.new(type='ShaderNodeTexChecker')

    set_principled_node(principled_node=principled_node, alpha=alpha)
    checker_texture_node.inputs['Scale'].default_value = size

    node_tree.links.new(checker_texture_node.outputs['Color'], principled_node.inputs['Base Color'])
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    # node_tree.nodes["Checker Texture"].inputs[1].default_value = (1, 1, 1, 1)
    # node_tree.nodes["Checker Texture"].inputs[2].default_value = (0.3, 0.8, 0.2, 1)

    # arrange_nodes(node_tree)

def create_plane_blender(location = (0.0, 0.0, 0.0),
                 rotation = (0.0, 0.0, 0.0),
                 size = 2.0,
                 name = None) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)

    current_object = bpy.context.object

    if name is not None:
        current_object.name = name

    return current_object

def build_plane(translation=(-1., -1., 0.), plane_size = 8., alpha=1):
    plane = create_plane_blender(size=plane_size, name="Floor")
    plane.location = translation
    floor_mat = add_material("Material_Plane", use_nodes=True, make_node_tree_empty=True)
    build_checker_board_nodes(floor_mat.node_tree, plane_size, alpha=alpha)
    plane.data.materials.append(floor_mat)

def bound_from_keypoint(keypoint, padding=0.1, min_z=0):
    v = keypoint[..., -1]
    k3d_flat = keypoint[v>0.01]
    lower = k3d_flat[:, :3].min(axis=0)
    lower[2] = max(min_z, lower[2])
    upper = k3d_flat[:, :3].max(axis=0)
    center = (lower + upper ) / 2
    scale = upper - lower
    return center, scale, np.stack([lower, upper])

def create_bbox3d(scale=(1., 1., 1.), location=(0., 0., 0.), pid=0):
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD')
    bpy.ops.object.modifier_add(type='WIREFRAME')
    obj = bpy.context.object
    obj.modifiers["Wireframe"].thickness = 0.04
    name = obj.name
    
    matname = "Material_{}".format(name)
    mat = add_material(matname, use_nodes=True, make_node_tree_empty=False)
    obj.data.materials.append(mat)

    obj.rotation_euler = (0, 0, 0)
    set_material_i(bpy.data.materials[matname], pid)
    obj.scale = scale
    obj.location = location
    try:
        obj.cycles_visibility.shadow = False
    except:
        print('Cannot set cycle')

def setMat_plastic(mesh, meshColor, AOStrength = 0.0):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 0.3
	tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
	tree.nodes["Principled BSDF"].inputs['Specular'].default_value = 0.5
	tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
	tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = 0
	tree.nodes["Principled BSDF"].inputs['Clearcoat Roughness'].default_value = 0

	# add Ambient Occlusion
	tree.nodes.new('ShaderNodeAmbientOcclusion')
	tree.nodes.new('ShaderNodeGamma')
	MIXRGB = tree.nodes.new('ShaderNodeMixRGB')
	MIXRGB.blend_type = 'MULTIPLY'
	tree.nodes["Gamma"].inputs["Gamma"].default_value = AOStrength
	tree.nodes["Ambient Occlusion"].inputs["Distance"].default_value = 10.0
	tree.nodes["Gamma"].location.x -= 600

	# set color using Hue/Saturation node
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	HSVNode.inputs['Color'].default_value = meshColor.RGBA
	HSVNode.inputs['Saturation'].default_value = meshColor.S
	HSVNode.inputs['Value'].default_value = meshColor.V
	HSVNode.inputs['Hue'].default_value = meshColor.H
	HSVNode.location.x -= 200

	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = meshColor.B
	BCNode.inputs['Contrast'].default_value = meshColor.C
	BCNode.location.x -= 400

	# link all the nodes
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Ambient Occlusion'].inputs['Color'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['Color'], MIXRGB.inputs['Color1'])
	tree.links.new(tree.nodes["Ambient Occlusion"].outputs['AO'], tree.nodes['Gamma'].inputs['Color'])
	tree.links.new(tree.nodes["Gamma"].outputs['Color'], MIXRGB.inputs['Color2'])
	tree.links.new(MIXRGB.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

def load_humanmesh(filename, pid, meshColor=None, translation=None, rotation=None, with_bbox=True):
    assert filename.endswith('.obj'), filename
    obj, name, matname = myimport(filename)
    obj.rotation_euler = (0, 0, +0)
    if rotation is not None:
        obj.rotation_euler = (rotation[0], rotation[1], rotation[2])
        # if with_bbox:
        #     box = np.array(obj.bound_box)
        #     box_min = np.min(box, axis=0)
        #     box_max = np.max(box, axis=0)
        #     box_center = (box_min + box_max) / 2
        #     box_scale = (box_max - box_min) / 2
        #     if translation is not None:
        #         box_center += np.array(translation)
        #     locations.append(np.array(box_center))
        #     load_bbox(scale=box_scale, location=box_center, pid=pid)
        # else:
        #     locations.append(np.array(obj.location))
    if translation is not None:
        obj.location += Vector(translation)
    bpy.ops.object.shade_smooth()
    setMat_plastic(obj, meshColor)
    # mat = bpy.data.materials[matname]
    # set_material_i(mat, pid)
    return obj