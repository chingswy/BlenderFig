'''
  @ Date: 2022-04-24 15:39:58
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-11-15 15:13:34
  @ FilePath: /EasyMocapPublic/easymocap/blender/geometry.py
'''
import bpy
import bmesh
import os
import math
from os.path import join
from .material import get_rgb, set_material_i, set_principled_node, add_material, setMat_plastic
import numpy as np
from mathutils import Matrix, Vector, Quaternion, Euler

log = print

assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'objs'))

print(assets_dir)
print(sorted(os.listdir(assets_dir)))


def myimport(filename):
    keys_old = set(bpy.data.objects.keys())
    mat_old = set(bpy.data.materials.keys())
    image_old = set(bpy.data.images.keys())
    assert os.path.exists(filename), filename
    if filename.endswith('.obj'):
        # Blender 3.3 中 wm.obj_import 不支持 axis_forward 和 axis_up 参数
        # 使用默认参数导入，轴方向通过后续的旋转调整
        bpy.ops.wm.obj_import(filepath=filename)
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
    
    # 检查是否有新材质被创建，如果没有则创建一个默认材质
    mat_diff = list(mat_new - mat_old)
    if len(mat_diff) > 0:
        mat = mat_diff[0]  # 材质名称（字符串）
    else:
        # 如果没有材质，创建一个默认材质并分配给对象
        mat_name = "Material_{}".format(key if isinstance(key, str) else key[0])
        mat_obj = add_material(mat_name, use_nodes=True, make_node_tree_empty=False)
        # 确保对象有材质槽
        if len(current_obj.data.materials) == 0:
            current_obj.data.materials.append(mat_obj)
        else:
            current_obj.data.materials[0] = mat_obj
        current_obj.active_material = mat_obj
        mat = mat_name  # 返回材质名称而不是对象
        print('>>> Created default material {} for object {}'.format(mat_name, key))
    
    return current_obj, key, mat

def create_any_mesh(filename, vid, scale=(1, 1, 1),
    rotation=(0., 0., 0.), location=(0, 0, 0), shadow=True, **kwargs):
    cylinder, name, matname = myimport(filename)
    cylinder.scale = scale
    cylinder.rotation_euler = rotation
    cylinder.location = location
    # set_material_i(bpy.data.materials[matname], vid, **kwargs)
    set_material_i(cylinder, vid, **kwargs)
    if not shadow:
        try:
            # 确保材质存在
            if matname in bpy.data.materials:
                bpy.data.materials[matname].shadow_method = 'NONE'
        except (AttributeError, KeyError):
            # shadow_method不存在于当前Blender版本中，或材质不存在
            pass
        try:
            cylinder.cycles_visibility.shadow = False
        except:
            pass
    return cylinder

def create_cylinder(vid, **kwargs):
    create_any_mesh(join(assets_dir, 'cylinder_100.obj'), vid, **kwargs)

def create_plane(vid, radius=1, center=(0, 0), **kwargs):
    scale = (radius*2, radius*2, 0.02)
    # 注意：方向有点反
    location = (center[0]+radius, center[1]-radius, 0)
    return create_any_mesh(join(assets_dir, 'cube.obj'), vid=vid,
        scale=scale, location=location, **kwargs)

def create_points(vid, radius=1, center=(0, 0, 0), basename='sphere.obj', **kwargs):
    scale = (radius, radius, radius)
    return create_any_mesh(join(assets_dir, basename), vid=vid,
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
    cylinder_radius=0.02, cone_radius=0.04, cone_height=0.1,
    shadow=False, **kwargs):
    """Create an arrow from start to end with a cylinder shaft and cone head.

    Args:
        vid: Material/color index
        start: Starting position (x, y, z)
        end: End position (x, y, z)
        cylinder_radius: Radius of the arrow shaft
        cone_radius: Radius of the arrow head base
        cone_height: Height of the arrow head cone
        shadow: Whether to cast shadows
        **kwargs: Additional arguments passed to material setup

    Returns:
        Tuple of (cylinder, cone) objects
    """
    start, end = np.array(start), np.array(end)
    length = np.linalg.norm(end - start)

    if length < 1e-6:
        return None, None

    direction = (end - start) / length

    # Cylinder shaft (from start to end minus cone height)
    shaft_length = max(length - cone_height, 0.01)
    shaft_end = start + direction * shaft_length
    shaft_center = (start + shaft_end) / 2

    # Create cylinder
    cylinder_scale = (cylinder_radius, cylinder_radius, shaft_length / 2)
    cylinder = create_any_mesh(join(assets_dir, 'cylinder_100.obj'), vid,
        scale=cylinder_scale, location=shaft_center, shadow=shadow, **kwargs)
    orient_along_direction(cylinder, direction)

    # Cone head at the end (cone origin is at its base, not center)
    cone_center = shaft_end
    cone_scale = (cone_radius, cone_radius, cone_height / 2)
    cone = create_any_mesh(join(assets_dir, 'cone_100.obj'), vid,
        scale=cone_scale, location=cone_center, shadow=shadow, **kwargs)
    orient_along_direction(cone, direction)

    return cylinder, cone


def _set_simple_color(obj, color):
    """Set a simple solid color on an object using Principled BSDF.
    
    Args:
        obj: Blender mesh object
        color: RGB tuple (r, g, b) with values 0-1
    """
    mat = bpy.data.materials.new(name="SimpleColor")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.0
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    obj.active_material = mat


def create_simple_arrow(start, end, color=(1, 0, 0),
    cylinder_radius=0.02, cone_radius=0.04, cone_height=0.1):
    """Create a simple arrow with pure RGB color.

    Args:
        start: Starting position (x, y, z)
        end: End position (x, y, z)
        color: RGB tuple (r, g, b) with values 0-1, e.g. (1,0,0) for red
        cylinder_radius: Radius of the arrow shaft
        cone_radius: Radius of the arrow head base
        cone_height: Height of the arrow head cone

    Returns:
        Tuple of (cylinder, cone) objects
    """
    start, end = np.array(start), np.array(end)
    length = np.linalg.norm(end - start)

    if length < 1e-6:
        return None, None

    direction = (end - start) / length

    # Cylinder shaft
    shaft_length = max(length - cone_height, 0.01)
    shaft_end = start + direction * shaft_length
    shaft_center = (start + shaft_end) / 2

    # Create cylinder using Blender primitive
    bpy.ops.mesh.primitive_cylinder_add(
        radius=cylinder_radius,
        depth=shaft_length,
        location=shaft_center
    )
    cylinder = bpy.context.object
    orient_along_direction(cylinder, direction)
    _set_simple_color(cylinder, color)
    cylinder.visible_shadow = False

    # Create cone
    bpy.ops.mesh.primitive_cone_add(
        radius1=cone_radius,
        radius2=0,
        depth=cone_height,
        location=shaft_end + direction * cone_height / 2
    )
    cone = bpy.context.object
    orient_along_direction(cone, direction)
    _set_simple_color(cone, color)
    cone.visible_shadow = False

    return cylinder, cone


def create_coordinate_axes(origin=(0, 0, 0), length=1.0,
    cylinder_radius=0.02, cone_radius=0.04, cone_height=0.1):
    """Create RGB coordinate axes (X=Red, Y=Green, Z=Blue).

    Args:
        origin: Origin point of the axes
        length: Length of each axis
        cylinder_radius: Radius of the arrow shaft
        cone_radius: Radius of the arrow head base
        cone_height: Height of the arrow head cone

    Returns:
        Dict with 'x', 'y', 'z' keys, each containing (cylinder, cone) tuple
    """
    origin = np.array(origin)
    
    axes = {}
    # X axis - Red
    axes['x'] = create_simple_arrow(
        origin, origin + np.array([length, 0, 0]),
        color=(1, 0, 0),
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cone_height=cone_height
    )
    # Y axis - Green
    axes['y'] = create_simple_arrow(
        origin, origin + np.array([0, length, 0]),
        color=(0, 1, 0),
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cone_height=cone_height
    )
    # Z axis - Blue
    axes['z'] = create_simple_arrow(
        origin, origin + np.array([0, 0, length]),
        color=(0, 0, 1),
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cone_height=cone_height
    )
    
    return axes


def orient_along_direction(obj, direction):
    """Orient an object (cylinder/cone along Z-axis) to point in the given direction.

    Args:
        obj: Blender object to orient
        direction: Target direction as numpy array or tuple (dx, dy, dz)
    """
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    # Default cylinder/cone is along Z-axis (0, 0, 1)
    z_axis = Vector((0, 0, 1))
    dir_vec = Vector(direction)

    # Compute rotation quaternion from Z to target direction
    rot_quat = z_axis.rotation_difference(dir_vec)
    obj.rotation_euler = rot_quat.to_euler()


def look_at(obj_camera, point):
    loc_camera = obj_camera.location
    direction = Vector(point - loc_camera)
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

def create_line(vid, radius, start=(0., 0., 0.), end=(1., 1., 1.), **kwargs):
    """Create a line (cylinder) from start to end.

    Args:
        vid: Material/color index
        radius: Radius of the line
        start: Starting position (x, y, z)
        end: End position (x, y, z)
        **kwargs: Additional arguments passed to material setup

    Returns:
        The cylinder object
    """
    start, end = np.array(start), np.array(end)
    length = np.linalg.norm(end - start)
    if length < 1e-6:
        return None
    direction = (end - start) / length
    scale = (radius, radius, length / 2)
    location = (start + end) / 2
    cylinder = create_any_mesh(join(assets_dir, 'cylinder_100.obj'), vid,
        scale=scale, location=location, shadow=True, **kwargs)
    orient_along_direction(cylinder, direction)
    return cylinder

def create_ellipsold(vid, radius, start=(0., 0., 0.), end=(1., 1., 1.), **kwargs):
    """Create an ellipsoid stretched from start to end.

    Args:
        vid: Material/color index
        radius: Base radius of the ellipsoid
        start: Starting position (x, y, z)
        end: End position (x, y, z)
        **kwargs: Additional arguments passed to material setup

    Returns:
        The ellipsoid object
    """
    start, end = np.array(start), np.array(end)
    length = np.linalg.norm(end - start)
    if length < 1e-6:
        return None
    direction = (end - start) / length
    radius = radius * length / 0.2
    scale = (radius, radius, length / 2)
    location = (start + end) / 2
    ellipsoid = create_any_mesh(join(assets_dir, 'sphere.obj'), vid,
        scale=scale, location=location, **kwargs)
    orient_along_direction(ellipsoid, direction)
    return ellipsoid

def create_ray(vid, start=(0., 0., 0.), end=(1., 1., 1.),
    cone_radius=0.03, cone_height=0.1,
    cylinder_radius=0.01, shadow=False, **kwargs):
    """Create a ray (arrow) from start to end.

    This is an alias for create_arrow with default parameters suitable for rays.

    Args:
        vid: Material/color index
        start: Starting position (x, y, z)
        end: End position (x, y, z)
        cone_radius: Radius of the arrow head base
        cone_height: Height of the arrow head cone
        cylinder_radius: Radius of the ray shaft
        shadow: Whether to cast shadows (default False for rays)
        **kwargs: Additional arguments passed to material setup

    Returns:
        Tuple of (cylinder, cone) objects
    """
    return create_arrow(
        vid=vid,
        start=start,
        end=end,
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cone_height=cone_height,
        shadow=shadow,
        **kwargs
    )

def _create_image(imgname, remove_shadow=False):
    filename = join(assets_dir, 'background.obj')
    image_mesh, name, matname = myimport(filename)
    key, key_image = name
    bpy.data.images[key_image].filepath = imgname
    obj = bpy.data.objects[key]
    mat = obj.active_material
    nodes = mat.node_tree.nodes
    if remove_shadow:
        links = mat.node_tree.links
        out = nodes[1]
        image = nodes[2]
        node = nodes.new('ShaderNodeBackground')
        links.new(image.outputs[0], node.inputs[0])
        links.new(node.outputs[0], out.inputs[0])
    return image_mesh

def create_image_corners(imgname, corners, remove_shadow=False):
    image_mesh = _create_image(imgname, remove_shadow)
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
    location=None, rotation=None, frame=None,
    camera=None):
    if camera is None:
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
    print(f'Set camera.location: {camera.location}')
    print(f'Set camera.rotation_euler: {camera.rotation_euler}')
    camera.data.lens = focal
    if frame is not None:
        print(f"Added keyframe at frame {frame} for camera position and rotation")
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    return camera
    

def build_checker_board_nodes(node_tree: bpy.types.NodeTree, size: float, alpha: float=1.,
    white_color=(1, 1, 1, 1), black_color=(0, 0, 0, 1),
    roughness: float=0.5, metallic: float=0.0, specular: float=0.5) -> None:
    """
    Build checkerboard nodes for a plane.

    Args:
        node_tree: The node tree to build nodes in
        size: Scale of the checker pattern
        alpha: Transparency (1.0 = opaque)
        white_color: Color for white squares
        black_color: Color for black squares
        roughness: Surface roughness (0.0 = mirror-like, 1.0 = matte)
        metallic: Metallic property (0.0 = dielectric, 1.0 = metallic)
        specular: Specular reflection intensity
    """
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    checker_texture_node = node_tree.nodes.new(type='ShaderNodeTexChecker')

    set_principled_node(principled_node=principled_node, alpha=alpha,
                       roughness=roughness, metallic=metallic, specular=specular)
    checker_texture_node.inputs['Scale'].default_value = size

    node_tree.links.new(checker_texture_node.outputs['Color'], principled_node.inputs['Base Color'])
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    node_tree.nodes["Checker Texture"].inputs[1].default_value = white_color
    node_tree.nodes["Checker Texture"].inputs[2].default_value = black_color

    # arrange_nodes(node_tree)

def build_checker_board_transparent_nodes(node_tree: bpy.types.NodeTree, size: float, alpha: float=1.) -> None:
    output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = node_tree.nodes.new(type='ShaderNodeBsdfTransparent')
    checker_texture_node = node_tree.nodes.new(type='ShaderNodeTexChecker')

    # set_principled_node(principled_node=principled_node, alpha=alpha)
    checker_texture_node.inputs['Scale'].default_value = size

    node_tree.links.new(checker_texture_node.outputs['Color'], principled_node.inputs['Color'])
    node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    node_tree.nodes["Checker Texture"].inputs[1].default_value = (1, 1, 1, 1)
    node_tree.nodes["Checker Texture"].inputs[2].default_value = (0, 0, 0, 1)

    # arrange_nodes(node_tree)

def create_plane_blender(location = (0.0, 0.0, 0.0),
                 rotation = (0.0, 0.0, 0.0),
                 size = 2.0,
                 name = None,
                 shadow=True) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)

    current_object = bpy.context.object

    if name is not None:
        current_object.name = name
    if not shadow:
        if hasattr(current_object, 'visible_shadow'):
            current_object.visible_shadow = False
        else:
            current_object.cycles_visibility.shadow = False
    return current_object

def build_plane(translation=(-1., -1., 0.), plane_size = 8., alpha=1, use_transparent=False,
                white=(1.,1.,1.,1.), black=(0.,0.,0.,0.),
                roughness=0.5, metallic=0.0, specular=0.5):
    """
    Build a checkerboard plane.

    Args:
        translation: Position of the plane
        plane_size: Size of the plane
        alpha: Transparency (1.0 = opaque)
        use_transparent: Use transparent shader instead of principled BSDF
        white: Color for white squares
        black: Color for black squares
        roughness: Surface roughness (0.0 = mirror-like reflection, 1.0 = matte)
        metallic: Metallic property (0.0 = dielectric, 1.0 = metallic)
        specular: Specular reflection intensity
    """
    plane = create_plane_blender(size=plane_size, name="Floor")
    plane.location = translation
    floor_mat = add_material("Material_Plane", use_nodes=True, make_node_tree_empty=True)
    if use_transparent:
        build_checker_board_transparent_nodes(floor_mat.node_tree, plane_size, alpha=alpha,
        white_color=white, black_color=black)
    else:
        build_checker_board_nodes(floor_mat.node_tree, plane_size, alpha=alpha,
        white_color=white, black_color=black,
        roughness=roughness, metallic=metallic, specular=specular)
    plane.data.materials.append(floor_mat)
    return plane

def bound_from_keypoint(keypoint, padding=0.1, min_z=0):
    v = keypoint[..., -1]
    k3d_flat = keypoint[v>0.01]
    lower = k3d_flat[:, :3].min(axis=0)
    lower[2] = max(min_z, lower[2])
    upper = k3d_flat[:, :3].max(axis=0)
    center = (lower + upper ) / 2
    scale = upper - lower
    return center, scale, np.stack([lower, upper])

def create_bbox3d(scale=(1., 1., 1.), location=(0., 0., 0.), rotation=None, pid=0):
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD')
    bpy.ops.object.modifier_add(type='WIREFRAME')
    obj = bpy.context.object
    obj.modifiers["Wireframe"].thickness = 0.04
    name = obj.name

    matname = "Material_{}".format(name)
    mat = add_material(matname, use_nodes=True, make_node_tree_empty=False)
    obj.data.materials.append(mat)
    if rotation is not None:
        obj.rotation_mode = 'QUATERNION'
        obj.rotation_quaternion = rotation
    else:
        obj.rotation_euler = (0, 0, 0)
    set_material_i(bpy.data.materials[matname], pid, use_plastic=False)
    obj.scale = scale
    obj.location = location
    if hasattr(obj, 'visible_shadow'):
        obj.visible_shadow = False
    try:
        obj.cycles_visibility.shadow = False
    except:
        print('Cannot set cycle')

def add_material_to_blender_primitive(obj, pid):
    name = obj.name
    matname = "Material_{}".format(name)
    mat = add_material(matname, use_nodes=True, make_node_tree_empty=False)
    obj.data.materials.append(mat)

    set_material_i(bpy.data.materials[matname], pid, use_plastic=False)

def create_camera_blender(R, T, scale=0.1, pid=0):
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD')
    obj = bpy.context.object
    obj.scale = (scale, scale, scale)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for i in [0, 2, 4, 6]:
        bm.verts[i].select_set(True)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.merge(type='CENTER')

    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.modifier_add(type='WIREFRAME')
    obj.modifiers["Wireframe"].thickness = 0.1
    add_material_to_blender_primitive(obj, pid)
    center = - R.T @ T
    obj.location = center.T[0]
    obj.rotation_euler = Matrix(R.T).to_euler()
    if hasattr(obj, 'visible_shadow'):
        obj.visible_shadow = False
    return obj


def create_camera_blender_animated(camera_RT, scale=0.1, pid=0, start_frame=0, convert_axis=True):
    """Create an animated camera visualization from world_to_camera matrices.

    Args:
        camera_RT: numpy array of shape (num_frames, 3, 4) or (num_frames, 4, 4)
                   containing [R|T] matrices in world_to_camera format
        scale: scale of the camera visualization object
        pid: material/color index for the camera object
        start_frame: starting frame number for the animation
        convert_axis: if True, convert from Y-up to Z-up coordinate system
                      (needed when FBX import uses Blender's default Z-up but camera data is Y-up)

    Returns:
        obj: the created camera visualization object
    """
    # Coordinate system conversion matrix from Y-up to Z-up
    # This rotates -90 degrees around X axis: new_Y = -old_Z, new_Z = old_Y
    if convert_axis:
        # Y-up to Z-up transformation matrix
        axis_convert = np.array([
            [1,  0,  0],
            [0,  0,  1],
            [0, -1,  0]
        ], dtype=np.float64)
    else:
        axis_convert = np.eye(3)

    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD')
    obj = bpy.context.object
    obj.scale = (scale, scale, scale)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='DESELECT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    for i in [0, 2, 4, 6]:
        bm.verts[i].select_set(True)
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.merge(type='CENTER')

    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.modifier_add(type='WIREFRAME')
    obj.modifiers["Wireframe"].thickness = 0.1
    add_material_to_blender_primitive(obj, pid)
    if hasattr(obj, 'visible_shadow'):
        obj.visible_shadow = False

    # Animate the camera for each frame
    num_frames = camera_RT.shape[0]
    for frame_idx in range(num_frames):
        R = camera_RT[frame_idx, :3, :3]
        T = camera_RT[frame_idx, :3, 3:4]

        # Convert world_to_camera to camera position and orientation
        # camera_position = -R^T @ T (in original coordinate system)
        center_original = - R.T @ T

        # Apply coordinate system conversion
        # Transform the camera center position
        center = axis_convert @ center_original

        # Transform the rotation matrix: R_new = axis_convert @ R @ axis_convert.T
        R_converted = axis_convert @ R @ axis_convert.T

        obj.location = (center[0, 0], center[1, 0], center[2, 0])
        obj.rotation_euler = Matrix(R_converted.T).to_euler()

        # Insert keyframes
        frame = start_frame + frame_idx
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    print(f"Created animated camera visualization with {num_frames} frames (axis_convert={convert_axis})")
    return obj

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


def load_smpl_npz(filename, default_rotation=(0., 0., 0.), speedup=1.0):
    """Load SMPL NPZ file and return the mesh object, key, and material name.

    Args:
        filename: Path to NPZ file
        default_rotation: Default rotation for the object
        speedup: Speedup factor for animation (e.g., 2.0 means 2x faster)
    """
    keys_old = set(bpy.data.objects.keys())
    mat_old = set(bpy.data.materials.keys())
    image_old = set(bpy.data.images.keys())
    if filename.endswith('.npz'):
        bpy.ops.object.smplx_add_animation(filepath=filename)
        # bpy.ops.import_scene.obj(filepath=filename, axis_forward='X', axis_up='Z')
    keys_new = set(bpy.data.objects.keys())
    mat_new = set(bpy.data.materials.keys())
    image_new = set(bpy.data.images.keys())
    key = list(keys_new - keys_old)[0]
    current_obj = bpy.data.objects[key]
    # set default rotation to 0.
    current_obj.rotation_euler = default_rotation
    key_image = list(image_new-image_old)
    if len(key_image) > 0:
        print('>>> Loading image {}'.format(key_image[0]))
        key = (key, key_image[0])
    mat = list(mat_new - mat_old)[0]
    # Select the mesh object instead of the armature
    smplx_obj = current_obj

    if "Body" in bpy.data.objects:
        smplx_obj = bpy.data.objects["Body"]
    else:
        # Find the mesh object that's a child of the armature
        for obj in bpy.data.objects:
            if obj.parent == smplx_obj and obj.type == 'MESH':
                smplx_obj = obj
                break

    # Check if there's animation and downsample if needed
    scene = bpy.context.scene
    if speedup > 1.0 and scene.frame_end > scene.frame_start:
        # Check if any object has animation
        has_animation = False
        for obj in bpy.data.objects:
            if obj.animation_data and obj.animation_data.action:
                has_animation = True
                break

        if has_animation:
            downsample_animation(speedup)

    return smplx_obj, key, mat

def downsample_animation(speedup=2.0):
    """Downsample animation keyframes to speed up playback.

    Args:
        speedup: Speedup factor (e.g., 2.0 means 2x faster, keeping every Nth frame)
    """
    if speedup <= 1.0:
        return

    speedup = int(speedup)
    scene = bpy.context.scene
    original_frame_start = scene.frame_start
    original_frame_end = scene.frame_end

    # Find all objects with animation
    animated_objects = []
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            animated_objects.append(obj)

    if not animated_objects:
        print("No animated objects found for downsampling")
        return

    # Calculate new frame range
    new_frame_start = original_frame_start
    new_frame_end = original_frame_start + (original_frame_end - original_frame_start) // speedup

    # Downsample keyframes for each animated object
    for obj in animated_objects:
        action = obj.animation_data.action
        if not action:
            continue

        # Process each fcurve (animation channel)
        for fcurve in action.fcurves:
            keyframe_points = fcurve.keyframe_points

            if len(keyframe_points) == 0:
                continue

            # Collect indices of keyframes to keep (every Nth frame)
            indices_to_keep = list(range(0, len(keyframe_points), speedup))
            # Always keep the last frame
            if len(keyframe_points) - 1 not in indices_to_keep:
                indices_to_keep.append(len(keyframe_points) - 1)
            indices_to_keep = sorted(set(indices_to_keep))

            # Remap frames for keyframes we're keeping
            for idx in indices_to_keep:
                kp = keyframe_points[idx]
                original_frame = kp.co[0]
                new_frame = new_frame_start + (original_frame - original_frame_start) // speedup
                # Update frame number
                kp.co[0] = new_frame
                kp.handle_left[0] = new_frame
                kp.handle_right[0] = new_frame

            # Remove keyframes that are not in indices_to_keep (in reverse order)
            for idx in reversed(range(len(keyframe_points))):
                if idx not in indices_to_keep:
                    keyframe_points.remove(keyframe_points[idx])

    # Update scene frame range
    scene.frame_start = new_frame_start
    scene.frame_end = new_frame_end

    print(f"Animation downsampled by {speedup}x: frames {original_frame_start}-{original_frame_end} -> {new_frame_start}-{new_frame_end}")

def load_fbx(filename, default_rotation=(0., 0., 0.), speedup=1.0):
    """Load FBX file and return the mesh object, key, and material name.

    Args:
        filename: Path to FBX file
        default_rotation: Default rotation for the object
        speedup: Speedup factor for animation (e.g., 2.0 means 2x faster)
    """
    keys_old = set(bpy.data.objects.keys())
    mat_old = set(bpy.data.materials.keys())
    image_old = set(bpy.data.images.keys())

    if filename.endswith('.fbx') or filename.endswith('.FBX'):
        bpy.ops.import_scene.fbx(filepath=filename)

    keys_new = set(bpy.data.objects.keys())
    mat_new = set(bpy.data.materials.keys())
    image_new = set(bpy.data.images.keys())

    # Get imported object names
    obj_names = [o.name for o in bpy.context.selected_objects]
    if not obj_names:
        # If no objects are selected, get all newly imported objects
        obj_names = list(keys_new - keys_old)

    # Find armature and mesh objects
    armature = None
    mesh_object = None
    mesh_object_list = []

    for obj_name in obj_names:
        obj = bpy.data.objects[obj_name]
        if obj.type == 'ARMATURE' or (obj.animation_data and obj.animation_data.action):
            armature = obj
        if obj.type == 'MESH' and mesh_object is None:
            mesh_object = obj
        if obj.type == 'MESH':
            mesh_object_list.append(obj)

    # Set scene frame range if armature has animation
    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end
        print(f"Animation frames set: {frame_start} to {frame_end}")

        # Downsample animation if speedup > 1.0
        if speedup > 1.0:
            downsample_animation(speedup)

    # Use the first mesh object, or armature if no mesh found
    if mesh_object:
        current_obj = mesh_object
    elif armature:
        current_obj = armature
    else:
        # Fallback to first imported object
        current_obj = bpy.data.objects[list(keys_new - keys_old)[0]]

    current_obj.rotation_euler = default_rotation

    key = current_obj.name
    key_image = list(image_new - image_old)
    if len(key_image) > 0:
        print('>>> Loading image {}'.format(key_image[0]))
        key = (key, key_image[0])

    mat = list(mat_new - mat_old)[0] if (mat_new - mat_old) else None

    return current_obj, key, mat

def export_smpl_npz_to_fbx(filename):
    # Select all objects in the scene
    bpy.ops.object.select_all(action='DESELECT')

    # Find and select only armature objects
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            obj.select_set(True)
            # Also make this the active object
            bpy.context.view_layer.objects.active = obj
            print(f"Selected armature object: {obj.name}")
        elif obj.type == 'MESH' and 'SMPL' in obj.name:
            obj.select_set(True)
            print(f"Selected mesh object: {obj.name}")
    # Create output filename by replacing .npz with .fbx
    output_filename = filename.replace('.npz', '.fbx')
    output_filepath = output_filename

    # Select only the SMPLX object for export

    # bpy.ops.object.select_all(action='DESELECT')
    # smplx_obj.select_set(True)
    # print(smplx_obj.type)
    # breakpoint()
    # Export only the animation data of the SMPLX object as FBX
    bpy.ops.export_scene.fbx(
        filepath=output_filepath,
        check_existing=False,
        use_selection=True,
        bake_anim=True,
        path_mode='RELATIVE'
    )

def addGround(location=(0, 0, 0), groundSize=100, shadowBrightness=0.7, normal_axis="Z", tex_fn=None, alpha=0.5, shadow_catcher=False):
    # initialize a ground for shadow
    bpy.context.scene.cycles.film_transparent = True
    if normal_axis.upper() == "Z":
        bpy.ops.mesh.primitive_plane_add(location=location, size=groundSize, rotation=(0, 0, 0))
    elif normal_axis.upper() == "Y":
        bpy.ops.mesh.primitive_plane_add(location=location, size=groundSize, rotation=(math.radians(90), 0, 0))
    elif normal_axis.upper() == "X":
        bpy.ops.mesh.primitive_plane_add(location=location, size=groundSize, rotation=(0, math.radians(90), 0))
    else:
        raise ValueError

    # pdb.set_trace()
    # NOTE not working, weird
    # if normal_axis.lower() == "Y":
    #     ground.rotation_euler[0] = math.radians(90)
    # elif normal_axis.lower() == 'X':
    #     ground.rotation_euler[1] = math.radians(90)
    ground = bpy.context.object
    ground.name = "ground"

    try:
        ground.is_shadow_catcher = shadow_catcher  # for blender 3.X
    except:
        ground.cycles.is_shadow_catcher = shadow_catcher  # for blender 2.X

    # set material
    checker_mat = bpy.data.materials.new("CheckerboardMaterial")
    checker_mat.use_nodes = True
    checker_mat.blend_method = "BLEND"  # 启用透明混合

    # 设置阴影方法 - 兼容不同版本的Blender
    try:
        checker_mat.shadow_method = "HASHED"  # 优化阴影表现 (Blender 3.x+)
    except AttributeError:
        # 在较新版本的Blender中，shadow_method可能不存在或已更改
        pass
    ground.data.materials.append(checker_mat)
    ground.active_material = checker_mat
    tree = checker_mat.node_tree

    # 移除默认节点
    for node in list(tree.nodes):
        if node.name != "Material Output":
            tree.nodes.remove(node)

    # 添加纹理坐标节点
    texcoord_node = tree.nodes.new("ShaderNodeTexCoord")

    # 添加Mapping节点用于平铺纹理
    mapping_node = tree.nodes.new("ShaderNodeMapping")
    mapping_node.inputs["Scale"].default_value = (groundSize, groundSize, groundSize)
    tree.links.new(texcoord_node.outputs["UV"], mapping_node.inputs["Vector"])

    if tex_fn is None:
        # 使用棋盘格纹理
        checker_node = tree.nodes.new("ShaderNodeTexChecker")
        checker_node.inputs["Scale"].default_value = 1.0  # 1 square per meter
        checker_node.inputs["Color1"].default_value = (1, 1, 1, 1)  # 纯白
        checker_node.inputs["Color2"].default_value = (0, 0, 0, 1)  # 纯黑
        tree.links.new(mapping_node.outputs["Vector"], checker_node.inputs["Vector"])
        texture_output = checker_node.outputs["Color"]
    else:
        # 加载外部图片纹理
        assert os.path.isfile(tex_fn), f"Texture file not found: {tex_fn}"
        image_node = tree.nodes.new("ShaderNodeTexImage")
        image_node.image = bpy.data.images.load(tex_fn)
        image_node.image.alpha_mode = "STRAIGHT"  # 保留alpha通道
        image_node.extension = "REPEAT"  # 设置平铺模式
        tree.links.new(mapping_node.outputs["Vector"], image_node.inputs["Vector"])
        texture_output = image_node.outputs["Color"]

    if shadow_catcher:
        # 当作为阴影捕捉器时，使用简单的透明材质
        # Blender的shadow catcher功能会自动处理阴影显示
        transparent_node = tree.nodes.new("ShaderNodeBsdfTransparent")
        transparent_node.inputs["Color"].default_value = (1, 1, 1, 1)  # 白色透明
        tree.links.new(transparent_node.outputs["BSDF"], tree.nodes["Material Output"].inputs["Surface"])
    else:
        # 添加Principled BSDF节点
        bsdf_node = tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf_node.inputs["Roughness"].default_value = 1
        bsdf_node.inputs["Alpha"].default_value = alpha  # 设置透明度

        # 连接节点
        tree.links.new(texture_output, bsdf_node.inputs["Base Color"])
        tree.links.new(bsdf_node.outputs["BSDF"], tree.nodes["Material Output"].inputs["Surface"])

    return ground


def add_root_trajectory(positions, start_color=(0.2, 0.6, 1.0, 1.0), end_color=(1.0, 0.3, 0.5, 1.0),
                        line_thickness=0.02, emission_strength=3.0, pelvis_height=1.0,
                        alpha=1.0, curve_name="RootTrajectory",
                        extend_start=0.6, extend_end=0.6, add_arrow=True, arrow_scale=1.0):
    """
    Add a colorful gradient trajectory line connecting the root/pelvis positions of multiple characters.
    This emphasizes "Global Motion" and "Dynamics" by showing the movement path in world space.

    Args:
        armature_list: List of (armature, mesh_object_list) tuples from load_fbx_at_frame
        start_color: RGBA color at the beginning of trajectory (default: cyan blue)
        end_color: RGBA color at the end of trajectory (default: magenta pink)
        line_thickness: Thickness of the trajectory line
        emission_strength: Emission strength for the glowing effect
        pelvis_height: Height of the pelvis/root joint from ground (default: 1.0m)
        alpha: Transparency value (0=fully transparent, 1=opaque)
        curve_name: Name for the curve object
        extend_start: How much to extend the trajectory before the first point (in meters)
        extend_end: How much to extend the trajectory after the last point (in meters)
        add_arrow: Whether to add an arrowhead at the end
        arrow_scale: Scale factor for the arrowhead size

    Returns:
        The created curve object
    """
    from mathutils import Vector

    # IMPORTANT: Update scene and dependency graph to ensure all armature poses are evaluated
    # This is critical for getting correct bone positions, especially for the last imported armature
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Extend trajectory at the start (before first point)
    if extend_start > 0 and len(positions) >= 2:
        start_dir = (positions[0] - positions[1]).normalized()
        extended_start = positions[0] + start_dir * extend_start
        positions.insert(0, extended_start)
        print(f"Extended trajectory start by {extend_start}m")

    # Extend trajectory at the end (after last point)
    if extend_end > 0 and len(positions) >= 2:
        end_dir = (positions[-1] - positions[-2]).normalized()
        extended_end = positions[-1] + end_dir * extend_end
        positions.append(extended_end)
        print(f"Extended trajectory end by {extend_end}m")

    # Detect trajectory direction and adjust color mapping if reversed
    # This ensures color gradient matches visual direction regardless of default_trans sign
    if len(positions) >= 2:
        x_start = positions[0].x
        x_end = positions[-1].x
        is_reversed = x_end < x_start  # If end x < start x, trajectory goes right to left
        
        if is_reversed:
            # Swap color mapping so visual start uses start_color and visual end uses end_color
            start_color, end_color = end_color, start_color
            print(f"Trajectory is reversed (right to left), swapping colors")
    
    # Arrow color should match the curve end color (after potential swap)
    arrow_color = start_color

    # Create a curve for the trajectory
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    curve_data.bevel_depth = line_thickness
    curve_data.bevel_resolution = 4
    curve_data.fill_mode = 'FULL'

    # Create a smooth spline through all positions
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(positions) - 1)

    for i, pos in enumerate(positions):
        bp = spline.bezier_points[i]
        bp.co = pos
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    # Create the curve object
    curve_obj = bpy.data.objects.new(curve_name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Create gradient emission material using curve parameter
    mat_name = f"{curve_name}_Material"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    # Enable transparency if alpha < 1
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'HASHED'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Mix Shader to combine emission with principled for nice glow effect
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (400, 0)
    mix_shader.inputs['Fac'].default_value = 0.7  # More emission than diffuse

    # Principled BSDF for base color
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (200, -150)
    principled.inputs['Roughness'].default_value = 0.3
    principled.inputs['Metallic'].default_value = 0.2
    principled.inputs['Alpha'].default_value = alpha  # Set transparency

    # Emission shader for glow effect
    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (200, 100)
    emission.inputs['Strength'].default_value = emission_strength

    # Color Ramp for gradient
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-100, 0)
    # Set gradient colors (start to end)
    color_ramp.color_ramp.elements[0].color = start_color
    color_ramp.color_ramp.elements[1].color = end_color
    # Add middle color for more vibrant transition
    mid_elem = color_ramp.color_ramp.elements.new(0.5)
    mid_color = (
        (start_color[0] + end_color[0]) * 0.5 + 0.2,  # Slightly boosted
        (start_color[1] + end_color[1]) * 0.3,
        (start_color[2] + end_color[2]) * 0.6,
        1.0
    )
    mid_elem.color = mid_color

    # Spline Parameter node (gives 0-1 along the curve)
    # For curves, we can use the Generated texture coordinate's X component
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-500, 0)

    # Separate XYZ to get the spline parameter
    separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    separate_xyz.location = (-300, 0)

    # Connect nodes
    # Use Generated coordinates X for curve parameter (0-1 along spline)
    links.new(tex_coord.outputs['Generated'], separate_xyz.inputs['Vector'])
    links.new(separate_xyz.outputs['X'], color_ramp.inputs['Fac'])

    # Connect color to both emission and principled
    links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])

    # Connect shaders to mix
    links.new(principled.outputs['BSDF'], mix_shader.inputs[1])
    links.new(emission.outputs['Emission'], mix_shader.inputs[2])

    # Connect to output
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    # Apply material to curve
    curve_obj.data.materials.append(mat)

    # Add arrowhead at the end of trajectory
    arrow_obj = None
    if add_arrow and len(positions) >= 2:
        # Calculate arrow direction (pointing in the direction of motion)
        arrow_dir = (positions[-1] - positions[-2]).normalized()
        arrow_pos = positions[-1]

        # Create a cone for the arrowhead
        arrow_height = line_thickness * 8 * arrow_scale
        arrow_radius = line_thickness * 2 * arrow_scale

        bpy.ops.mesh.primitive_cone_add(
            vertices=16,
            radius1=arrow_radius,
            radius2=0,
            depth=arrow_height,
            location=arrow_pos
        )
        arrow_obj = bpy.context.active_object
        arrow_obj.name = f"{curve_name}_Arrow"

        # Rotate arrow to point in the direction of motion
        # Default cone points in +Z, we need to align it with arrow_dir
        from mathutils import Matrix
        import math

        # Calculate rotation to align Z-axis with arrow_dir
        z_axis = Vector((0, 0, 1))
        rotation_axis = z_axis.cross(arrow_dir)
        if rotation_axis.length > 0.0001:
            rotation_axis.normalize()
            angle = math.acos(max(-1, min(1, z_axis.dot(arrow_dir))))
            rot_matrix = Matrix.Rotation(angle, 4, rotation_axis)
            arrow_obj.matrix_world = Matrix.Translation(arrow_pos) @ rot_matrix

            # Offset arrow so its base is at the curve end
            arrow_obj.location = arrow_pos + arrow_dir * (arrow_height * 0.5)
        else:
            # arrow_dir is parallel to Z axis
            if arrow_dir.z < 0:
                # Pointing down, flip
                arrow_obj.rotation_euler = (math.pi, 0, 0)
            arrow_obj.location = arrow_pos + arrow_dir * (arrow_height * 0.5)

        # Create solid color material for arrow (using end_color, no gradient)
        arrow_mat_name = f"{curve_name}_Arrow_Material"
        arrow_mat = bpy.data.materials.new(name=arrow_mat_name)
        arrow_mat.use_nodes = True

        # Enable transparency if alpha < 1
        if alpha < 1.0:
            arrow_mat.blend_method = 'BLEND'
            arrow_mat.shadow_method = 'HASHED'

        arrow_nodes = arrow_mat.node_tree.nodes
        arrow_links = arrow_mat.node_tree.links

        # Clear default nodes
        arrow_nodes.clear()

        # Output node
        arrow_output = arrow_nodes.new(type='ShaderNodeOutputMaterial')
        arrow_output.location = (600, 0)

        # Mix Shader to combine emission with principled for nice glow effect
        arrow_mix_shader = arrow_nodes.new(type='ShaderNodeMixShader')
        arrow_mix_shader.location = (400, 0)
        arrow_mix_shader.inputs['Fac'].default_value = 0.7  # More emission than diffuse

        # Principled BSDF for base color
        arrow_principled = arrow_nodes.new(type='ShaderNodeBsdfPrincipled')
        arrow_principled.location = (200, -150)
        arrow_principled.inputs['Roughness'].default_value = 0.3
        arrow_principled.inputs['Metallic'].default_value = 0.2
        arrow_principled.inputs['Alpha'].default_value = alpha  # Set transparency
        arrow_principled.inputs['Base Color'].default_value = arrow_color  # Use arrow_color to match curve end

        # Emission shader for glow effect
        arrow_emission = arrow_nodes.new(type='ShaderNodeEmission')
        arrow_emission.location = (200, 100)
        arrow_emission.inputs['Strength'].default_value = emission_strength
        arrow_emission.inputs['Color'].default_value = arrow_color  # Use arrow_color to match curve end

        # Connect shaders to mix
        arrow_links.new(arrow_principled.outputs['BSDF'], arrow_mix_shader.inputs[1])
        arrow_links.new(arrow_emission.outputs['Emission'], arrow_mix_shader.inputs[2])

        # Connect to output
        arrow_links.new(arrow_mix_shader.outputs['Shader'], arrow_output.inputs['Surface'])

        # Apply solid color material to arrow
        arrow_obj.data.materials.append(arrow_mat)
        print(f"Added arrowhead at end of trajectory '{curve_name}'")

    print(f"Root trajectory '{curve_name}' created with {len(positions)} points, alpha={alpha}, gradient from {start_color[:3]} to {end_color[:3]}")
    return curve_obj