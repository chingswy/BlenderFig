import bpy
import time
from myblender.geometry import build_plane
from myblender.material import set_material_i, setup_mist_fog
from myblender.setup import setup, set_output_properties, parse_args
from myblender.setup import set_cycles_renderer, add_sunlight, add_area_light
from myblender.geometry import set_camera
from myblender.fbxtools import load_fbx_at_frame
import os


def add_root_trajectory(armature_list, start_color=(0.2, 0.6, 1.0, 1.0), end_color=(1.0, 0.3, 0.5, 1.0),
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

    if len(armature_list) < 2:
        print("Warning: Need at least 2 positions to create trajectory")
        return None

    # IMPORTANT: Update scene and dependency graph to ensure all armature poses are evaluated
    # This is critical for getting correct bone positions, especially for the last imported armature
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Collect root positions from each armature at the current frame
    positions = []
    for idx, (armature, mesh_object_list) in enumerate(armature_list):
        root_pos = None
        found_bone = False

        # Get the evaluated armature (with all modifiers and poses applied)
        armature_eval = armature.evaluated_get(depsgraph)

        # Try to find pelvis/hips bone for accurate position
        if armature_eval.pose and armature_eval.pose.bones:
            # Common pelvis bone names in SMPL/FBX
            pelvis_names = ['pelvis', 'Pelvis', 'mixamorig:Hips', 'Hips', 'hips',
                           'm_avg_Pelvis', 'f_avg_Pelvis', 'root', 'Root']

            # Debug: print available bone names for first armature
            if idx == 0:
                print(f"Available bones: {[b.name for b in armature_eval.pose.bones]}")

            for bone_name in pelvis_names:
                if bone_name in armature_eval.pose.bones:
                    bone = armature_eval.pose.bones[bone_name]
                    # Get bone world position using bone.matrix (includes pose)
                    # bone.matrix is in armature space, need to transform to world
                    bone_world_matrix = armature_eval.matrix_world @ bone.matrix
                    root_pos = bone_world_matrix.translation.copy()
                    found_bone = True
                    print(f"Frame {idx}: Found bone '{bone_name}' at world position {root_pos}")
                    break

        # Fallback: use armature location + pelvis height
        if not found_bone:
            armature_loc = armature.location.copy()
            root_pos = Vector((armature_loc.x, armature_loc.y, armature_loc.z + pelvis_height))
            print(f"Frame {idx}: Using armature location + pelvis_height: {root_pos}")

        positions.append(root_pos)

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

if __name__ == "__main__":
    # ${blender} --python examples/fig_v2m/render_ground_seq_v2.py -- --debug
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ground", default=(200/255, 200/255, 200/255, 1.0), nargs=4, type=float,
    help="Ground color: (94/255, 124/255, 226/255, 1.0) 青色")
    parser.add_argument("--body", default=[0.05, 0.326, 1.], nargs=3, type=float)
    args = parse_args(parser)

    setup(rgb=(1,1,1,0))


    # 读入fbx
    config_name = "keling2"
    fbxname = "assets/2026V2M/00000001_keling2_pred_seed42.fbx"
    default_trans = 0.9

    config_name = "keling1"
    default_trans = -1.2

    fbxname = [
        "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed44.fbx",
        "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed42.fbx",
        "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed51.fbx",
        "assets/2026V2M/evaluation_original_wv_demo_koala_keyframe_seed16/epoch40/00000001_keling1_pred_seed55.fbx",
    ]

    # 检查所有FBX文件是否存在
    for fbx in fbxname:
        assert os.path.exists(fbx), fbx

    config = {
        "keling1": {
            "frames": [
                {"frame": 0, "x_offset": 0.3},
                {"frame": 22, "x_offset": 0.2}, # 多样的关键帧
                {"frame": 42, "x_offset": 0.},
                {"frame": 62, "x_offset": 0.}, # 多样的关键帧
                {"frame": 82, "x_offset": 0.},
                # {"frame": 102, "x_offset": -0.1},
                {"frame": 152, "x_offset": 0},
            ]
        },
        "keling2": {
            # Main character frames
            "frames": [
                {"frame": 60, "x_offset": 0},
                {"frame": 80, "x_offset": 0},
                # {"frame": 85, "x_offset": 0},
                {"frame": 90, "x_offset": -0.15},
                {"frame": 95, "x_offset": 0.07},
                {"frame": 98, "x_offset": 0},
                # {"frame": 105, "x_offset": 0},
                {"frame": 110, "x_offset": 0},
                # {"frame": 120, "x_offset": 0},
                {"frame": 130, "x_offset": 0},
                # {"frame": 150, "x_offset": 0},
                # {"frame": 180, "x_offset": 0},
            ]
        }
    }[config_name]


    num_frames = len(config["frames"])

    # Collect armature list for trajectory visualization
    armature_list = []

    def get_pelvis_world_xy(armature):
        """获取 Pelvis 骨骼在世界坐标中的 XY 位置"""
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        armature_eval = armature.evaluated_get(depsgraph)
        
        pelvis_names = ['pelvis', 'Pelvis', 'mixamorig:Hips', 'Hips', 'hips',
                       'm_avg_Pelvis', 'f_avg_Pelvis', 'root', 'Root']
        
        if armature_eval.pose and armature_eval.pose.bones:
            for bone_name in pelvis_names:
                if bone_name in armature_eval.pose.bones:
                    bone = armature_eval.pose.bones[bone_name]
                    bone_world_matrix = armature_eval.matrix_world @ bone.matrix
                    world_pos = bone_world_matrix.translation
                    return world_pos.x, world_pos.y
        
        # Fallback: 使用 armature 位置
        return armature.location.x, armature.location.y

    # Step 1: Load main character frames
    # 对于每个帧，加载所有FBX文件，第一个不透明，其余透明
    ghost_alpha = 0.5  # 非主角FBX的透明度
    
    for ii, frame_config in enumerate(config["frames"]):
        color_progress = ii / max(num_frames - 1, 1)
        x_offset = frame_config["x_offset"] + default_trans * ii
        print(f"Loading main frame {frame_config['frame']} at x={x_offset:.2f}, color_progress={color_progress:.2f}")
        
        # 记录第一个FBX的Pelvis世界坐标，用于对齐其他FBX
        reference_pelvis_xy = None
        
        # 遍历所有FBX文件
        for fbx_idx, fbx_path in enumerate(fbxname):
            armature, mesh_object_list = load_fbx_at_frame(
                fbx_path,
                frame_config["frame"],
                x_offset,
                target_frame=1
            )
            
            # 基于Pelvis骨骼位置对齐
            current_pelvis_x, current_pelvis_y = get_pelvis_world_xy(armature)
            
            if fbx_idx == 0:
                # 记录第一个FBX的Pelvis世界坐标作为参考
                reference_pelvis_xy = (current_pelvis_x, current_pelvis_y)
                print(f"  Reference Pelvis XY: ({current_pelvis_x:.3f}, {current_pelvis_y:.3f})")
            else:
                # 计算偏移量，使当前FBX的Pelvis对齐到参考位置
                offset_x = reference_pelvis_xy[0] - current_pelvis_x
                offset_y = reference_pelvis_xy[1] - current_pelvis_y
                armature.location.x += offset_x
                armature.location.y += offset_y
                print(f"  FBX {fbx_idx} Pelvis offset: ({offset_x:.3f}, {offset_y:.3f})")
            
            # 第一个FBX不透明，其他透明
            alpha = 1.0 if fbx_idx == 0 else ghost_alpha
            for mesh_obj in mesh_object_list:
                set_material_i(mesh_obj, tuple(args.body), use_plastic=False, alpha=alpha)
                # 为透明材质启用blend模式
                if alpha < 1.0:
                    mesh_obj.active_material.blend_method = 'BLEND'
                    mesh_obj.active_material.shadow_method = 'HASHED'
            
            # 只将第一个FBX加入轨迹列表
            if fbx_idx == 0:
                armature_list.append((armature, mesh_object_list))

    # Add root trajectory line for main character
    # Extended at both ends to "pierce through" the body, with arrow at the end
    trajectory_curve = add_root_trajectory(
        armature_list,
        start_color=(1.0, 0.3, 0.3, 1.0),   # Stronger red (start of motion, less pale)
        end_color=(0.6, 0.0, 0.0, 1.0),     # Deep red (end of motion)
        line_thickness=0.025,
        emission_strength=2.5,
        curve_name="MainTrajectory",
        extend_start=0.8,    # Extend before first character
        extend_end=0.8,      # Extend after last character
        add_arrow=True,      # Add arrowhead at the end
        arrow_scale=1.2      # Slightly larger arrow
    )

    center_x = default_trans * (num_frames - 1) / 2

    # 添加带有镜面反射效果的棋盘格地面
    # roughness: 0.0 = 完美镜子, 0.1 = 轻微模糊的反射, 0.5 = 默认
    # metallic: 0.0 = 非金属反射, 1.0 = 金属反射
    # specular: 反射强度
    build_plane(translation=(center_x, 0, 0), plane_size=100,
                white=(1,1,1,1), black=args.ground,
                roughness=0.1, metallic=0.8, specular=0.8)

    setup_mist_fog(
        bpy.context.scene,
        start=12,
        depth=20,
        fog_color=(1, 1, 1) # 稍微蓝一点，证明雾存在
    )
    add_sunlight(
        location=(center_x, 0, 5),
        lookat=(center_x, 0, 0),
        strength=10,
        cast_shadow=False,
    )

    add_area_light(
        location=(center_x, 5, 2),
        lookat=(center_x, 0, 1),
        strength=30,
    )

    focal, distance = 40, 10
    focal, distance = 35, 10

    set_camera(
        location=(center_x, distance, 4),
        center=(center_x, 0, 1),
        focal=focal  # Portrait lens focal length
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f'output/v2m_teaser_seq_{timestamp}.jpg'


    if args.debug:
        res_x, res_y = 1024, 1024
        num_samples = 16
    else:
        res_x, res_y = 4096, 4096
        num_samples = 512

    set_output_properties(
        bpy.context.scene,
        output_file_path=output_path,
        format='JPEG',
        res_x=res_x,
        res_y=res_y,
        tile_x=res_x,
        tile_y=res_y,
        resolution_percentage=100,
    )

    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=num_samples,
        use_transparent_bg=False,
        use_denoising=True,
    )

    bpy.ops.render.render(write_still=True, animation=False)