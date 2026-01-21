import bpy
import math

def find_armature_and_mesh(obj_names):
    """Find armature and mesh objects from a list of object names."""
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
    return armature, mesh_object, mesh_object_list


def shift_action_frames(action, offset):
    """
    Shift all keyframes in an action by offset frames.

    Args:
        action: The Blender action to modify
        offset: Number of frames to shift (positive = forward, negative = backward)
    """
    for fcurve in action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.co.x += offset
            keyframe.handle_left.x += offset
            keyframe.handle_right.x += offset


def zero_xy_translation_at_frame(action, target_frame):
    """
    Zero out x and y translation at target_frame by subtracting their values from all keyframes.

    Args:
        action: The Blender action to modify
        target_frame: The frame at which x and y should become 0
    """
    # Find location.x and location.y fcurves
    for fcurve in action.fcurves:
        # Check if this is a location channel (x=0, y=1)
        if fcurve.data_path == 'location' and fcurve.array_index in [0, 1]:
            # Get the value at target_frame
            value_at_frame = fcurve.evaluate(target_frame)
            # Subtract this value from all keyframes to make target_frame's value = 0
            for keyframe in fcurve.keyframe_points:
                keyframe.co.y -= value_at_frame
                keyframe.handle_left.y -= value_at_frame
                keyframe.handle_right.y -= value_at_frame


def zero_pelvis_xy_translation_at_frame(action, target_frame):
    """
    Zero out the Pelvis bone's x and y translation at target_frame.

    Args:
        action: The Blender action to modify
        target_frame: The frame at which x and y should become 0
    """
    # Common names for root/pelvis bone
    pelvis_names = ['Pelvis', 'pelvis', 'Hips', 'hips', 'Root', 'root', 'mixamorig:Hips']

    for fcurve in action.fcurves:
        # Check if this is a pose bone location channel
        for pelvis_name in pelvis_names:
            data_path = f'pose.bones["{pelvis_name}"].location'
            if fcurve.data_path == data_path and fcurve.array_index in [0, 1]:
                # Get the value at target_frame
                value_at_frame = fcurve.evaluate(target_frame)
                # Subtract this value from all keyframes to make target_frame's value = 0
                for keyframe in fcurve.keyframe_points:
                    keyframe.co.y -= value_at_frame
                    keyframe.handle_left.y -= value_at_frame
                    keyframe.handle_right.y -= value_at_frame
                break


def rotate_animation_trajectory(action, angle_degrees):
    """
    Rotate XY translation keyframes around Z-axis.
    
    This transforms the animation trajectory so that movement in the original
    direction becomes movement in a rotated direction.
    
    Args:
        action: The Blender action to modify
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
    """
    if angle_degrees == 0:
        return
    
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    # Common names for root/pelvis bone
    pelvis_names = ['Pelvis', 'pelvis', 'Hips', 'hips', 'Root', 'root', 'mixamorig:Hips']
    
    # Build list of data_paths to check for XY location
    data_paths_to_rotate = ['location']  # Armature-level location
    for pelvis_name in pelvis_names:
        data_paths_to_rotate.append(f'pose.bones["{pelvis_name}"].location')
    
    # For each data_path, find the x (index 0) and y (index 1) fcurves
    for data_path in data_paths_to_rotate:
        fcurve_x = None
        fcurve_y = None
        
        for fcurve in action.fcurves:
            if fcurve.data_path == data_path:
                if fcurve.array_index == 0:
                    fcurve_x = fcurve
                elif fcurve.array_index == 1:
                    fcurve_y = fcurve
        
        # Only rotate if both x and y fcurves exist
        if fcurve_x is not None and fcurve_y is not None:
            # Get all keyframe data
            # We need to collect all values first, then apply rotation
            x_keyframes = [(kf.co.x, kf.co.y, kf.handle_left.y, kf.handle_right.y) 
                           for kf in fcurve_x.keyframe_points]
            y_keyframes = [(kf.co.x, kf.co.y, kf.handle_left.y, kf.handle_right.y) 
                           for kf in fcurve_y.keyframe_points]
            
            # Assuming x and y have the same keyframe times
            # Apply rotation: x' = x*cos - y*sin, y' = x*sin + y*cos
            for i, (kf_x, kf_y) in enumerate(zip(fcurve_x.keyframe_points, fcurve_y.keyframe_points)):
                x_val = x_keyframes[i][1]  # co.y is the value
                y_val = y_keyframes[i][1]
                
                x_handle_left = x_keyframes[i][2]
                y_handle_left = y_keyframes[i][2]
                
                x_handle_right = x_keyframes[i][3]
                y_handle_right = y_keyframes[i][3]
                
                # Rotate main value
                new_x = x_val * cos_a - y_val * sin_a
                new_y = x_val * sin_a + y_val * cos_a
                
                # Rotate handles
                new_x_handle_left = x_handle_left * cos_a - y_handle_left * sin_a
                new_y_handle_left = x_handle_left * sin_a + y_handle_left * cos_a
                
                new_x_handle_right = x_handle_right * cos_a - y_handle_right * sin_a
                new_y_handle_right = x_handle_right * sin_a + y_handle_right * cos_a
                
                # Apply new values
                kf_x.co.y = new_x
                kf_y.co.y = new_y
                
                kf_x.handle_left.y = new_x_handle_left
                kf_y.handle_left.y = new_y_handle_left
                
                kf_x.handle_right.y = new_x_handle_right
                kf_y.handle_right.y = new_y_handle_right


def get_mesh_lowest_z(mesh_object_list, depsgraph):
    """
    Calculate the lowest z coordinate of all mesh objects after deformation.

    Args:
        mesh_object_list: List of mesh objects
        depsgraph: Dependency graph for evaluated mesh

    Returns:
        The lowest z coordinate across all meshes
    """
    min_z = float('inf')
    for mesh_obj in mesh_object_list:
        # Get the evaluated mesh (with armature deformation applied)
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh_data = eval_obj.to_mesh()

        # Get world matrix to convert local coords to world coords
        world_matrix = eval_obj.matrix_world

        for vert in mesh_data.vertices:
            # Transform vertex to world coordinates
            world_co = world_matrix @ vert.co
            if world_co.z < min_z:
                min_z = world_co.z

        # Clean up temporary mesh data
        eval_obj.to_mesh_clear()

    return min_z


def load_fbx_at_frame(fbx_path, frame, x_offset, y_offset=0, z_offset=0,
                      target_frame=1, z_rotation=0, rotate_trajectory=False):
    """
    Load FBX file and shift animation so that the specified frame becomes target_frame.

    Args:
        fbx_path: Path to the FBX file
        frame: The original frame number to extract
        x_offset: X-axis offset for positioning
        y_offset: Y-axis offset for positioning
        target_frame: The frame number where the specified frame should appear (default: 1)
        z_rotation: Rotation around Z-axis in degrees (default: 0)
        rotate_trajectory: If True, also rotate the animation trajectory (root motion).
                          If False (default), only rotate the armature facing direction.

    Returns:
        Tuple of (armature, mesh_object_list)
    """
    # Track objects before import
    keys_old = set(bpy.data.objects.keys())

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    keys_new = set(bpy.data.objects.keys())
    obj_names = list(keys_new - keys_old)


    # Find armature and mesh objects
    armature, mesh_object, mesh_object_list = find_armature_and_mesh(obj_names)
    # Set scene frame range if armature has animation
    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end
        print(f"Animation frames set: {frame_start} to {frame_end}")

    # Shift animation so that 'frame' becomes 'target_frame'
    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        # Calculate offset: we want frame -> target_frame, so offset = target_frame - frame
        offset = target_frame - frame
        shift_action_frames(action, offset)
        # Zero out x and y translation at target_frame (armature level)
        zero_xy_translation_at_frame(action, target_frame)
        # Zero out Pelvis bone's x and y translation at target_frame
        zero_pelvis_xy_translation_at_frame(action, target_frame)

    # Apply x_offset after zeroing out original translation
    armature.location.x += x_offset
    armature.location.y += y_offset

    # Apply z rotation
    if z_rotation != 0:
        # Optionally rotate the animation trajectory (XY translation keyframes)
        if rotate_trajectory and armature.animation_data and armature.animation_data.action:
            rotate_animation_trajectory(armature.animation_data.action, z_rotation)
        # Rotate the armature's facing direction
        armature.rotation_euler[2] += math.radians(z_rotation)

    # # Set material based on whether this is a virtual/ghost character
    # for mesh_obj in mesh_object_list:
    #     if is_virtual:
    #         matname = f"GhostMaterial_inst{instance_id}_f{frame}"
    #         set_transparent_ghost_material(mesh_obj, color_progress, matname, alpha=virtual_alpha)
    #     else:
    #         matname = f"GradientBlue_inst{instance_id}_f{frame}"
    #         set_gradient_blue_material(mesh_obj, color_progress, matname)

    # Calculate lowest z at target_frame and adjust if below ground
    bpy.context.scene.frame_set(target_frame)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    min_z = get_mesh_lowest_z(mesh_object_list, depsgraph)

    if min_z < 0:
        # Move armature up so the lowest point is at ground level (z=0)
        armature.location.z -= min_z
    armature.location.z += z_offset

    return armature, mesh_object_list