'''
  @ Date: 2024-12-20
  @ Author: Qing Shuai
  @ Description: Visualize foot motion with rings and velocity arrows
    - Green ring: stationary foot
    - Red ring: moving foot with velocity arrow
'''
import os
import numpy as np
import bpy
from mathutils import Vector, Matrix

from myblender.geometry import (
    set_camera,
    build_plane,
    look_at,
)

from myblender.setup import (
    setLight_sun,
    setLight_ambient,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

from myblender.material import add_material, set_principled_node


def create_torus(location=(0, 0, 0), major_radius=0.1, minor_radius=0.02,
                 color=(0, 1, 0, 1), name="torus"):
    """Create a torus (ring) at the specified location with the given color.

    Args:
        location: (x, y, z) position
        major_radius: radius of the ring
        minor_radius: thickness of the ring
        color: RGBA color tuple
        name: name of the object

    Returns:
        The created torus object
    """
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        location=location
    )
    torus = bpy.context.object
    torus.name = name

    # Create and assign material
    mat = bpy.data.materials.new(name=f"Material_{name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Create emission shader for bright color
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs['Color'].default_value = color
    emission_node.inputs['Strength'].default_value = 2.0

    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    torus.data.materials.append(mat)

    # Disable shadow
    if hasattr(torus, 'visible_shadow'):
        torus.visible_shadow = False

    return torus


def create_velocity_arrow(start, velocity, color=(1, 0, 0, 1),
                          cylinder_radius=0.01, cone_radius=0.03,
                          cone_height=0.05, name="arrow"):
    """Create an arrow representing velocity vector.

    Args:
        start: starting position (x, y, z)
        velocity: velocity vector (vx, vy, vz)
        color: RGBA color tuple
        cylinder_radius: radius of the arrow shaft
        cone_radius: radius of the arrow head
        cone_height: height of the arrow head
        name: name prefix for the objects

    Returns:
        Tuple of (cylinder, cone) objects
    """
    start = np.array(start)
    velocity = np.array(velocity)
    speed = np.linalg.norm(velocity)

    if speed < 1e-6:
        return None, None

    # Scale the arrow length based on velocity magnitude
    # Clamp the length for visualization
    arrow_length = min(speed * 0.5, 0.5)  # Scale factor and max length
    direction = velocity / speed
    end = start + direction * arrow_length

    # Create cylinder (arrow shaft)
    length = arrow_length - cone_height
    if length > 0:
        mid_point = start + direction * (length / 2)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=cylinder_radius,
            depth=length,
            location=mid_point
        )
        cylinder = bpy.context.object
        cylinder.name = f"{name}_shaft"

        # Rotate to align with velocity direction
        # Default cylinder is along Z axis
        z_axis = Vector((0, 0, 1))
        dir_vec = Vector(direction)
        rot_quat = z_axis.rotation_difference(dir_vec)
        cylinder.rotation_euler = rot_quat.to_euler()

        # Create and assign material
        mat = bpy.data.materials.new(name=f"Material_{name}_shaft")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            nodes.remove(node)

        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        emission_node = nodes.new(type='ShaderNodeEmission')
        emission_node.inputs['Color'].default_value = color
        emission_node.inputs['Strength'].default_value = 2.0
        links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

        cylinder.data.materials.append(mat)

        if hasattr(cylinder, 'visible_shadow'):
            cylinder.visible_shadow = False
    else:
        cylinder = None

    # Create cone (arrow head)
    cone_location = end - direction * (cone_height / 2)
    bpy.ops.mesh.primitive_cone_add(
        radius1=cone_radius,
        radius2=0,
        depth=cone_height,
        location=cone_location
    )
    cone = bpy.context.object
    cone.name = f"{name}_head"

    # Rotate to align with velocity direction
    z_axis = Vector((0, 0, 1))
    dir_vec = Vector(direction)
    rot_quat = z_axis.rotation_difference(dir_vec)
    cone.rotation_euler = rot_quat.to_euler()

    # Create and assign material
    mat = bpy.data.materials.new(name=f"Material_{name}_head")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs['Color'].default_value = color
    emission_node.inputs['Strength'].default_value = 2.0
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

    cone.data.materials.append(mat)

    if hasattr(cone, 'visible_shadow'):
        cone.visible_shadow = False

    return cylinder, cone


def visualize_foot_motion(foot_positions, foot_velocities,
                          velocity_threshold=0.05,
                          ring_major_radius=0.08,
                          ring_minor_radius=0.015,
                          ring_height_offset=0.01,
                          static_color=(0, 1, 0, 1),
                          moving_color=(1, 0, 0, 1)):
    """Visualize foot motion with rings and velocity arrows.

    Args:
        foot_positions: numpy array of shape (frames, N, 3) - foot positions
        foot_velocities: numpy array of shape (frames, N, 3) - foot velocities
        velocity_threshold: threshold to determine if foot is stationary
        ring_major_radius: radius of the rings
        ring_minor_radius: thickness of the rings
        ring_height_offset: height offset for rings above ground
        static_color: RGBA color for stationary foot
        moving_color: RGBA color for moving foot

    Returns:
        Dictionary containing all created objects organized by frame
    """
    num_frames = foot_positions.shape[0]
    num_feet = foot_positions.shape[1]

    # Store all created objects
    all_objects = {}

    # Create a collection to organize objects
    collection_name = "FootMotionVisualization"
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    for frame_idx in range(num_frames):
        frame_objects = []

        for foot_idx in range(num_feet):
            pos = foot_positions[frame_idx, foot_idx]
            vel = foot_velocities[frame_idx, foot_idx]
            speed = np.linalg.norm(vel)

            is_stationary = speed < velocity_threshold

            # Ring position (slightly above ground)
            ring_pos = (pos[0], pos[1], pos[2] + ring_height_offset)

            # Choose color based on motion state
            color = static_color if is_stationary else moving_color

            # Create ring
            ring_name = f"ring_f{frame_idx:04d}_foot{foot_idx}"
            ring = create_torus(
                location=ring_pos,
                major_radius=ring_major_radius,
                minor_radius=ring_minor_radius,
                color=color,
                name=ring_name
            )

            # Move to collection
            for coll in ring.users_collection:
                coll.objects.unlink(ring)
            collection.objects.link(ring)

            frame_objects.append(ring)

            # If moving, create velocity arrow
            if not is_stationary:
                arrow_name = f"arrow_f{frame_idx:04d}_foot{foot_idx}"
                cylinder, cone = create_velocity_arrow(
                    start=ring_pos,
                    velocity=vel,
                    color=moving_color,
                    name=arrow_name
                )

                if cylinder:
                    for coll in cylinder.users_collection:
                        coll.objects.unlink(cylinder)
                    collection.objects.link(cylinder)
                    frame_objects.append(cylinder)

                if cone:
                    for coll in cone.users_collection:
                        coll.objects.unlink(cone)
                    collection.objects.link(cone)
                    frame_objects.append(cone)

        all_objects[frame_idx] = frame_objects

    return all_objects


def visualize_foot_motion_animated(foot_positions, foot_velocities,
                                    velocity_threshold=0.05,
                                    ring_major_radius=0.08,
                                    ring_minor_radius=0.015,
                                    ring_height_offset=0.01,
                                    static_color=(0, 1, 0, 1),
                                    moving_color=(1, 0, 0, 1)):
    """Visualize foot motion with animated rings that change color based on motion state.

    This version creates objects that animate over time rather than creating
    separate objects for each frame.

    Args:
        foot_positions: numpy array of shape (frames, N, 3) - foot positions
        foot_velocities: numpy array of shape (frames, N, 3) - foot velocities
        velocity_threshold: threshold to determine if foot is stationary
        ring_major_radius: radius of the rings
        ring_minor_radius: thickness of the rings
        ring_height_offset: height offset for rings above ground
        static_color: RGBA color for stationary foot
        moving_color: RGBA color for moving foot

    Returns:
        Dictionary containing created objects
    """
    num_frames = foot_positions.shape[0]
    num_feet = foot_positions.shape[1]

    # Create collection
    collection_name = "FootMotionVisualization"
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    foot_objects = {}

    for foot_idx in range(num_feet):
        # Create ring for this foot
        initial_pos = foot_positions[0, foot_idx]
        ring_pos = (initial_pos[0], initial_pos[1], initial_pos[2] + ring_height_offset)

        ring_name = f"foot_ring_{foot_idx}"
        ring = create_torus(
            location=ring_pos,
            major_radius=ring_major_radius,
            minor_radius=ring_minor_radius,
            color=static_color,
            name=ring_name
        )

        # Move to collection
        for coll in ring.users_collection:
            coll.objects.unlink(ring)
        collection.objects.link(ring)

        # Animate ring position and color
        for frame_idx in range(num_frames):
            pos = foot_positions[frame_idx, foot_idx]
            vel = foot_velocities[frame_idx, foot_idx]
            speed = np.linalg.norm(vel)

            is_stationary = speed < velocity_threshold

            # Set ring position
            ring.location = (pos[0], pos[1], pos[2] + ring_height_offset)
            ring.keyframe_insert(data_path="location", frame=frame_idx + 1)

            # Animate material color
            mat = ring.data.materials[0]
            emission_node = None
            for node in mat.node_tree.nodes:
                if node.type == 'EMISSION':
                    emission_node = node
                    break

            if emission_node:
                color = static_color if is_stationary else moving_color
                emission_node.inputs['Color'].default_value = color
                emission_node.inputs['Color'].keyframe_insert(
                    data_path="default_value", frame=frame_idx + 1
                )

        foot_objects[f"ring_{foot_idx}"] = ring

        # Create arrow objects for this foot (we'll animate visibility)
        arrow_name = f"foot_arrow_{foot_idx}"

        # Create a single arrow that we'll animate
        initial_vel = foot_velocities[0, foot_idx]
        cylinder, cone = create_velocity_arrow(
            start=ring_pos,
            velocity=initial_vel if np.linalg.norm(initial_vel) > velocity_threshold else np.array([0.1, 0, 0]),
            color=moving_color,
            name=arrow_name
        )

        if cylinder:
            for coll in cylinder.users_collection:
                coll.objects.unlink(cylinder)
            collection.objects.link(cylinder)
            foot_objects[f"arrow_shaft_{foot_idx}"] = cylinder

        if cone:
            for coll in cone.users_collection:
                coll.objects.unlink(cone)
            collection.objects.link(cone)
            foot_objects[f"arrow_head_{foot_idx}"] = cone

        # Animate arrow position, rotation and visibility
        for frame_idx in range(num_frames):
            pos = foot_positions[frame_idx, foot_idx]
            vel = foot_velocities[frame_idx, foot_idx]
            speed = np.linalg.norm(vel)

            is_stationary = speed < velocity_threshold

            ring_pos = np.array([pos[0], pos[1], pos[2] + ring_height_offset])

            if not is_stationary and speed > 1e-6:
                # Calculate arrow geometry
                arrow_length = min(speed * 0.5, 0.5)
                direction = vel / speed
                cone_height = 0.05
                shaft_length = arrow_length - cone_height

                if cylinder and shaft_length > 0:
                    mid_point = ring_pos + direction * (shaft_length / 2)
                    cylinder.location = mid_point

                    z_axis = Vector((0, 0, 1))
                    dir_vec = Vector(direction)
                    rot_quat = z_axis.rotation_difference(dir_vec)
                    cylinder.rotation_euler = rot_quat.to_euler()
                    cylinder.scale = (1, 1, shaft_length / 0.1)  # Adjust based on initial depth

                    cylinder.hide_viewport = False
                    cylinder.hide_render = False

                if cone:
                    end = ring_pos + direction * arrow_length
                    cone_location = end - direction * (cone_height / 2)
                    cone.location = cone_location

                    z_axis = Vector((0, 0, 1))
                    dir_vec = Vector(direction)
                    rot_quat = z_axis.rotation_difference(dir_vec)
                    cone.rotation_euler = rot_quat.to_euler()

                    cone.hide_viewport = False
                    cone.hide_render = False
            else:
                # Hide arrow when stationary
                if cylinder:
                    cylinder.hide_viewport = True
                    cylinder.hide_render = True
                if cone:
                    cone.hide_viewport = True
                    cone.hide_render = True

            # Insert keyframes
            if cylinder:
                cylinder.keyframe_insert(data_path="location", frame=frame_idx + 1)
                cylinder.keyframe_insert(data_path="rotation_euler", frame=frame_idx + 1)
                cylinder.keyframe_insert(data_path="scale", frame=frame_idx + 1)
                cylinder.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                cylinder.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)
            if cone:
                cone.keyframe_insert(data_path="location", frame=frame_idx + 1)
                cone.keyframe_insert(data_path="rotation_euler", frame=frame_idx + 1)
                cone.keyframe_insert(data_path="hide_viewport", frame=frame_idx + 1)
                cone.keyframe_insert(data_path="hide_render", frame=frame_idx + 1)

    return foot_objects


def generate_test_data(num_frames=120, num_feet=2):
    """Generate test data simulating walking motion.

    Args:
        num_frames: number of frames
        num_feet: number of feet to simulate

    Returns:
        Tuple of (foot_positions, foot_velocities)
    """
    foot_positions = np.zeros((num_frames, num_feet, 3))
    foot_velocities = np.zeros((num_frames, num_feet, 3))

    # Simulate walking motion
    # Left foot (foot_idx=0) and right foot (foot_idx=1)

    stride_length = 0.6  # meters
    step_duration = 30   # frames per step
    foot_separation = 0.2  # lateral separation

    for frame in range(num_frames):
        t = frame / step_duration

        for foot_idx in range(num_feet):
            # Alternate feet
            phase_offset = foot_idx * 0.5
            phase = (t + phase_offset) % 1.0

            # Forward position (X)
            step_number = int(t + phase_offset)

            if phase < 0.5:
                # Stance phase (foot on ground, stationary)
                x = step_number * stride_length
                z = 0.0
                vx, vy, vz = 0.0, 0.0, 0.0
            else:
                # Swing phase (foot moving)
                swing_progress = (phase - 0.5) * 2  # 0 to 1 during swing
                x = step_number * stride_length + swing_progress * stride_length
                z = 0.1 * np.sin(swing_progress * np.pi)  # Arc trajectory

                # Velocity during swing
                vx = stride_length / (step_duration * 0.5) * 30  # Convert to per-frame velocity
                vz = 0.1 * np.pi * np.cos(swing_progress * np.pi) / (step_duration * 0.5) * 30
                vy = 0.0

            # Lateral position (Y)
            y = (foot_idx - 0.5) * foot_separation

            foot_positions[frame, foot_idx] = [x, y, z]
            foot_velocities[frame, foot_idx] = [vx, vy, vz]

    return foot_positions, foot_velocities


if __name__ == '__main__':
    # $blender  --python examples/render_foot_motion.py -- --test --animated --out output/
    parser = get_parser()
    parser.add_argument('--foot_data', type=str, default=None,
                        help='Path to .npz file containing foot_positions and foot_velocities')
    parser.add_argument('--velocity_threshold', type=float, default=0.05,
                        help='Velocity threshold to determine stationary foot')
    parser.add_argument('--animated', action='store_true',
                        help='Use animated visualization (single objects that change over time)')
    parser.add_argument('--test', action='store_true',
                        help='Use test data instead of loading from file')
    args = parse_args(parser)

    setup()

    # Camera setup
    camera = set_camera(location=(2, -3, 2), center=(1.5, 0, 0.2), focal=35)

    # Lighting
    lightAngle = [-45, -45, 0]
    strength = 2
    shadowSoftness = 0.3
    sun = setLight_sun(lightAngle, strength, shadowSoftness)
    setLight_ambient(color=(0.1, 0.1, 0.1, 1))

    # Ground plane
    build_plane(translation=(0, 0, 0), plane_size=10)

    # Load or generate foot motion data
    if args.test or args.foot_data is None:
        print("Using test data for foot motion visualization")
        foot_positions, foot_velocities = generate_test_data(num_frames=120, num_feet=2)
    else:
        print(f"Loading foot data from: {args.foot_data}")
        data = np.load(args.foot_data)
        foot_positions = data['foot_positions']  # Expected shape: (frames, N, 3)
        foot_velocities = data['foot_velocities']  # Expected shape: (frames, N, 3)

    print(f"Foot positions shape: {foot_positions.shape}")
    print(f"Foot velocities shape: {foot_velocities.shape}")

    # Set scene frame range
    num_frames = foot_positions.shape[0]
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames

    # Visualize foot motion
    if args.animated:
        print("Creating animated visualization...")
        foot_objects = visualize_foot_motion_animated(
            foot_positions, foot_velocities,
            velocity_threshold=args.velocity_threshold,
            static_color=(0, 1, 0, 1),    # Green for stationary
            moving_color=(1, 0, 0, 1),    # Red for moving
        )
    else:
        print("Creating per-frame visualization...")
        # For per-frame visualization, we need to handle visibility per frame
        # This is simpler but creates many objects
        all_objects = visualize_foot_motion(
            foot_positions, foot_velocities,
            velocity_threshold=args.velocity_threshold,
            static_color=(0, 1, 0, 1),
            moving_color=(1, 0, 0, 1),
        )

        # Set up visibility animation for per-frame objects
        for frame_idx, frame_objects in all_objects.items():
            for obj in frame_objects:
                # Make object visible only at its frame
                for f in range(1, num_frames + 1):
                    obj.hide_viewport = (f != frame_idx + 1)
                    obj.hide_render = (f != frame_idx + 1)
                    obj.keyframe_insert(data_path="hide_viewport", frame=f)
                    obj.keyframe_insert(data_path="hide_render", frame=f)

    # Setup renderer
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=64,
        use_transparent_bg=False,
        use_denoising=True,
    )

    outdir = args.out
    if not outdir.endswith(os.path.sep):
        outdir = outdir + os.path.sep

    set_output_properties(
        bpy.context.scene,
        output_file_path=outdir,
        res_x=1920,
        res_y=1080,
        tile_x=1920,
        tile_y=1080,
        resolution_percentage=100,
        format='JPEG'
    )

    print(f"Scene ready with {num_frames} frames")
    print(f"Output will be saved to: {outdir}")

