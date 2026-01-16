'''
  @ Date: 2024-12-24
  @ Author: Qing Shuai
  @ Description: Render a sequence of sampled frames from an animation file (FBX)
                 arranged from left to right along the x-axis as a single shot.
'''
import os
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
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

from myblender.material import set_material_i, add_material


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


def load_fbx_at_frame(fbx_path, frame, x_offset, instance_id, material_index=0, target_frame=1):
    """
    Load FBX file and shift animation so that the specified frame becomes target_frame.

    Args:
        fbx_path: Path to the FBX file
        frame: The original frame number to extract
        x_offset: X-axis offset for positioning
        instance_id: Unique ID for naming
        material_index: Index for material color
        target_frame: The frame number where the specified frame should appear (default: 1)

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
    armature.location.x += x_offset

    # Shift animation so that 'frame' becomes 'target_frame'
    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        # Calculate offset: we want frame -> target_frame, so offset = target_frame - frame
        offset = target_frame - frame
        shift_action_frames(action, offset)

    # Rename objects to avoid conflicts
    for obj_name in obj_names:
        obj = bpy.data.objects[obj_name]
        obj.name = f"{obj_name}_inst{instance_id}_f{frame}"

    # Set the scene to target_frame to display the desired pose
    # bpy.context.scene.frame_set(target_frame)

    # Set material for mesh objects only if no existing materials
    for mesh_obj in mesh_object_list:
        if len(mesh_obj.data.materials) == 0:
            # Add new material
            matname = f"Material_inst{instance_id}_f{frame}"
            mat = add_material(matname, use_nodes=True, make_node_tree_empty=False)
            mesh_obj.data.materials.append(mat)
            set_material_i(bpy.data.materials[matname], material_index, use_plastic=False)

    return armature, mesh_object_list


def sample_frames(frame_start, frame_end, num_frames=8, frame_indices=None):
    """
    Get the frame indices to sample.

    Args:
        frame_start: Start frame of animation
        frame_end: End frame of animation
        num_frames: Number of frames to sample (default 8)
        frame_indices: Manual list of frame indices (overrides num_frames)

    Returns:
        List of frame indices
    """
    if frame_indices is not None:
        # Use manually specified frame indices
        return [int(f) for f in frame_indices]
    else:
        # Uniformly sample frames
        total_frames = frame_end - frame_start + 1
        if num_frames >= total_frames:
            return list(range(frame_start, frame_end + 1))
        else:
            # Sample uniformly including first and last frames
            indices = np.linspace(frame_start, frame_end, num_frames, dtype=int)
            return indices.tolist()


def get_animation_range(fbx_path):
    """
    Load FBX temporarily to get animation frame range, then delete it.

    Returns:
        Tuple of (frame_start, frame_end)
    """
    # Track objects before import
    keys_old = set(bpy.data.objects.keys())

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    keys_new = set(bpy.data.objects.keys())
    obj_names = list(keys_new - keys_old)

    # Find armature
    armature, _, _ = find_armature_and_mesh(obj_names)

    # Get frame range
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])

    # Delete the temporarily imported objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj_name in obj_names:
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].select_set(True)
    bpy.ops.object.delete()

    return frame_start, frame_end


if __name__ == '__main__':
    # ${blender} -noaudio --python examples/render_smpl_shot.py -- test_baseline.fbx --num_frames 8 --material_index 0 --plane_size 20.0
    parser = get_parser()
    parser.add_argument('--num_frames', type=int, default=6,
                        help='Number of frames to sample and visualize (default: 8)')
    parser.add_argument('--skip_start', type=int, default=10,
                        help='Number of frames to sample and visualize (default: 8)')
    parser.add_argument('--skip_end', type=int, default=10,
                        help='Number of frames to sample and visualize (default: 8)')

    parser.add_argument('--frame_indices', type=int, nargs='+', default=None,
                        help='Manual list of frame indices to visualize (overrides --num_frames)')
    parser.add_argument('--spacing', type=float, default=1.0,
                        help='Spacing between frames along x-axis (default: 1.0)')
    parser.add_argument('--material_index', type=int, default=1,
                        help='Material color index (default: 0)')
    args = parse_args(parser)

    # Setup scene
    setup()

    # Load the FBX file
    fbx_path = args.path
    assert fbx_path.endswith('.fbx') or fbx_path.endswith('.FBX'), \
        f"Input file must be an FBX file, got: {fbx_path}"

    print(f"Loading FBX file: {fbx_path}")

    # First, get animation frame range
    frame_start, frame_end = get_animation_range(fbx_path)
    frame_start += args.skip_start
    frame_end -= args.skip_end
    print(f"Animation frame range: {frame_start} to {frame_end}")

    # Get frames to sample
    frames_to_sample = sample_frames(
        frame_start, frame_end,
        num_frames=args.num_frames,
        frame_indices=args.frame_indices
    )
    print(f"Sampling frames: {frames_to_sample}")

    # Calculate x positions (centered around 0)
    num_samples = len(frames_to_sample)
    x_positions = [(i - (num_samples - 1) / 2) * args.spacing for i in range(num_samples)]

    # Load FBX for each sampled frame
    all_armatures = []
    all_meshes = []
    for i, (frame, x_pos) in enumerate(zip(frames_to_sample, x_positions)):
        print(f"Loading instance {i+1}/{num_samples}: frame {frame} at x={x_pos:.2f}")
        armature, mesh_list = load_fbx_at_frame(
            fbx_path, frame, x_pos,
            instance_id=i,
            material_index=args.material_index
        )
        if armature:
            all_armatures.append(armature)
        all_meshes.extend(mesh_list)

    # Calculate camera position based on the arrangement
    camera_distance = 20
    total_width = (num_samples - 1) * args.spacing

    camera = set_camera(
        location=(0, -25, 1.5),
        center=(0, 0, 1),
        focal=100
    )

    # Setup lighting
    lightAngle = [45, 45, 0]
    strength = 3
    shadowSoftness = 0.3
    sun = setLight_sun(lightAngle, strength, shadowSoftness)
    setLight_ambient(color=(0.1, 0.1, 0.1, 1))

    # Build ground plane
    build_plane(translation=(0, 0, -0.05), plane_size=num_samples * args.spacing)

    # Setup renderer
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=256,
        use_transparent_bg=True,
        use_denoising=True,
    )

    # Set output properties
    outdir = args.out

    set_output_properties(
        bpy.context.scene,
        output_file_path=outdir,
        res_x=2048,
        res_y=512,
        tile_x=2048,
        tile_y=512,
        resolution_percentage=100,
        format='PNG'
    )

    # Set to frame 1 for single frame render
    bpy.context.scene.frame_set(1)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1

    # Save blend file if specified
    if args.out_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)
        print(f"Saved blend file to: {args.out_blend}")

    print(f"Scene setup complete. Output will be saved to: {outdir}")
    print(f"Sampled {num_samples} frames: {frames_to_sample}")
    bpy.ops.render.render(write_still=True)
