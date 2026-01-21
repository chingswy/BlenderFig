import os
import bpy
import sys
import math
import time
import numpy as np
from mathutils import Vector
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myblender.setup import (
    setup,
    set_cycles_renderer,
    set_output_properties,
    get_parser,
    parse_args,
    render_with_progress,
)
from myblender.geometry import set_camera, build_plane, create_camera_blender
from myblender.material import set_material_i, add_material
from myblender.fbxtools import load_fbx_at_frame
from myblender.material import set_material_i, add_material
from myblender.geometry import create_simple_arrow, add_root_trajectory

# =====================================================
# Main Script
# =====================================================

if __name__ == '__main__':
    # ${blender} -noaudio --python examples/fig_v2m/render_wv_space.py -- /Users/shuaiqing/Desktop/t2m/00000000_00.fbx
    parser = get_parser()
    parser.add_argument("--ground", default=(200/255, 200/255, 200/255, 1.0), nargs=4, type=float,
    help="Ground color: (94/255, 124/255, 226/255, 1.0) 青色")
    parser.add_argument("--body", default=[0.05, 0.326, 1.], nargs=3, type=float)
    args = parse_args(parser)
    
    fbx_path = args.path
    assert os.path.exists(fbx_path), f"FBX file not found: {fbx_path}"
    
    # Initialize scene
    setup()
    arrow_cfg = {
        'cylinder_radius': 0.05,
        'cone_radius': 0.1,
        'cone_height': 0.2,
    }
    # x
    create_simple_arrow(start=(0, 0, 0), end=(2, 0, 0), color=(1, 0, 0), **arrow_cfg)
    # y
    create_simple_arrow(start=(0, 0, 0), end=(0, 0, 2), color=(0, 1, 0), **arrow_cfg)
    # z
    create_simple_arrow(start=(0, 0, 0), end=(0, -2, 0), color=(0, 0, 1), **arrow_cfg)
    # Configuration
    camera_height = 2
    rotation_angle = 45  # Convert to radians
    vis_camera_space = True
    build_plane(translation=(0, 0, 0), plane_size=100,
                white=(1,1,1,1), black=args.ground,
                roughness=0.1, metallic=0.8, specular=0.8)

    # 画坐标轴
    set_camera(
        location=(0, -11, 7),
        center=(0, 0, 1),
        focal=40  # Portrait lens focal length
    )
    # =====================================================
    # Step 1: Get animation range
    # =====================================================
    print("Getting animation range...")
    keyframes = [0, 10, 20, 30, 40, 50, 60]
    print(f"Sampled keyframes: {keyframes}")
    
    # =====================================================
    # Step 3: Load standard coordinate characters (left side)
    # =====================================================
    print("\n=== Loading Standard Coordinate Characters ===")
    standard_armatures = []
    standard_positions = []  # (x, y, z) positions for trajectory
    
    for idx, frame in enumerate(keyframes):
        # Calculate x offset for this character        
        # Load FBX at specific frame
        armature, mesh_list = load_fbx_at_frame(
            fbx_path,
            frame,
            x_offset=0,
            y_offset=0,
            target_frame=1,
            z_rotation=rotation_angle,
            rotate_trajectory=True
        )
        for mesh_obj in mesh_list:
            set_material_i(mesh_obj, tuple(args.body), use_plastic=False)

        # Store armature and position
        standard_armatures.append((armature, mesh_list))
        
        # Get pelvis/root position for trajectory
        bpy.context.scene.frame_set(1)
        bpy.context.view_layer.update()
        pelvis_pos = (0, 0, 1.0)  # Approximate pelvis height
        standard_positions.append(pelvis_pos)
    
    # =====================================================
    # Step 4: Create orbiting cameras for standard coordinates
    # =====================================================
    print("\n=== Creating Standard Coordinate Cameras ===")
    standard_cameras = []
    
    # Camera orbits from right-front (45 deg) to left-front (-45 deg)
    # Angle 0 = directly in front (negative Y direction)

    # 用 angle 旋转一下 camera_center
    base_center = np.array([0, 0, 1])
    # 使用rotation_angle是step 3中的旋转（degree），要转为弧度
    angle_rad = math.radians(rotation_angle)
    rot_mat = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad)],
        [math.sin(angle_rad),  math.cos(angle_rad)]
    ])
    # 只旋转xy，z保持
    camera_center_xy = rot_mat @ base_center[:2]
    camera_center = np.array([camera_center_xy[0], camera_center_xy[1], base_center[2]])

    camera_configs = [
        {
            'radius': [4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6],
            'angle': [-45, -30, -15, 0, 15, 30, 45],
            'height': 2,
            'vid': 0
        },
        {
            'radius': [4, 4, 4, 4, 4, 4, 4],
            'angle': [-45, -60, -75, -90, -105, -120, -135],
            'height': 1,
            'vid': 1
        },

    ]

    for camera_config in camera_configs:
        camera_height = camera_config['height']
        angles = camera_config['angle']

        positions = []

        for index, angle_deg in enumerate(angles):
            camera_radius = camera_config['radius'][index]
            angle = math.radians(angle_deg + rotation_angle)
            
            # Camera position (orbiting around character center)
            cam_x = camera_center[0] + camera_radius * math.sin(angle)
            cam_y = camera_center[1] - camera_radius * math.cos(angle)  # Negative Y is front
            cam_z = camera_center[2] + camera_height

            cam_pos = np.array([cam_x, cam_y, cam_z])
            positions.append(Vector(cam_pos))
            look_at_pos = camera_center
            
            # Calculate world to camera R, T matrices
            # Camera coordinate system: -Z is forward (looking direction), Y is up, X is right
            forward = look_at_pos - cam_pos
            forward = forward / np.linalg.norm(forward)  # Normalize, points toward target
            
            world_up = np.array([0, 0, 1])
            
            # Handle case where forward is nearly parallel to world_up
            if abs(np.dot(forward, world_up)) > 0.999:
                right = np.array([1, 0, 0])
            else:
                # Right = world_up × forward (right-hand rule: thumb=up, fingers curl toward forward, palm faces right)
                right = np.cross(world_up, forward)
                right = right / np.linalg.norm(right)
            
            # Up = forward × right (perpendicular to both, forms right-handed system)
            up = np.cross(forward, right)
            up = up / np.linalg.norm(up)
            
            # Camera to world rotation matrix (columns are camera axes in world coords)
            # The pyramid shape in create_camera_blender has its tip at +Z direction
            # So we want camera's +Z to point toward target (forward)
            R_cam2world = np.column_stack([right, up, forward])
            
            # World to camera rotation matrix
            R = R_cam2world.T
            
            # World to camera translation: T = R @ (-cam_pos) = -R @ cam_pos
            # Actually for world_to_camera: point_cam = R @ point_world + T
            # Camera position in camera coords should be origin, so:
            # 0 = R @ cam_pos + T => T = -R @ cam_pos
            T = -R @ cam_pos
            T = T.reshape(3, 1)  # Column vector
            
            # Create camera visualization using R, T
            cam_viz = create_camera_blender(R, T, scale=0.3, pid=camera_config['vid'])
            
            standard_cameras.append({
                'viz': cam_viz,
                'position': cam_pos,
                'look_at': look_at_pos,
                'angle': angle_deg
            })
            
            print(f"Camera {idx}: pos=({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}), angle={angle_deg}°")
        # trajectory_curve = add_root_trajectory(
        #     positions,
        #     start_color=(1.0, 0.3, 0.3, 1.0),   # Stronger red (start of motion, less pale)
        #     end_color=(0.6, 0.0, 0.0, 1.0),     # Deep red (end of motion)
        #     line_thickness=0.025,
        #     emission_strength=2.5,
        #     curve_name=f"CameraTrajectory_{camera_config['vid']}",
        #     extend_start=0.8,    # Extend before first character
        #     extend_end=0.8,      # Extend after last character
        #     add_arrow=True,      # Add arrowhead at the end
        #     arrow_scale=1.2      # Slightly larger arrow
        # )
        
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=512,
        use_transparent_bg=False,
        use_denoising=True,
    )


    set_output_properties(
        bpy.context.scene,
        output_file_path='output/wv_space.jpg',
        res_x=(2048+1024)//2,
        res_y=(1024+1024)//2,
        tile_x=512,
        tile_y=512,
        resolution_percentage=100,
        format='JPEG'
    )

    if not args.debug:
        render_with_progress(write_still=True)