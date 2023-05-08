import bpy
import numpy as np
from mathutils import Vector, Quaternion, Matrix

def get_calibration_matrix_K_from_blender(mode='simple'):

    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px

    camdata = scene.camera.data

    if mode == 'simple':

        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':

        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else: # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0 # only use rectangular pixels

        K = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ], dtype=np.float32)
    
    return K

def set_intrinsic(K, camera, sensor_width=1.0):
    scene = bpy.context.scene
    # Intrinsic
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    image_width = c_x * 2  # principal point x assumed at the center
    cam = camera.data
    cam.name = 'CamFrom3x4P'
    cam.type = 'PERSP'
    cam.lens = f_x / image_width * sensor_width
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width = sensor_width
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = f_y / f_x
    Knew = get_calibration_matrix_K_from_blender(mode='complete')
    print(K)
    print(Knew)

def set_extrinsic(R_world2cv, T_world2cv, camera):
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_cv2world = R_world2cv.T
    rotation = Matrix(R_cv2world.tolist()) @ R_bcam2cv
    location = -R_cv2world @ T_world2cv
    camera.location = location
    camera.matrix_world = Matrix.Translation(location) @ rotation.to_4x4()