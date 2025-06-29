import os
import bpy

def setup_video_in_3d(video_path, down=1):
    # 清理场景    
    # 获取视频信息
    assert os.path.exists(video_path), f"视频文件不存在: {video_path}"
    
    # 加载视频剪辑
    video_clip = bpy.data.movieclips.load(video_path)
    total_frames = video_clip.frame_duration
    width, height = video_clip.size
    print()
    if down > 1:
        width = width // down
        height = height // down
    fps = video_clip.fps
    # 创建平面并匹配视频比例
    bpy.ops.mesh.primitive_plane_add(size=2)
    plane = bpy.context.active_object
    plane.scale.x = width / min(width, height)
    plane.scale.y = height / min(width, height)
    # 旋转平面使其成为xz平面（绕x轴旋转90度）
    plane.rotation_euler[0] = 1.5708  # 90度，转换为弧度是π/2
    # 绕z轴旋转180度
    plane.rotation_euler[2] = 3.1416  # 180度，转换为弧度是π
    # 将平面底端放置在地面上
    # 由于旋转后，平面的高度是沿着z轴的，需要将平面上移一半高度
    plane.location.z = plane.scale.y  # 将平面上移其高度的一半
    
    # 确保平面位于场景中心的地面上
    plane.location.x = 0
    plane.location.y = -3.
    
    
    # 创建视频材质
    material = bpy.data.materials.new(name="VideoMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # 清除默认节点
    for node in nodes:
        nodes.remove(node)
    
    # 创建必要节点
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    mapping = nodes.new(type='ShaderNodeMapping')
    image_texture = nodes.new(type='ShaderNodeTexImage')
    emission = nodes.new(type='ShaderNodeEmission')
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # 节点布局
    tex_coord.location = (-600, 0)
    mapping.location = (-400, 0)
    image_texture.location = (-200, 0)
    emission.location = (200, 0)
    output.location = (400, 0)
    
    # 链接节点
    links.new(tex_coord.outputs['UV'], mapping.inputs[0])
    links.new(mapping.outputs[0], image_texture.inputs[0])
    links.new(image_texture.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # 设置视频纹理
    # 关键修正：将MovieClip转换为Image
    # 加载视频为IMAGE序列
    img = bpy.data.images.load(video_path)
    img.source = 'MOVIE'  # 设置为视频模式
    
    # 应用到图像纹理节点
    image_texture.image = img

    image_texture.image_user.frame_duration = total_frames
    image_texture.image_user.use_auto_refresh = True
    
    # 应用材质到平面
    plane.data.materials.append(material)
    return width, height, int(fps)
    # 设置时间轴范围
    # scene = bpy.context.scene
    # scene.frame_start = 1
    # scene.frame_end = total_frames
    # scene.render.fps = int(fps)
    
    # # 设置摄像机位置
    # camera_pos = (0, 0, max(width, height) * 0.8 / min(width, height))
    # if not scene.camera:
    #     bpy.ops.object.camera_add(location=camera_pos)
    #     scene.camera = bpy.context.active_object
    # else:
    #     scene.camera.location = camera_pos
    # scene.camera.rotation_euler = (0, 0, 0)
    
    # 配置渲染设置
    # scene.render.image_settings.file_format = 'FFMPEG'
    # scene.render.ffmpeg.format = 'MPEG4'
    # scene.render.ffmpeg.codec = 'H264'
    # scene.render.resolution_x = width
    # scene.render.resolution_y = height
    # scene.render.resolution_percentage = 100
    # scene.render.filepath = video_path + "_output.mp4"

    # 设置自动播放时间线（可选）
    # bpy.ops.screen.animation_play()
    
    # print(f"视频设置完成！帧率: {fps}FPS, 总帧数: {total_frames}, 分辨率: {width}x{height}")