'''
  @ Date: 2022-04-24 16:09:24
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-31 11:48:52
  @ FilePath: /EasyMocapPublic/easymocap/blender/material.py
'''
import os
import bpy

hex2rgb = lambda x:list(map(lambda v:int('0x'+v, 16)/255., [x[:2], x[2:4], x[4:]]))

colors_rgb = [
    (94/255, 124/255, 226/255), # 青色
    (255/255, 200/255, 87/255), # yellow
    (74/255.,  189/255.,  172/255.), # green
    (8/255, 76/255, 97/255), # blue
    (219/255, 58/255, 52/255), # red
    (77/255, 40/255, 49/255), # brown
    (110/255, 211/255, 207/255), # light green
    (23/255, 126/255, 137/255),
    (74/255.,  189/255.,  172/255.), # 2 green
    (146/255, 180/255, 244/255), # 3 blue
    (8/255, 76/255, 97/255),
    (219/255, 58/255, 52/255), # 4 red
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    # (82/255, 97/255, 135/255),
    # (157/255, 215/255, 213/255),
    # (0.65098039, 0.74117647, 0.85882353),
    # (251/255.,  128/255.,  114/255.),
    # (219/255, 58/255, 52/255), # 2 green
    # (0.2, 0.2, 0),
    # (0, 0.8, 0),
    # (1, 0, 1),
    # (0, 0, 1),
    # (0, 0.2, 0.2),
    # (1, 0, 0)
]

colors_table = {
    'gray': [194/255, 157/255, 115/255],
    'b': [0.65098039, 0.74117647, 0.85882353],
    '_pink': [.9, .7, .7],
    '_mint': [ 166/255.,  229/255.,  204/255.],
    '_mint2': [ 202/255.,  229/255.,  223/255.],
    '_green': [ 153/255.,  216/255.,  201/255.],
    '_green2': [ 171/255.,  221/255.,  164/255.],
    'r': [ 251/255.,  128/255.,  114/255.],
    '_orange': [ 253/255.,  174/255.,  97/255.],
    'y': [ 250/255.,  230/255.,  154/255.],
    '_r':[255/255,0,0],
    'g':[0,255/255,0],
    '_b':[0,0,255/255],
    'k':[0,0,0],
    'w':[1,1,1],
    '_y':[255/255,255/255,0],
    'purple':[128/255,0,128/255],
    'smap_b':[51/255,153/255,255/255],
    'smap_r':[255/255,51/255,153/255],
    'smap_b':[51/255,255/255,153/255],
}

def get_rgb(pid):
    if isinstance(pid, str) and pid in colors_table:
        col = colors_table[pid]
    elif isinstance(pid, str):
        col = hex2rgb(pid)
    elif isinstance(pid, list) or isinstance(pid, tuple):
        col = pid[:3]
    else:
        col = colors_rgb[pid]
    col = list(col) + [1]
    return col

def set_principled_node(principled_node: bpy.types.Node,
                        base_color = (1., 1., 1., 1.0),
                        subsurface: float = 0.0,
                        subsurface_color = (1., 1., 1., 1.0),
                        subsurface_radius = (1.0, 0.2, 0.1),
                        metallic: float = 0.0,
                        specular: float = 0.5,
                        specular_tint: float = 0.0,
                        roughness: float = 0.5,
                        anisotropic: float = 0.0,
                        anisotropic_rotation: float = 0.0,
                        sheen: float = 0.0,
                        sheen_tint: float = 0.5,
                        clearcoat: float = 0.0,
                        clearcoat_roughness: float = 0.03,
                        ior: float = 1.45,
                        transmission: float = 0.0,
                        transmission_roughness: float = 0.0,
                        alpha: float=1.0) -> None:
    principled_node.inputs['Base Color'].default_value = base_color
    principled_node.inputs['Subsurface'].default_value = subsurface
    principled_node.inputs['Subsurface Color'].default_value = subsurface_color
    principled_node.inputs['Subsurface Radius'].default_value = subsurface_radius
    principled_node.inputs['Metallic'].default_value = metallic
    principled_node.inputs['Specular'].default_value = specular
    principled_node.inputs['Specular Tint'].default_value = specular_tint
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['Anisotropic'].default_value = anisotropic
    principled_node.inputs['Anisotropic Rotation'].default_value = anisotropic_rotation
    principled_node.inputs['Sheen'].default_value = sheen
    principled_node.inputs['Sheen Tint'].default_value = sheen_tint
    principled_node.inputs['Clearcoat'].default_value = clearcoat
    principled_node.inputs['Clearcoat Roughness'].default_value = clearcoat_roughness
    principled_node.inputs['IOR'].default_value = ior
    principled_node.inputs['Transmission'].default_value = transmission
    principled_node.inputs['Transmission Roughness'].default_value = transmission_roughness
    principled_node.inputs['Alpha'].default_value = alpha

def build_pbr_nodes(node_tree,
                    base_color, **kwargs) -> None:
    # output_node = node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    # principled_node = node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
    # node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    principled_node = node_tree.nodes["Principled BSDF"]
    set_principled_node(principled_node=principled_node,
                        base_color=base_color,
                        **kwargs)

def clean_nodes(nodes: bpy.types.Nodes) -> None:
    for node in nodes:
        nodes.remove(node)

def add_material(name: str = "Material",
                 use_nodes: bool = False,
                 make_node_tree_empty: bool = False) -> bpy.types.Material:
    '''
    https://docs.blender.org/api/current/bpy.types.BlendDataMaterials.html
    https://docs.blender.org/api/current/bpy.types.Material.html
    '''

    # TODO: Check whether the name is already used or not

    material = bpy.data.materials.new(name)
    material.use_nodes = use_nodes

    if use_nodes and make_node_tree_empty:
        clean_nodes(material.node_tree.nodes)

    return material

class colorObj(object):
    def __init__(self, RGBA, \
    H = 0.5, S = 1.0, V = 1.0,\
    B = 0.0, C = 0.0):
        self.H = H # hue
        self.S = S # saturation
        self.V = V # value
        self.RGBA = RGBA
        self.B = B # birghtness
        self.C = C # contrast

def setMat_plastic(mesh, meshColor, AOStrength = 0.0, alpha=1.,
                   roughness=0.1, metallic=0.2, specular=0.6, **kwargs):
    mat = bpy.data.materials.new('MeshMaterial')
    mesh.data.materials.append(mat)
    mesh.active_material = mat
    mat.use_nodes = True
    tree = mat.node_tree
    # set principled BSDF
    tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = roughness
    tree.nodes["Principled BSDF"].inputs['Metallic'].default_value = metallic
    tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Specular'].default_value = specular
    tree.nodes["Principled BSDF"].inputs['IOR'].default_value = 1.45
    tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Clearcoat Roughness'].default_value = 0
    tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = alpha

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

def set_material_i(mat, pid, metallic=0.5, specular=0.5, roughness=0.9, use_plastic=True, **kwargs):
    if isinstance(pid, int):
        color = get_rgb(pid)
    else:
        color = get_rgb(pid)
    print(f"Setting material color: {color}")
    if not use_plastic:
        # Handle both mesh objects and material objects
        if isinstance(mat, bpy.types.Object):
            # mat is actually a mesh object, get its active material
            if mat.active_material is None:
                # Create a new material if none exists
                new_mat = bpy.data.materials.new('MeshMaterial')
                new_mat.use_nodes = True
                mat.data.materials.append(new_mat)
                mat.active_material = new_mat
            material = mat.active_material
        else:
            material = mat
        build_pbr_nodes(material.node_tree, base_color=color,
            metallic=metallic, specular=specular, roughness=roughness, **kwargs)
    else:
        setMat_plastic(mat, colorObj(color, B=0.3))

def setHDREnv(fn, strength=1.0):
    if fn is None:
        return
    assert os.path.isfile(fn)
    scene = bpy.context.scene
    # Get the environment node tree of the current scene
    node_tree = scene.world.node_tree
    tree_nodes = node_tree.nodes

    # Clear all nodes
    node_tree.nodes.clear()
    # Add Background node
    node_background = node_tree.nodes.new(type="ShaderNodeBackground")
    node_background.inputs["Strength"].default_value = strength  # reduce env lighting
    # Add Environment Texture node
    node_environment = node_tree.nodes.new("ShaderNodeTexEnvironment")
    # Load and assign the image to the node property
    node_environment.image = bpy.data.images.load(fn)  # Relative path
    node_environment.location = -300, 0
    # Add Output node
    node_output = node_tree.nodes.new(type="ShaderNodeOutputWorld")
    node_output.location = 200, 0

    # Link all nodes
    node_tree.links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    node_tree.links.new(node_background.outputs["Background"], node_output.inputs["Surface"])


def setup_mist_fog(scene, start=5.0, depth=20.0, fog_color=(0.7, 0.8, 0.9)):
    """
    修正后的雾气设置：
    1. 增强了节点连接的鲁棒性（适配新旧版本Blender）。
    2. 移除了 Set Alpha 节点，确保雾气能覆盖背景。
    3. 默认颜色稍微调深一点蓝色，以便在白色背景下能看清。
    """
    # 1. 开启 Mist Pass
    scene.view_layers[0].use_pass_mist = True

    # 2. 配置世界 Mist 参数
    scene.world.mist_settings.use_mist = True
    scene.world.mist_settings.start = start
    scene.world.mist_settings.depth = depth
    scene.world.mist_settings.falloff = 'QUADRATIC'

    # 3. 设置合成节点
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    nodes.clear()

    # --- 节点创建 ---

    # 渲染层输入
    render_layers = nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (-300, 0)

    # 雾的颜色 (稍微带点蓝灰，增加对比度)
    fog_color_node = nodes.new(type='CompositorNodeRGB')
    fog_color_node.location = (-100, 200)
    fog_color_node.outputs[0].default_value = (*fog_color, 1.0)

    # 添加 Bright/Contrast 节点来增强雾色亮度，确保呈现明亮的白色
    bright_contrast = nodes.new(type='CompositorNodeBrightContrast')
    bright_contrast.location = (0, 200)
    bright_contrast.inputs['Bright'].default_value = 0.3  # 增加亮度
    bright_contrast.inputs['Contrast'].default_value = 0.0  # 保持对比度不变

    # 添加 Math 节点来增强 Mist pass 强度，使雾更快达到全强度
    # 使用公式：clamp((value - 0.3) / 0.7, 0, 1) 将 [0.3, 1.0] 映射到 [0.0, 1.0]
    # 这样雾会在更近的距离达到全强度
    math_subtract = nodes.new(type='CompositorNodeMath')
    math_subtract.location = (-300, 0)
    math_subtract.operation = 'SUBTRACT'
    math_subtract.inputs[1].default_value = 0.3  # 减去 0.3
    
    math_divide = nodes.new(type='CompositorNodeMath')
    math_divide.location = (-200, 0)
    math_divide.operation = 'DIVIDE'
    math_divide.inputs[1].default_value = 0.7  # 除以 0.7 (1.0 - 0.3)
    
    # 使用 MAXIMUM 和 MINIMUM 来实现 clamp 功能
    math_max = nodes.new(type='CompositorNodeMath')
    math_max.location = (-100, 0)
    math_max.operation = 'MAXIMUM'
    math_max.inputs[1].default_value = 0.0  # 确保最小值是 0
    
    math_min = nodes.new(type='CompositorNodeMath')
    math_min.location = (-50, 0)
    math_min.operation = 'MINIMUM'
    math_min.inputs[1].default_value = 1.0  # 确保最大值是 1

    # Mix 节点 (核心)
    # 自动适配 Blender 版本差异
    try:
        # Blender 3.4+ / 4.0+
        mix = nodes.new(type='CompositorNodeMix')
        mix.data_type = 'RGBA'
        mix.blend_type = 'MIX'
        # 新版 Mix 节点的输入索引：0:Factor, 1:A, 2:B (如果类型是RGBA)
        # 我们使用索引连接比使用名称更安全
        input_fac = 0
        input_image_a = 6 # 在某些版本RGBA模式下，A是6，B是7，或者直接按顺序
        # 为了最安全，我们下面通过连线逻辑动态判断，或者使用 MixRGB (旧节点兼容性最好)
    except:
        pass

    # 为了保证绝对兼容性，推荐直接使用 MixRGB (在Python API中依然可用且稳定)
    mix = nodes.new(type='CompositorNodeMixRGB')
    mix.blend_type = 'MIX'
    mix.location = (100, 0)

    # 输出节点
    composite = nodes.new(type='CompositorNodeComposite')
    composite.location = (400, 0)

    # --- 节点连接 ---

    # 逻辑：
    # Factor = Mist Pass (经过 Math 节点增强，距离越远，值越大)
    # Input 1 (Top) = 原始图像 (近处清晰)
    # Input 2 (Bottom) = 雾颜色 (经过 Bright/Contrast 增强，远处全是雾)

    # 1. Mist 经过 Math 节点处理：subtract(0.3) -> divide(0.7) -> max(0) -> min(1)，然后连接到 Factor
    links.new(render_layers.outputs['Mist'], math_subtract.inputs[0])
    links.new(math_subtract.outputs['Value'], math_divide.inputs[0])
    links.new(math_divide.outputs['Value'], math_max.inputs[0])
    links.new(math_max.outputs['Value'], math_min.inputs[0])
    links.new(math_min.outputs['Value'], mix.inputs['Fac'])

    # 2. 原始图像 连接到 Image 1
    links.new(render_layers.outputs['Image'], mix.inputs[1])

    # 3. 雾颜色 经过 Bright/Contrast 增强后连接到 Image 2
    links.new(fog_color_node.outputs['RGBA'], bright_contrast.inputs['Image'])
    links.new(bright_contrast.outputs['Image'], mix.inputs[2])

    # 4. 结果直接输出 (不要再 Set Alpha，让雾充满背景)
    links.new(mix.outputs['Image'], composite.inputs['Image'])

    print(f"Fog setup complete. Start: {start}, Depth: {depth}")