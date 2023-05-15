'''
  @ Date: 2022-04-24 16:09:24
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-08-31 11:48:52
  @ FilePath: /EasyMocapPublic/easymocap/blender/material.py
'''
import bpy

hex2rgb = lambda x:list(map(lambda v:int('0x'+v, 16), [x[:2], x[2:4], x[4:]]))

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
    if isinstance(pid, str):
        col = colors_table[pid]
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

def setMat_plastic(mesh, meshColor, AOStrength = 0.0, 
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

def set_material_i(mat, pid, metallic=0.5, specular=0.5, roughness=0.9, **kwargs):
    if isinstance(pid, int):
        color = get_rgb(pid)
    else:
        color = pid
    if False:
        build_pbr_nodes(mat.node_tree, base_color=color, 
            metallic=metallic, specular=specular, roughness=roughness, **kwargs)
    else:
        setMat_plastic(mat, colorObj(color, B=0.3))