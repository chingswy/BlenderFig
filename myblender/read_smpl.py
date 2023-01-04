'''
  @ Date: 2022-09-20 20:55:05
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-21 12:03:47
  @ FilePath: /EasyMocapPublic/easymocap/blender/read_smpl.py
'''
import os
import json
import numpy as np
from os.path import join

def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data
    
def read_smpl(outname):
    assert os.path.exists(outname), outname
    datas = read_json(outname)
    outputs = []
    if isinstance(datas, dict):
        datas = datas['annots']
    for data in datas:
        for key in ['Rh', 'Th', 'poses', 'shapes']:
            data[key] = np.array(data[key])
        outputs.append(data)
    return outputs

def merge_params(param_list, share_shape=True):
    output = {}
    for key in ['poses', 'shapes', 'Rh', 'Th', 'expression']:
        if key in param_list[0].keys():
            output[key] = np.vstack([v[key] for v in param_list])
    if share_shape:
        output['shapes'] = output['shapes'].mean(axis=0, keepdims=True)
    return output
    
def load_smpl_params(path):
    from glob import glob
    filenames = sorted(glob(join(path, '*.json')))
    motions = {}
    for filename in filenames:
        infos = read_smpl(filename)
        for data in infos:
            pid = data['id']
            if pid not in motions.keys():
                motions[pid] = []
            motions[pid].append(data)
    keys = list(motions.keys())
    # BUG: not strictly equal: (Rh, Th, poses) != (Th, (Rh, poses))
    for pid in motions.keys():
        motions[pid] = merge_params(motions[pid])
        if motions[pid]['poses'].shape[1] == 69:
            motions[pid]['poses'] = np.hstack([np.zeros((motions[pid]['poses'].shape[0], 3)), motions[pid]['poses']])
        # motions[pid]['poses'][:, :3] = motions[pid]['Rh']
        # motions[pid]['poses'][:, :3] = 0
    return motions