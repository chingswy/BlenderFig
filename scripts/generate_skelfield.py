import os
from os.path import join
import numpy as np
import cv2
from easymocap.mytools.file_utils import read_json, save_json

def bound_from_keypoint(keypoint, padding=0.1, min_z=0):
    v = keypoint[..., -1]
    k3d_flat = keypoint[v>0.01]
    lower = k3d_flat[:, :3].min(axis=0) - padding
    lower[2] = max(min_z, lower[2])
    upper = k3d_flat[:, :3].max(axis=0) + padding
    center = (lower + upper ) / 2
    scale = (upper - lower)/2
    return center, scale, np.stack([lower, upper])

def caculate_distance(xyz, gt, sigma):
    # xyz: (-1, 3)
    # gt: (N, J, 3)
    distance = np.linalg.norm(xyz[:, None, None] - gt[None, ..., :3], axis=-1)
    distance = distance.min(axis=1)
    conf = np.exp(-distance ** 2 / sigma ** 2)
    return distance, conf


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--offset', type=float, default=0.1)
    args = parser.parse_args()

    record = read_json(args.path)
    keypoints = np.array(record['gt'])
    # center = keypoints[..., :3].mean(axis=1).mean(axis=0)
    center, scale, bounds = bound_from_keypoint(keypoints, min_z=0, padding=0.5)
    step = 0.005
    z_set = center[1] + args.offset
    w_range = np.arange(bounds[0, 0], bounds[1, 0], step)
    h_range = np.arange(bounds[0, 2], bounds[1, 2], step)
    w, h = np.meshgrid(w_range, h_range)
    xyz = np.stack([w, np.zeros_like(w) + z_set, h], axis=-1)
    xyz_flat = xyz.reshape(-1, 3)
    distance, conf = caculate_distance(xyz_flat, keypoints, sigma=0.2)
    print(distance.min(), distance.max(), conf.max())
    conf_zero = 1 - conf.max(axis=1, keepdims=True)
    conf = np.hstack([conf_zero, conf])
    conf = conf / (1e-5 + conf.sum(axis=1, keepdims=True))
    print(conf.min(), conf.max())
    color_bar = cv2.applyColorMap((16*np.arange(16).reshape(1, -1)).astype(np.uint8), cv2.COLORMAP_JET)
    color_bar_float = color_bar[0].astype(np.float32) / 255
    color_bar_float[0] = 1
    conf_color = ((conf @ color_bar_float)*255).astype(np.uint8)
    conf_color = conf_color.reshape(*xyz.shape[:2], 3)
    outname = join(args.out + '_texture.jpg')
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    cv2.imwrite(outname, conf_color)
    records = {
        'keypoints3d': keypoints.tolist(),
        'texture': outname,
        'center': center.tolist(),
        'scale': scale.tolist(),
        # 'corners': [
        #     [w_range[0], z_set, h_range[0]],
        #     [w_range[-1], z_set, h_range[0]],
        #     [w_range[-1], z_set, h_range[-1]],
        #     [w_range[0], z_set, h_range[-1]],
        # ],
        'corners': [
            [w_range[-1], z_set, h_range[0]],
            [w_range[-1], z_set, h_range[-1]],
            [w_range[0], z_set, h_range[-1]],
            [w_range[0], z_set, h_range[0]],
        ]
    }
    save_json(join(args.out + '_records.json', records)