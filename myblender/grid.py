import numpy as np
from .color import color_jet
from .geometry import create_points

def plot_grids(grids, confs, radius, res, MIN_THRES=0.15, gamma=0.5):
    print(confs.max(), confs.min())
    input()
    confs = np.clip(confs, 0, 1)
    confs = confs / confs.max()
    confs = np.power(confs, gamma)
    # confs[confs<MIN_THRES] = 0
    print((confs>MIN_THRES).sum(), confs[confs>MIN_THRES].mean(), confs.max(), len(confs))
    for i, (grid, conf) in enumerate(zip(grids, confs)):
        if conf < MIN_THRES:
            conf = 0
        print('>>> Loading {}/{}'.format(i, len(grids)))
        conf_int = int(conf * 255)
        create_points(vid=color_jet[conf_int], center=grid, radius=radius, 
            alpha=np.power(conf, 0.5),
            basename='sphere_{}.obj'.format(res), shadow=False)