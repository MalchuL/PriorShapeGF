import os
import pickle
from pathlib import Path

BASE_PATH = '/home/malchul/work/projects/3d_reconstruction/BSP_Net/data/shape_net_nearest_points_in_square'

COPY_PATH = '/home/malchul/work/projects/3d_reconstruction/CustomShapeGF/data/shapenet_sampled'

#os.mkdir(COPY_PATH)

with open('/home/malchul/work/projects/3d_reconstruction/CustomShapeGF/data/split.pkl','rb') as f:
    paths = pickle.load(f)
    for mode in paths:
        for path in paths[mode]:
            copy_path = Path(os.path.join(COPY_PATH,mode,path))
            copy_path.parent.mkdir(parents=True,exist_ok=True)
            os.symlink(os.path.join(BASE_PATH,path),os.path.join(COPY_PATH,mode,path))