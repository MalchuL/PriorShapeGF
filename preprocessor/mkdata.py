import os
import pickle
from pathlib import Path

BASE_PATH = '/home/malchul/work/projects/3d_reconstruction/BSP_Net/data/shape_net_nearest_points_in_square'

COPY_PATH = '/home/malchul/work/projects/3d_reconstruction/NearestNet/repo/data/shapenet_sampled'

#os.mkdir(COPY_PATH)


split_file = 'split.pkl'
try:
    with open(os.path.join(opt.outf, split_file), 'rb') as f:
        splits = pickle.load(f)
    train_indexes = splits['train']
    test_indexes = splits['test']
    print('load split', len(train_indexes), len(test_indexes))
except:
    indexes = get_folders(opt.dataset_path)
    # random.seed = 42
    np.random.seed(opt.manualSeed)

    split_percentage = 0.9
    K = int(split_percentage * len(indexes))

    # np.random.choice()
    train_indexes = list(np.random.choice(indexes, size=K, replace=False))
    test_indexes = [path for path in indexes if
                    path not in train_indexes]
    splits = {}
    splits['train'] = train_indexes
    splits['test'] = test_indexes
    with open(os.path.join(opt.outf, split_file), 'wb') as f:
        splits = pickle.dump(splits, f)
    print('dump split', len(train_indexes), len(test_indexes))

    print(len(train_indexes), len(test_indexes), len(indexes), len(train_indexes + test_indexes))

with open('/home/malchul/work/projects/3d_reconstruction/NearestNet/repo/utils/split.pkl','rb') as f:
    paths = pickle.load(f)
    for mode in paths:
        for path in paths[mode]:
            copy_path = Path(os.path.join(COPY_PATH,mode,path))
            copy_path.parent.mkdir(parents=True,exist_ok=True)
            os.symlink(os.path.join(BASE_PATH,path),os.path.join(COPY_PATH,mode,path))