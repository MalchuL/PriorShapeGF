import os
import pickle
from pathlib import Path

from torch.utils.data import Dataset

import random
import numpy as np

class SampledShapeNetV2Dataset(Dataset):
    """ShapeNet dataset."""

    PATTERN = '*.npy'

    def __init__(self, mesh_sampler_config, root_dir, split, is_single_class=False, use_all_points=False):
        self.root_dir = root_dir
        self.is_single_class = is_single_class

        assert split in ['test', 'train', 'val']
        self.split = split

        self.indexes = self.get_folders(self.root_dir)
        self.mesh_sampler_config = mesh_sampler_config
        self.use_all_points = use_all_points

    def get_folders(self, root_dir):
        files = Path(root_dir).rglob(self.split + '/' + self.PATTERN)
        files = list(map(str, files))
        return files



    def __len__(self):
        return len(self.indexes)


    def __getitem__(self, idx):

        result = {}

        # Data in [Nx3] N == 15000
        data = np.load(self.indexes[idx])

        if self.use_all_points:
            result[self.mesh_sampler_config.sample_points] = data.astype(np.float32)
        else:
            sample_count = self.mesh_sampler_config.sample_settings.sample_points_count
            choises = np.random.choice(data.shape[0], size=sample_count, replace=False)
            result[self.mesh_sampler_config.sample_points] = data[choises].astype(np.float32)


        return result

