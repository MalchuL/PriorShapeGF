import os
import pickle
from pathlib import Path

from torch.utils.data import Dataset

import random
import numpy as np

class SampledShapeNetV2Dataset(Dataset):
    """ShapeNet dataset."""

    PATTERN = '*.npy'

    ALL_POINTS = 'all_points'

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

    def normalize_points_1_1(self, point_cloud):
        center = 0.5 * (point_cloud.max(axis=0) + point_cloud.min(axis=0))
        scale = (point_cloud.max(axis=0) - point_cloud.min(axis=0)).max() * 0.5
        norm_vert = (point_cloud - center) / scale
        return norm_vert

    def normalize_points_radius_1(self, point_cloud):
        center = 0.5 * (point_cloud.max(axis=0) + point_cloud.min(axis=0))
        scale = np.sqrt(((point_cloud.max(axis=0) - point_cloud.min(axis=0)) ** 2).sum()) * 0.5
        norm_vert = (point_cloud - center) / scale
        return norm_vert

    def __getitem__(self, idx):

        result = {}

        # Data in [Nx3] N == 15000
        data = np.load(self.indexes[idx])
        #data = self.normalize_points_radius_1(data)

        sample_count = self.mesh_sampler_config.sample_settings.sample_points_count
        choises = np.random.choice(data.shape[0], size=sample_count, replace=False)
        result[self.mesh_sampler_config.sample_points] = data[choises].astype(np.float32)


        if self.use_all_points:
            result[self.mesh_sampler_config.all_points] = data.astype(np.float32)
        else:
            all_samples_choises = np.random.choice(data.shape[0], size=self.mesh_sampler_config.all_points_count, replace=False)
            result[self.mesh_sampler_config.all_points] = data[all_samples_choises].astype(np.float32)

        return result

