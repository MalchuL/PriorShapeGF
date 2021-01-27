import os
import pickle
from pathlib import Path
from .dataset import ShapeNetV2Dataset
import random
import numpy as np

class PreprocessedPickleDataset(ShapeNetV2Dataset):
    """ShapeNet dataset."""

    PATH_TO_FILE = r"../models"
    PATH_KEY = 'path'
    ID_KEY = 'obj_id'

    PICKLE_PATTERN = 'calculated_data_*.pkl'

    def __init__(self, root_dir, mesh_sampler_config, use_embedding, subsample_partirions_count=3, pickle_pattern=PICKLE_PATTERN):
        super().__init__(root_dir, mesh_sampler_config)
        self.use_embedding = use_embedding
        self.pickle_pattern = pickle_pattern
        self.subsample_partirions_count = subsample_partirions_count

        self.sort_elements_by_distance ={'key': None,'value': None}# {'key':'triangle','value':'closed_point'} #
        self.remove_nans = True


    def __getitem__(self, idx):
        dir = os.path.join(self.root_dir, self._get_folder_by_id(idx))

        files = list(Path(dir).glob(self.pickle_pattern))
        dumped_data = []

        if self.subsample_partirions_count:
            files = random.choices(files, k=self.subsample_partirions_count)

        for file in files:
            with open(file, 'rb') as f:
                try:
                    dumped_data.append(pickle.load(f))
                except Exception as e:
                    print(e)

        result = {}
        for data in dumped_data:
            for key in data:
                if key == self.ID_KEY:
                    continue

                if isinstance(data[key], np.ndarray):
                    result[key] = np.concatenate([result.get(key, np.zeros([0,*data[key].shape[1:]])), data[key]],axis=0)

                elif isinstance(data[key], (tuple,list)):
                    result[key] = result.get(key, []).extend(data[key])
                elif isinstance(data[key], (int, float)):
                    result[key] = np.array([data[key] for _ in range(len(result[list(result.keys())[0]]))])
                elif isinstance(data[key], str):
                    pass
                else:
                    raise TypeError(f'Unsupported type {data}: {type(data[key])}')

        try:
            size = len(result[list(result.keys())[0]])
        except:
            print("can't load item", idx)
            return None
        sample_count = self.mesh_sampler_config.sample_settings.sample_points_count
        assert size > sample_count, f'{size} != {sample_count}'


        for key in result:
            assert len(result[key]) == size and size > 0, f'{len(result[key])} != {size}, {key}'

        choises_list = list(range(size))
        #Remove nans
        if self.remove_nans:

            excluded = set()
            for key in result:
                excluded = excluded.union(set(np.argwhere(np.isnan(result[key]))[:, 0]))
            for el in excluded:
                choises_list.remove(el)

        choises = random.choices(choises_list, k=sample_count)
        for key in result:
            result[key] = result[key][choises].astype(np.float32)

        if self.sort_elements_by_distance['key'] in result:
            if not self.remove_nans:
                for i in range(len(choises)):
                    point = result[self.sort_elements_by_distance['value']][i]
                    triangle = result[self.sort_elements_by_distance['key']][i]
                    sorted_points = np.array(sorted(triangle, key=lambda triangle_point: sum((point - triangle_point)**2)))
                    result[self.sort_elements_by_distance['key']][i] = sorted_points
            else:
                raise NotImplemented('not removed nans feature is not implemented')

        if self.use_embedding:
            self._check_key(result, self.ID_KEY)
            result[self.ID_KEY] = idx
        else:
            self._check_key(result, self.ID_KEY)
            result[self.ID_KEY] = -1

        return result

