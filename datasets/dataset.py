
import os
import kaolin as kal
import trimesh

from .utils import as_mesh
from .utils.three_d_sampler import ThreeDData
from torch.utils.data import Dataset

def get_folders(root_dir):

    indexes = []
    def _get_folders(dir):
        current_dir = map(lambda x: os.path.join(dir, x), os.listdir(dir))
        return tuple(filter(lambda x: os.path.isdir(x), current_dir))

    folders = _get_folders(root_dir)
    for folder in folders:
        inner_fodlers = _get_folders(folder)
        inner_fodlers = list(
            map(lambda path: os.path.relpath(path, start=root_dir), inner_fodlers))  # exclude root folder
        indexes.extend(inner_fodlers)

    return indexes

class ShapeNetV2Dataset(Dataset):
    """ShapeNet dataset."""

    PATH_TO_FILE = r"models"
    PATH_KEY = 'path'
    ID_KEY = 'obj_id'

    def __init__(self, root_dir, mesh_sampler_config=None, target_extension="obj", max_MB_file_size=None, to_cuda=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_transform, mesh_transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.to_cuda = to_cuda
        self.root_dir = root_dir
        self.target_extension = target_extension

        self.indexes = get_folders(root_dir)


        self.mesh_sampler_config = mesh_sampler_config
        self.mesh_sampler = ThreeDData(self.mesh_sampler_config)

        self.max_size = max_MB_file_size * 1024 * 1024 if max_MB_file_size else None

    def _get_folder_by_id(self, id):
        return self.indexes[id]

    def __len__(self):
        return len(self.indexes)



    def get_mesh_data(self, path):

        if self.mesh_sampler_config.backend == 'kaolin':
            mesh = kal.rep.TriangleMesh.from_obj(path)
            if self.to_cuda:
                mesh.cuda()
        else:
            mesh = as_mesh(trimesh.load(path))
        mesh = self.mesh_sampler.preprocess(mesh)
        mesh_data = self.mesh_sampler.sample_data(mesh)

        return mesh_data




    def __getitem__(self, idx):
        dir = os.path.join(self.root_dir, self._get_folder_by_id(idx))
        file = tuple(filter(lambda x: x.endswith("." + self.target_extension),
                            os.listdir(os.path.join(dir, self.PATH_TO_FILE))))[0]
        full_path = os.path.join(dir, self.PATH_TO_FILE, file)

        try:
            self._assert_size(full_path)
        except:
            return None

        result = {}
        if self.mesh_sampler_config:
            mesh_data = self.get_mesh_data(full_path)
            result.update(mesh_data)

        self._check_key(result, self.PATH_KEY)
        self._check_key(result, self.ID_KEY)
        result[self.PATH_KEY] = self._get_folder_by_id(idx)
        result[self.ID_KEY] = idx
        return result


    def _assert_size(self, path):
        if self.max_size:
            size = os.path.getsize(path)
            if size > self.max_size:
                raise FileExistsError(f"File must be smaller than {self.max_size} got {size}")


    @staticmethod
    def _check_key(check_dict, new_key_or_dict):
        if isinstance(new_key_or_dict, (list, tuple, dict)):
            for key in new_key_or_dict:
                assert key not in check_dict
        else:
            assert new_key_or_dict not in check_dict

