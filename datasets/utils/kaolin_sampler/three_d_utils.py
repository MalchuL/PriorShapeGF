import kaolin as kal
import numpy as np
import torch


class ThreeDUtils:

    @staticmethod
    def sample_surface_points(mesh, count, noise, sample_even=False):
        if sample_even:
            points, _ = kal.conversions.trianglemesh_to_pointcloud(mesh, count)
        else:
            raise NotImplemented()
        sampled = points
        sampled = sampled + torch.from_numpy(np.random.uniform(-noise, noise, sampled.shape)).to(dtype=sampled.dtype, device=sampled.device)
        return sampled

    @staticmethod
    def find_closed_triangle(mesh, points):
        raise NotImplemented()


    @staticmethod
    def calculate_sdf(mesh, points, sampled_points_for_distance=10000):
        sdf = kal.conversions.trianglemesh_to_sdf(mesh, sampled_points_for_distance)

        sdfs = sdf(points)
        return -sdfs #because calculation with negative sign


class ThreeDPreprocessor:
    def __init__(self):
        self.normalizer = kal.transforms.NormalizeMesh(True)


    @staticmethod
    def fix_mesh(mesh):

        #TODO this feature is not supported in kaolin
        #mesh.remove_duplicate_faces()
        #repair.fix_normals(mesh, True)
        #repair.fix_winding(mesh)
        return mesh

    def normalize_mesh(self, mesh):
        bbox = [mesh.vertices.min(0)[0], mesh.vertices.max(0)[0]]
        center = (bbox[0] + bbox[1]) / 2
        scale = 1 / (np.sqrt(np.sum((bbox[1] - bbox[0]).cpu().numpy() ** 2)))
        return kal.transforms.Compose([kal.transforms.TranslateMesh(-center), kal.transforms.ScaleMesh(scale)])(mesh)