from collections import namedtuple

from ..utils.data_config import ThreeDConfig
import numpy as np

from ..utils.kaolin_sampler.three_d_utils import ThreeDUtils as KaolinThreeDUtils, ThreeDPreprocessor as  KaolinThreeDPreprocessor
from ..utils.trimesh_sampler.three_d_utils import ThreeDUtils as  TrimeshThreeDUtils, ThreeDPreprocessor as TrimeshThreeDPreprocessor

Backend = namedtuple('Backend', ['preprocessor', 'utils'])

BACKENDS = {
    'trimesh' : Backend(TrimeshThreeDPreprocessor, TrimeshThreeDUtils),
    'kaolin' : Backend(KaolinThreeDPreprocessor, KaolinThreeDUtils)
}

class ThreeDData:
    def __init__(self, config: ThreeDConfig, points_count_for_sdf=10000):
        self.config = config

        self.backend = config.backend

        self.preprocessor = BACKENDS[self.backend].preprocessor()
        self.utils = BACKENDS[self.backend].utils()
        self.points_count_for_sdf = points_count_for_sdf

    def preprocess(self, mesh):
        preprocess_config = self.config.preprocess_config
        if preprocess_config.normalize:
            mesh = self.preprocessor.normalize_mesh(mesh)
        if preprocess_config.fix_mesh:
            mesh = self.preprocessor.fix_mesh(mesh)
        return mesh


    def sample_data(self, mesh):
        sample_config = self.config
        result_dict = {}
        if sample_config.sample_points:
            count = sample_config.sample_settings.sample_points_count
            noise = sample_config.sample_settings.sample_points_noise

            sample_on_surface = sample_config.sample_settings.sample_on_surface
            if sample_on_surface:
                points = self.utils.sample_surface_points(mesh, count, noise, sample_config.sample_settings.sample_even)
            else:


                sub_count = count // 6 + 1

                MULTIPLIER = 1.1

                # Generate around mesh
                points2 = np.random.uniform(-0.5 * MULTIPLIER, 0.5 * MULTIPLIER, [sub_count * 4, 3])

                #Sample on surface
                points3 = self.utils.sample_surface_points(mesh, sub_count, noise, sample_config.sample_settings.sample_even)
                points4 = self.utils.sample_surface_points(mesh, sub_count, 0,
                                                           sample_config.sample_settings.sample_even)

                #Merge
                choice =  np.random.choice(count, count, replace=False)
                points = np.concatenate([points2, points3, points4])[choice]






            if sample_config.sample_points:
                result_dict[sample_config.sample_points] = points
            if sample_config.calculate_sdf:
                if self.backend == 'kaolin':
                    result_dict[sample_config.calculate_sdf] = self.utils.calculate_sdf(mesh, points, self.points_count_for_sdf)
                else:
                    result_dict[sample_config.calculate_sdf] = self.utils.calculate_sdf(mesh, points)

            if any([sample_config.calculate_closed_triangle, sample_config.calculate_closed_triangle_center,
                    sample_config.calculate_point_on_closed_triangle]):
                closed_points, triangles, triangles_center = self.utils.find_closed_triangle(mesh, points)
                if sample_config.calculate_closed_triangle:
                    result_dict[sample_config.calculate_closed_triangle] = triangles
                if sample_config.calculate_closed_triangle_center:
                    result_dict[sample_config.calculate_closed_triangle_center] = triangles_center
                if sample_config.calculate_point_on_closed_triangle:
                    result_dict[sample_config.calculate_point_on_closed_triangle] = closed_points

            if sample_config.sample_raw_surface_points:
                result_dict[sample_config.sample_raw_surface_points] = self.utils.sample_surface_points(mesh, count, 0, sample_config.sample_settings.sample_even)
        return result_dict
