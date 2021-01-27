import datasets.utils as utils

SAMPLE_POINTS_COUNT = 2500

def get_surface_sampling_config(point_count=SAMPLE_POINTS_COUNT):
    config = utils.data_config.ThreeDConfig()

    config.sample_points = 'points'

    config.sample_settings.sample_points_count = point_count
    config.sample_settings.sample_even = True

    config.backend = 'kaolin'

    return config


def get_point_sampling_config(point_count=SAMPLE_POINTS_COUNT):
    TriangleConfig = utils.data_config.ThreeDConfig()

    TriangleConfig.sample_points = 'surface_points'

    TriangleConfig.sample_settings.sample_points_count = point_count
    TriangleConfig.sample_settings.sample_points_noise = 0
    TriangleConfig.sample_settings.sample_even = True
    TriangleConfig.sample_settings.sample_on_surface = True

    TriangleConfig.backend = 'trimesh'

    return TriangleConfig