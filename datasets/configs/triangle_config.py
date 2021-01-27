import datasets.utils as utils

SAMPLE_POINTS_COUNT = 2500

def get_triangle_samplit_config(point_count=SAMPLE_POINTS_COUNT):
    TriangleConfig = utils.data_config.ThreeDConfig()

    TriangleConfig.calculate_closed_triangle = 'triangle'
    TriangleConfig.calculate_closed_triangle_center = 'triangle_center'
    TriangleConfig.calculate_point_on_closed_triangle = 'closed_point'
    TriangleConfig.sample_points = 'points'

    TriangleConfig.sample_raw_surface_points = 'surface_points'

    TriangleConfig.sample_settings.sample_points_count = point_count
    TriangleConfig.sample_settings.sample_points_noise = 0.01
    TriangleConfig.sample_settings.sample_even = True
    TriangleConfig.sample_settings.sample_on_surface = False

    TriangleConfig.backend = 'trimesh'

    return TriangleConfig

def get_nearest_point_sampling_config(point_count=SAMPLE_POINTS_COUNT):
    TriangleConfig = utils.data_config.ThreeDConfig()


    TriangleConfig.calculate_point_on_closed_triangle = 'closed_point'
    TriangleConfig.sample_points = 'points'

    TriangleConfig.sample_raw_surface_points = 'surface_points'

    TriangleConfig.sample_settings.sample_points_count = point_count
    TriangleConfig.sample_settings.sample_points_noise = 0.01
    TriangleConfig.sample_settings.sample_even = True
    TriangleConfig.sample_settings.sample_on_surface = False

    TriangleConfig.backend = 'trimesh'

    return TriangleConfig

