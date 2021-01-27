import utils

SAMPLE_POINTS_COUNT = 2500

def get_surface_sampling_config(point_count=SAMPLE_POINTS_COUNT):
    config = utils.data_config.ThreeDConfig()

    config.sample_points = 'points'

    config.sample_settings.sample_points_count = point_count
    config.sample_settings.sample_even = True

    config.backend = 'kaolin'

    return config