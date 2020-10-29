"""
Specify name for each None parameter to get this attribute from result dict
"""


class ThreeDConfig:

    class SampleSettings:
        sample_points_count = 0
        sample_points_noise = 0
        sample_even = True
        sample_on_surface = True

    class PreprocessConfig:
        fix_mesh = True
        normalize = True

    def __init__(self):

        self.backend = None

        self.sample_settings = ThreeDConfig.SampleSettings()
        self.preprocess_config = ThreeDConfig.PreprocessConfig()

        self.sample_points = None
        self.calculate_sdf = None

        self.calculate_closed_triangle = None
        self.calculate_closed_triangle_center = None
        self.calculate_point_on_closed_triangle = None

        #additional params, this data has no calculated parameters
        self.sample_raw_surface_points = None

    def get_data(self):
        return self




