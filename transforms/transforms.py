import torch
from .augmentor import Augmentor, AugmentorCompose, AugmentorKeys
import numpy as np



class CopyProperty():
    def __init__(self, copy_from, copy_to):
        self.copy_to = copy_to
        self.copy_from = copy_from

    def __call__(self, force_apply, **dict_:dict):
        dict_[self.copy_to] = dict_[self.copy_from].copy()
        return dict_

class ConvertNHCtoNCH(Augmentor):
    @staticmethod
    def _convert_NHC_to_NCH(x):
        return x.transpose(1,0)

    def __init__(self, dict_key):
        super().__init__(dict_key, self._convert_NHC_to_NCH, 'x', None)


class ScalePointCloud(Augmentor):

    def _scale(self, x):
        return x * self.scale_value

    def __init__(self, dict_key, scale_value):
        super().__init__(dict_key, self._scale, 'x', None)
        self.scale_value = scale_value


class Converter8BitFrom01(Augmentor):

    def convert01to256(self, x):
        return (((x - self.min) / (self.max - self.min) * self.size).round()) / self.size + self.min

    def __init__(self, dict_key, min, max, size=256):
        super().__init__(dict_key, self.convert01to256, 'x', None)
        self.max = max
        self.min = min
        self.size = size


class Converter01From8Bit(Augmentor):

    def __init__(self, dict_key, min, max):
        self.converter = InverseConverter8Bit(min, max)
        super().__init__(dict_key, self.converter.convert256to01, 'x', None)



class InverseConverter8Bit():

    def convert256to01(self, x):
        result =  (x / self.size) * (self.max - self.min) + self.min
        if isinstance(result, np.ndarray):
            result = result.astype(np.float32)
        else:
            result = result.to(torch.float32)
        return result


    def __call__(self, x):
        return self.convert256to01(x)

    def __init__(self, min, max, size=256):
        self.max = max
        self.min = min
        self.size = size
