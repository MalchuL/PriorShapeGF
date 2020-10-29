import random
from typing import List, Dict

import numpy as np
import torch

from .augmentor import AugmentorCompose


class SeveralAugmentorCompose:
    """Compose augmentors."""

    def __init__(self, input_keys_map, augment_fn, output_key):
        """
        Args:
            key2augment_fn (Dict[str, Callable]): mapping from input key
                to augmentation function to apply
        """
        self.output_key = output_key
        self.augment_fn = augment_fn
        self.input_keys_map = input_keys_map


    def __call__(self, force_apply, **dictionary: dict) -> dict:
        """
        Args:
            dictionary (dict): item from dataset

        Returns:
            dict: dictionaty with augmented data
        """
        data = {key: dictionary[value] for key, value in self.input_keys_map.items()}
        dictionary[self.output_key] = self.augment_fn(**data)

        return dictionary


class RandomSigma(AugmentorCompose):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
            self,
            points_key='points',
            sigma_key='used_sigmas',
            sigma_begin=1,
            sigma_end=0.01,
            sigma_num=10

    ):
        super().__init__({points_key: self.add_sigmas})

        self.points_key = points_key
        self.sigma_key = sigma_key

        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        self.num_classes = sigma_num

        self.sigmas = np.exp(
            np.linspace(np.log(self.sigma_begin),
                        np.log(self.sigma_end),
                        self.num_classes)).astype(np.float32)

    def add_sigmas(self, input_dict):
        input_points = input_dict[self.points_key]
        # Randomly sample sigma https://github.com/RuojinCai/ShapeGF/blob/9311e54d40acbc67be22f1b41f82bdf356d1c1c4/trainers/ae_trainer_3D.py#L127
        if isinstance(input_points, np.ndarray):
            assert len(input_points.shape) == 2
            label = np.random.randint(
                0, len(self.sigmas))
            used_sigmas = np.array(self.sigmas)[label:label + 1]
        else:
            assert len(input_points.shape) == 3
            batch_size = input_points.shape[0]
            labels = torch.randint(
                0, len(self.sigmas), (batch_size,), device=input_points.device)
            used_sigmas = torch.tensor(
                np.array(self.sigmas))[labels].float().view(batch_size, 1).to(input_points.device)
        return {self.sigma_key: used_sigmas}




class AddSigmas(AugmentorCompose):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
            self,
            points_key='points',
            sigma_key='used_sigmas',
            sigma_begin=1,
            sigma_end=0.01,
            sigma_num=10,
            num_steps=10

    ):
        super().__init__({points_key: self.add_sigmas})

        self.points_key = points_key
        self.sigma_key = sigma_key

        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        self.num_classes = sigma_num
        self.num_steps = num_steps

        self.sigmas = np.repeat(np.exp(
            np.linspace(np.log(self.sigma_begin),
                        np.log(self.sigma_end),
                        self.num_classes)), self.num_steps).astype(np.float32)

    def add_sigmas(self, input_dict):
        input_points = input_dict[self.points_key]
        if isinstance(input_points, np.ndarray):
            assert len(input_points.shape) == 2
            used_sigmas = self.sigmas
        else:
            raise NotImplemented
        return {self.sigma_key: used_sigmas}




class RandomOffsetBySigma(SeveralAugmentorCompose):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
            self,
            points_key='points',
            sigma_key='used_sigmas',
            offset_points_key='offset_points'

    ):
        super().__init__({'points': points_key, 'sigma': sigma_key}, self.offset_by_sigmas, offset_points_key)



    def offset_by_sigmas(self, points, sigma):
        return points + (np.random.randn(*points.shape) * sigma).astype(np.float32)


class RandomOffsets(AugmentorCompose):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
            self,
            points_key='points',
            output_offset_points_key='offsets',
            sigma_num = 10,

    ):
        super().__init__({points_key: self.get_offsets})
        self.points_key = points_key
        self.output_offset_points_key = output_offset_points_key
        self.sigma_num = sigma_num

    def get_offsets(self, input_dict):
        input_points = input_dict[self.points_key]
        return {self.output_offset_points_key: np.random.randn(*input_points.shape, self.sigma_num).astype(np.float32)}