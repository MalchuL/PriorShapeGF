import os
import pickle
from pathlib import Path

import cv2
from torch.utils.data import Dataset

import random
import numpy as np

import albumentations as A
from albumentations.augmentations.functional import center_crop
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import Lambda
import torchvision.transforms as transforms

def get_transform(opt, isTrain, use_blur=False):
    very_rare_prob = 0.05
    rare_prob = 0.1
    medium_prob = 0.2
    normal_prob = 0.3
    often_prob = 0.6
    compression_prob = 0.35

    transform_list = []
    if isTrain:
        pre_process = [
            A.ShiftScaleRotate(shift_limit=0.001, rotate_limit=20, scale_limit=0.3, interpolation=cv2.INTER_CUBIC,
                               p=normal_prob),
            A.SmallestMaxSize(opt.loadSize, always_apply=True, interpolation=cv2.INTER_CUBIC),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(opt.fineSize, opt.fineSize, always_apply=True)]
    else:
        pre_process = [
            A.SmallestMaxSize(opt.loadSize, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomCrop(opt.fineSize, opt.fineSize, always_apply=True)]

    if use_blur:
        pre_process += [A.GaussianBlur(blur_limit=5, always_apply=True)]


    strong = [


        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=normal_prob),
            A.MotionBlur(p=rare_prob),
            A.Downscale(scale_min=0.6, scale_max=0.8, interpolation=cv2.INTER_CUBIC, p=rare_prob),
        ], p=normal_prob),

        A.OneOf([
            A.ToGray(p=often_prob),
            A.ToSepia(p=very_rare_prob)
        ], p=very_rare_prob),

        A.OneOf([
            A.ImageCompression(quality_lower=39, quality_upper=60, p=compression_prob),

            A.MultiplicativeNoise(multiplier=[0.92, 1.08], elementwise=True, per_channel=True, p=compression_prob),
            A.ISONoise(p=compression_prob)
        ], p=compression_prob),
        A.OneOf([
            A.CLAHE(p=normal_prob),
            A.Equalize(by_channels=False, p=normal_prob),
            A.RGBShift(p=normal_prob),
            A.RandomBrightnessContrast(p=normal_prob),
            # A.RandomShadow(p=very_rare_prob, num_shadows_lower=1, num_shadows_upper=1,
            #               shadow_dimension=5, shadow_roi=(0, 0, 1, 0.5)),
            A.RandomGamma(p=normal_prob),
        ]),
        A.HueSaturationValue(p=normal_prob, sat_shift_limit=(-50, 40)),


    ]

    post_process = [A.Normalize((0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5)),
        ToTensorV2()]


    if isTrain:
        composed = pre_process + strong + post_process
    else:
        composed = pre_process + post_process
        strong = []
    composed = A.Compose(composed, p=1)
    alb_transform = [Lambda(lambda x: composed(image=x)['image'])]
    return transforms.Compose(alb_transform)

class SampledShapeNetV2WithImageDataset(Dataset):
    """ShapeNet dataset."""

    PATTERN = '*.npy'
    IMAGE_PATTERN = '*.png'

    ALL_POINTS = 'all_points'

    def __init__(self, mesh_sampler_config, root_dir, split, image_folder, image_transform=None, is_single_class=False, use_all_points=False):
        self.root_dir = root_dir
        self.is_single_class = is_single_class

        assert split in ['test', 'train', 'val']
        self.split = split

        self.indexes = self.get_folders(self.root_dir)
        self.mesh_sampler_config = mesh_sampler_config
        self.use_all_points = use_all_points
        self.image_transform = image_transform
        self.image_folder = image_folder

    def get_folders(self, root_dir):
        files = Path(root_dir).rglob(self.split + '/' + self.PATTERN)
        files = list(map(str, files))
        return files



    def __len__(self):
        return len(self.indexes)

    def normalize_points_1_1(self, point_cloud):
        center = 0.5 * (point_cloud.max(axis=0) + point_cloud.min(axis=0))
        scale = (point_cloud.max(axis=0) - point_cloud.min(axis=0)).max()
        norm_vert = (point_cloud - center) / scale
        return norm_vert, center, scale

    def normalize_points_radius_1(self, point_cloud):
        center = 0.5 * (point_cloud.max(axis=0) + point_cloud.min(axis=0))
        scale = np.sqrt(((point_cloud.max(axis=0) - point_cloud.min(axis=0)) ** 2).sum()) * 0.5
        norm_vert = (point_cloud - center) / scale
        return norm_vert, center, scale

    def __getitem__(self, idx):

        result = {}

        # Get image
        folder = str(Path(self.indexes[idx]).stem)

        images_folder = os.path.join(self.image_folder, folder)
        #print('img_folder',images_folder)
        images_in_folder = list(Path(images_folder).rglob(self.IMAGE_PATTERN))
        image = str(random.choice(images_in_folder))

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_transform is not None:
            image = self.image_transform(image)
        result['image'] = image


        # Data in [Nx3] N == 15000
        data = np.load(self.indexes[idx])
        data, center, scale = self.normalize_points_1_1(data)
        result['center'] = center.astype(np.float32)
        result['scale'] = scale.astype(np.float32)

        sample_count = self.mesh_sampler_config.sample_settings.sample_points_count
        choises = np.random.choice(data.shape[0], size=sample_count, replace=False)
        result[self.mesh_sampler_config.sample_points] = data[choises].astype(np.float32)


        if self.use_all_points:
            result[self.mesh_sampler_config.all_points] = data.astype(np.float32)
        else:
            all_samples_choises = np.random.choice(data.shape[0], size=self.mesh_sampler_config.all_points_count, replace=False)
            result[self.mesh_sampler_config.all_points] = data[all_samples_choises].astype(np.float32)

        return result

