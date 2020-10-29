import itertools
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import torchvision
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets import utils
from datasets.SortedItemsDatasetWrapper import SortedItemsDatasetWrapper
from datasets.preprocessed_dataset import PreprocessedPickleDataset
from datasets.transformed_dataset import TransformedDataset
from losses.losses import define_loss, define_loss_from_params
from models import networks

import torch
import time
import numpy as np

from models.networks import define_encoder_from_params, define_decoder_from_params
from registry import registry, registries
from utils.point_cloud_utils import get_offsets, get_sigmas, get_prior


class ThreeDExperiment(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(ThreeDExperiment, self).__init__()

        self.hparams = hparams

        self.sigma_begin = float(self.hparams.trainer.sigma_begin)
        self.sigma_end = float(self.hparams.trainer.sigma_end)
        self.num_classes = int(self.hparams.trainer.sigma_num)
        self.sigmas = np.exp(
            np.linspace(np.log(self.sigma_begin),
                        np.log(self.sigma_end),
                        self.num_classes, dtype=np.float32))

        self.create_model()

    def get_scheduler(self, optimizer):

        args = {**self.hparams.scheduler_params}
        args['optimizer'] = optimizer

        return registries.SCHEDULERS.get_from_params(**args)

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def create_model(self):
        self.encoder = define_encoder_from_params(self.hparams.encoder_params)
        self.decoder = define_decoder_from_params(self.hparams.decoder_params)

    @staticmethod
    def get_nearest_point_sampling_config(point_count):
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

    def get_transforms(self, name):
        return None

    def prepare_data(self):
        config = self.get_nearest_point_sampling_config(self.hparams.points_count)
        use_embedding = False
        self.train_dataset = SortedItemsDatasetWrapper(items=['surface_points'], dataset=TransformedDataset(
            PreprocessedPickleDataset(self.hparams.data_params.train_dataset_path, config, use_embedding=use_embedding),
            self.get_transforms('train')))
        self.val_dataset = SortedItemsDatasetWrapper(items=['surface_points'], dataset=TransformedDataset(
            PreprocessedPickleDataset(self.hparams.data_params.val_dataset_path, config, use_embedding=use_embedding),
            self.get_transforms('valid')))

    def forward(self, perturbed_points, z, sigma):
        bs, num_pts = perturbed_points.size(0), perturbed_points.size(1)
        sigma = sigma.view(bs, 1)
        shape_latent = torch.cat((z, sigma), dim=1)
        y_pred = self.decoder(perturbed_points, shape_latent)

        return y_pred

    def reconstruction_loss(self, input_cloud, perturbed_points, y_pred, sigmas):
        bs, num_pts = input_cloud.size(0), input_cloud.size(1)
        y_gtr = - (perturbed_points - input_cloud).view(bs, num_pts, -1)

        # The loss for each sigma is weighted
        lambda_sigma = 1. / sigmas.view(-1, 1, 1)
        loss = 0.5 * ((y_gtr - y_pred) ** 2. * lambda_sigma).sum(dim=2).mean()
        return loss

    def training_step_end(self, outputs):
        out = outputs['out']
        batch_idx = 0
        if self.global_step % self.hparams.log_point_cloud == 0:
            points_gt = out['input_point_cloud'][batch_idx]
            self.log_point_cloud('input_point_cloud', points_gt.view(1, -1, 3))
            points_pred = out['perturbed_points'][batch_idx] + out['y_pred'][batch_idx] * out['sigmas'][batch_idx].view(
                -1, 1, 1)
            self.log_point_cloud('predicted_point_cloud', points_pred.view(1, -1, 3))

        return outputs

    def log_point_cloud(self, name, point_cloud):
        self.logger.experiment.add_mesh(name, point_cloud, global_step=self.global_step)

    def training_step(self, batch, batch_idx):

        input = batch[0]

        offsets = torch.from_numpy(get_offsets(input)).to(input.device)
        sigmas = torch.from_numpy(get_sigmas(self.sigmas, input.shape[0])).to(input.device)

        z_mu, z_sigma = self.encoder(input)
        z = z_mu + 0 * z_sigma

        perturbed_points = input + offsets * sigmas.view(-1, 1, 1)
        y_pred = self.forward(perturbed_points, z, sigmas)

        loss = self.reconstruction_loss(input, perturbed_points, y_pred, sigmas)

        out = {'input_point_cloud': input,
               'y_pred': y_pred,
               'perturbed_points': perturbed_points,
               'sigmas': sigmas
               }

        return {'loss': loss, 'out': out}

    def validation_step(self, batch, batch_nb):
        if batch_nb == 0:
            input = batch[0]
            z, _ = self.encoder(input)
            pred = self.langevin_dynamics(z, self.hparams.points_count, self.hparams.step_size_ratio,
                                          self.hparams.num_steps,
                                          self.hparams.weight)
            self.log_point_cloud('input_point_cloud', pred[0].view(1, -1, 3))
        return {}

    def langevin_dynamics(self, z, num_points=2048, step_size_ratio=1, num_steps=5, weight=1):
        with torch.no_grad():
            sigmas = self.sigmas
            x_list = []
            self.decoder.eval()
            x = get_prior(z.size(0), num_points, self.hparams.decoder_params.dim)
            x = x.to(z)
            x_list.append(x.clone())
            for sigma in sigmas:
                sigma = torch.ones((1,)).cuda() * sigma
                z_sigma = torch.cat((z, sigma.expand(z.size(0), 1)), dim=1)
                step_size = 2 * sigma ** 2 * step_size_ratio
                for t in range(num_steps):
                    z_t = torch.randn_like(x) * weight
                    x += torch.sqrt(step_size) * z_t
                    grad = self.decoder(x, z_sigma)
                    grad = grad / sigma ** 2
                    x += 0.5 * step_size * grad

        return x

    def validation_epoch_end(self, outputs):
        return {'val_loss': -self.current_epoch}

    def configure_optimizers(self):
        params = itertools.chain(self.encoder.parameters(), self.decoder.parameters())
        optimizer = registries.OPTIMIZERS.get_from_params(**{'params': params, **self.hparams.optimizer_params})

        return [optimizer], [self.get_scheduler(optimizer)]

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.hparams.data_params.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.data_params.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.hparams.data_params.num_workers)
