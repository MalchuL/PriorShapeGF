import itertools
from argparse import Namespace
from pathlib import Path

import pytorch_lightning as pl
import torchvision
from kaolin.metrics import SidedDistance
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn as nn

from datasets import utils
from datasets.SortedItemsDatasetWrapper import SortedItemsDatasetWrapper
from datasets.preprocessed_dataset import PreprocessedPickleDataset
from datasets.sampled_shapenetv2 import SampledShapeNetV2Dataset
from datasets.transformed_dataset import TransformedDataset
from losses import ChamferDistance, MaxChamferDistance
from losses.losses import define_loss, define_loss_from_params
from losses.point_loss import MaxDistance
from models import networks

import torch
import time
import numpy as np

from models.networks import define_encoder_from_params, define_decoder_from_params
from registry import registry, registries
from utils.point_cloud_utils import get_offsets, get_sigmas, get_prior
from utils.pointcloud_sorting import sort_verts


class ThreeDExperiment(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(ThreeDExperiment, self).__init__()

        self.hparams = hparams

        self.sigma_begin = float(self.hparams.trainer.sigma_begin)
        self.sigma_end = float(self.sigma_begin / self.hparams.trainer.sigma_denominator)
        self.num_classes = int(self.hparams.trainer.sigma_num)
        self.sigmas = np.exp(
            np.linspace(np.log(self.sigma_begin),
                        np.log(self.sigma_end),
                        self.num_classes, dtype=np.float32))
        self.sigma_momentum = self.hparams.trainer.sigma_momentum
        print(self.sigmas)
        self.create_model()
        self.folding_loss = MaxChamferDistance()
        self.identity_folding_loss = nn.MSELoss()
        self.val_loss = ChamferDistance()

        self.sigma_best = nn.Parameter(torch.ones(1) * self.sigma_begin)
        self.sigma_begin = nn.Parameter(torch.ones(1) * self.sigma_begin)
        self.sigma_loss = MaxDistance()


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
        self.folding_decoder = define_decoder_from_params(self.hparams.folding_decoder_params)

    @staticmethod
    def get_nearest_point_sampling_config(point_count):
        TriangleConfig = utils.data_config.ThreeDConfig()

        TriangleConfig.sample_points = 'surface_points'

        TriangleConfig.sample_settings.sample_points_count = point_count
        TriangleConfig.sample_settings.sample_points_noise = 0
        TriangleConfig.sample_settings.sample_even = True
        TriangleConfig.sample_settings.sample_on_surface = True

        TriangleConfig.backend = 'trimesh'

        return TriangleConfig

    def get_transforms(self, name):
        def _sort_verts(data):
            data['surface_points'] = np.array(sort_verts(data['surface_points']))
            return data
        return _sort_verts
        #return None

    def prepare_data(self):
        config = self.get_nearest_point_sampling_config(self.hparams.points_count)

        self.train_dataset = SortedItemsDatasetWrapper(items=['surface_points'], dataset=TransformedDataset(
            SampledShapeNetV2Dataset(config, self.hparams.data_params.train_dataset_path, 'train'),
            self.get_transforms('train')))
        self.val_dataset = SortedItemsDatasetWrapper(items=['surface_points'], dataset=TransformedDataset(
            SampledShapeNetV2Dataset(config, self.hparams.data_params.val_dataset_path, 'test', use_all_points=True),
            self.get_transforms('valid')))

    def forward(self, perturbed_points, z, sigma):
        bs, num_pts = perturbed_points.size(0), perturbed_points.size(1)
        sigma = sigma.view(bs, 1)
        shape_latent = torch.cat((z, sigma), dim=1)
        y_pred = self.decoder(perturbed_points, shape_latent)

        return y_pred

    def reconstruction_loss(self, input_cloud, perturbed_points, y_pred, sigmas):
        bs, num_pts = input_cloud.size(0), input_cloud.size(1)
        y_gtr = - (perturbed_points - input_cloud)

        # The loss for each sigma is weighted
        lambda_sigma = 1. / sigmas.view(-1, 1, 1)
        loss = 0.5 * ((y_gtr - y_pred) ** 2. * lambda_sigma).sum(dim=2).mean()
        return loss

    def training_step_end(self, outputs):
        out = outputs['out']
        batch_idx = 0
        #print(out['folding_points'][batch_idx].view(1, -1, 3).shape)
        if self.global_step % self.hparams.log_point_cloud == 0:
            points_gt = out['input_point_cloud'][batch_idx]
            self.log_point_cloud('train/' + 'input_point_cloud', points_gt.unsqueeze(0))
            points_pred = out['perturbed_points'][batch_idx:batch_idx+1] + out['y_pred'][batch_idx:batch_idx+1]
            self.log_point_cloud('train/' + 'predicted_point_cloud', points_pred)

            self.log_point_cloud('train/' + 'folding_point_cloud', out['folding_points'][batch_idx].unsqueeze(0))

        return outputs

    def log_point_cloud(self, name, point_cloud):
        self.logger.experiment.add_mesh(name, point_cloud, global_step=self.global_step)

    def update_sigmas(self):
        sigma_begin = self.sigma_best.data.detach().item()
        sigma_end = sigma_begin / self.hparams.trainer.sigma_denominator
        self.sigmas = np.exp(
            np.linspace(np.log(sigma_begin),
                        np.log(sigma_end),
                        self.num_classes, dtype=np.float32))

    def get_nearest_points(self, perturbed_points, input):
        sided_minimum_dist = SidedDistance()
        closest_index_in_S2 = sided_minimum_dist(
            perturbed_points, input)
        closest_S2 = torch.stack([torch.index_select(input[i], 0, closest_index_in_S2[i]) for i in range(input.shape[0])])
        return closest_S2

    def training_step(self, batch, batch_idx):

        input = batch[0]

        offsets = torch.from_numpy(get_offsets(input)).to(input.device)
        sigmas = torch.from_numpy(get_sigmas(self.sigmas, input.shape[0])).to(input.device)

        z_mu, z_sigma = self.encoder(input)
        z = z_mu + 0 * z_sigma

        folded_points, folded_points_first, grid = self.folding_decoder(z)
        folding_loss = self.folding_loss(folded_points, input) * self.hparams.trainer.folding_coef + self.folding_loss(folded_points_first, input) * self.hparams.trainer.folding_first_coef

        max_distance = self.sigma_loss(folded_points, input)
        clipped_distance = torch.min(torch.ones(1).to(z.device) * self.hparams.trainer.sigma_begin, max_distance)
        self.sigma_begin.data = self.sigma_momentum * clipped_distance + (1 - self.sigma_momentum) * self.sigma_begin.data
        if (self.global_step + 1) % self.hparams.trainer.update_sigmas_step == 0 and self.sigma_begin.data.item() < self.sigma_best.data.item() * self.hparams.trainer.sigma_decrease_coef:
            self.sigma_best.data = self.sigma_best.data * self.hparams.trainer.sigma_decrease_coef
            self.update_sigmas()
            print('new_sigmas is', self.sigmas)


        grid_coef = 0
        if self.global_step < self.hparams.trainer.folding_grid_steps:
            grid_coef = self.hparams.trainer.folding_grid_coef * (1 - self.global_step / self.hparams.trainer.folding_grid_steps)
            folding_loss += grid_coef * (self.identity_folding_loss(folded_points, grid) + self.identity_folding_loss(folded_points_first, grid))

        #self.sigmas_min.data = self.sigma_momentum * torch.sqrt(self.folding_loss(folded_points, input)) + (1 - self.sigma_momentum) * self.sigmas_min.data

        perturbed_points = input + offsets * sigmas.view(-1, 1, 1)
        y_pred = self.forward(perturbed_points, z, sigmas)

        with torch.no_grad():
            input_nearest = self.get_nearest_points(perturbed_points, input)

        reconstruction_loss = self.reconstruction_loss(input_nearest, perturbed_points, y_pred, sigmas)

        loss = reconstruction_loss + folding_loss

        out = {'input_point_cloud': input,
               'y_pred': y_pred,
               'perturbed_points': perturbed_points,
               'folding_points': folded_points,
               'sigmas': sigmas
               }
        log = {'train/loss': loss, 'max': input.max(), 'min': input.min(), 'folding_loss': folding_loss,
               'reconstruction_loss': reconstruction_loss, 'grid_coef': grid_coef, 'sigmas': self.sigma_begin.data.item()}
        return {'loss': loss, 'out': out, 'log': log, 'progress_bar': log}

    def validation_step(self, batch, batch_nb):
        self.update_sigmas()
        print(self.sigmas)
        if batch_nb == 0:
            input = batch[0]
            z, _ = self.encoder(input)
            prior_cloud, _, _ = self.folding_decoder(z)
            pred = self.langevin_dynamics(z, prior_cloud, self.hparams.valid_points_count, self.hparams.step_size_ratio,
                                          self.hparams.num_steps,
                                          self.hparams.weight)
            print(pred.shape)
            self.log_point_cloud('valid/' + 'input_point_cloud', input[0].unsqueeze(0))
            self.log_point_cloud('valid/' + 'pred_point_cloud', pred[0].unsqueeze(0))
            self.log_point_cloud('valid/' + 'folding_point_cloud', prior_cloud[0].unsqueeze(0))

        z, _ = self.encoder(batch[0])
        prior_cloud, _, _ = self.folding_decoder(z)
        pred = self.langevin_dynamics(z, prior_cloud, self.hparams.valid_points_count, self.hparams.step_size_ratio,
                                      self.hparams.num_steps,
                                      self.hparams.weight)
        loss = self.val_loss(batch[0], pred )
        #print('valid/chamfer_distance', loss)
        log = {'valid/chamfer_distance': loss}
        self.log_dict(log, prog_bar=False, on_step=True, on_epoch=True)


    def langevin_dynamics(self, z, prior_cloud, num_points=2048, step_size_ratio=1, num_steps=10, weight=1):
        with torch.no_grad():
            sigmas = self.sigmas
            x_list = []
            self.decoder.eval()
            current_points = prior_cloud.shape[1]
            copies = []
            while current_points < num_points:
                if current_points + prior_cloud.shape[1] > num_points:
                    ids = torch.randperm(prior_cloud.shape[1])[:num_points - current_points]
                    copy = prior_cloud[:,ids,:]
                else:
                    copy = prior_cloud
                current_points += copy.shape[1]
                copies.append(copy)
            assert current_points >= num_points

            x = torch.cat([prior_cloud, *copies], dim=1)
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
                    #print(grad)
                    grad = grad / sigma ** 2
                    x += 0.5 * step_size * grad
                #print('max/min for sigma', sigma, ':',x.max(),'/',x.min())
        return x


    def configure_optimizers(self):
        params = itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.folding_decoder.parameters())
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
