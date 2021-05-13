import itertools
import os
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
from datasets.sampled_shapenetv2_with_image import SampledShapeNetV2WithImageDataset, get_transform
from datasets.transformed_dataset import TransformedDataset
from experiment import ThreeDExperiment
from losses import ChamferDistance, MaxChamferDistance
from losses.losses import define_loss, define_loss_from_params
from torch.utils.data.dataset import ConcatDataset
from losses.point_loss import MaxDistance, SigmaDistance, RobustSigmaDistance, EMDLoss, distChamfer
from models import networks

import torch
import time
import numpy as np

from models.networks import define_encoder_from_params, define_decoder_from_params
from registry import registry, registries
from utils.point_cloud_utils import get_offsets, get_sigmas, get_prior
from utils.pointcloud_sorting import sort_verts


class SVRExperiment(pl.LightningModule):

    def __init__(self,
                 hparams: Namespace) -> None:
        super(SVRExperiment, self).__init__()

        self.hparams = hparams


        point_experiment = ThreeDExperiment.load_from_checkpoint(hparams.pretrained_experiment, map_location='cpu')

        self.create_model()

        self.encoder_points = point_experiment.encoder
        self.decoder = point_experiment.decoder
        self.folding_decoder = point_experiment.folding_decoder

        self.set_requires_grad(self.encoder_points, False)
        self.set_requires_grad(self.decoder, False)
        self.set_requires_grad(self.folding_decoder, False)

        self.folding_loss = EMDLoss()
        self.folding_loss_simple = MaxChamferDistance()
        self.feature_loss = nn.MSELoss()
        self.val_loss = ChamferDistance()
        self.test_loss = distChamfer

        self.sigma_begin = point_experiment.sigma_best_max
        self.sigma_end = point_experiment.sigma_best_min
        self.num_classes = point_experiment.num_classes
        print('sigma info', self.sigma_begin, self.sigma_end, self.num_classes)

        del point_experiment

        self._eval_point_network()

    def _eval_point_network(self):
        self.encoder_points.eval()
        self.decoder.eval()
        self.folding_decoder.eval()

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_scheduler(self, optimizer):

        args = {**self.hparams.scheduler_params}
        args['optimizer'] = optimizer

        return registries.SCHEDULERS.get_from_params(**args)


    def create_model(self):
        self.encoder = define_encoder_from_params(self.hparams.encoder_params)


    @staticmethod
    def get_nearest_point_sampling_config(point_count, all_points_count):
        TriangleConfig = utils.data_config.ThreeDConfig()

        TriangleConfig.sample_points = 'surface_points'
        TriangleConfig.all_points = 'all_points'
        TriangleConfig.all_points_count = all_points_count

        TriangleConfig.sample_settings.sample_points_count = point_count
        TriangleConfig.sample_settings.sample_points_noise = 0
        TriangleConfig.sample_settings.sample_even = True
        TriangleConfig.sample_settings.sample_on_surface = True

        TriangleConfig.backend = 'trimesh'

        return TriangleConfig

    def get_transforms(self, name):
        def _sort_verts(data):
            #data['surface_points'] = np.array(sort_verts(data['surface_points']))
            return data
        return _sort_verts
        #return None

    # Fix bug in lr scheduling
    def on_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.check_val_every_n_epoch != 0:
            with torch.no_grad():
                self.trainer.optimizer_connector.update_learning_rates(interval='epoch')

    def prepare_data(self):
        config = self.get_nearest_point_sampling_config(self.hparams.points_count, self.hparams.nearest_points_count)

        self.train_dataset = SortedItemsDatasetWrapper(items=['image', 'surface_points', 'all_points'], dataset=TransformedDataset(
                SampledShapeNetV2WithImageDataset(config, self.hparams.data_params.train_dataset_path, 'train', self.hparams.data_params.train_image_dataset_path, get_transform(self.hparams.image_transform, True)),
                self.get_transforms('train')))
        self.val_dataset = SortedItemsDatasetWrapper(items=['image', 'surface_points', 'all_points', 'center', 'scale'], dataset=TransformedDataset(
            SampledShapeNetV2WithImageDataset(config, self.hparams.data_params.val_dataset_path, 'val', self.hparams.data_params.val_image_dataset_path, get_transform(self.hparams.image_transform, False), use_all_points=True),
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
            self.log_point_cloud('train/' + 'perturbed_points', out['perturbed_points'][batch_idx].unsqueeze(0))

            self.log_point_cloud('train/' + 'folding_point_cloud', out['folding_points'][batch_idx].unsqueeze(0))

            # Log images
            out_image = out['image']
            grid = torchvision.utils.make_grid(out_image)

            grid = grid * 0.5 + 0.5
            grid = torch.clamp(grid, 0.0, 1.0)

            self.logger.experiment.add_image('train_image', grid, self.global_step)

        return outputs

    def log_point_cloud(self, name, point_cloud):
        self.logger.experiment.add_mesh(name, point_cloud, global_step=self.global_step)

    def update_sigmas(self):
        sigma_begin = self.sigma_begin.item()
        sigma_end = self.sigma_end.item()
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

    def on_train_epoch_start(self) -> None:
        self.update_sigmas()
        print(self.sigmas)

    def training_step(self, batch, batch_idx):

        image, input, all_points = batch[0], batch[1], batch[2]

        offsets = torch.from_numpy(get_offsets(input)).to(input.device)
        sigmas = torch.from_numpy(get_sigmas(self.sigmas, input.shape[0])).to(input.device)


        with torch.no_grad():
            z_mu_gt, z_sigma_gt = self.encoder_points(input)
        z_mu, z_sigma = self.encoder(image)

        z = z_mu + 0 * z_sigma

        feature_loss = self.feature_loss(z_mu, z_mu_gt) * self.hparams.trainer.feature_coef

        folded_points, folded_points_first, grid = self.folding_decoder(z)

        folding_loss = self.folding_loss(folded_points,
                                         input) * self.hparams.trainer.folding_coef + self.folding_loss_simple(
            folded_points_first, input) * self.hparams.trainer.folding_first_coef


        perturbed_points = input + offsets * sigmas.view(-1, 1, 1)
        y_pred = self.forward(perturbed_points, z, sigmas)

        with torch.no_grad():
            if self.hparams.trainer.use_nearest:
                input_nearest = self.get_nearest_points(perturbed_points, all_points)
            else:
                input_nearest = input

        reconstruction_loss = self.reconstruction_loss(input_nearest, perturbed_points, y_pred, sigmas) * self.hparams.trainer.reconstruction_coef

        loss = reconstruction_loss + folding_loss + feature_loss

        out = {'input_point_cloud': input,
               'y_pred': y_pred,
               'perturbed_points': perturbed_points,
               'folding_points': folded_points,
               'sigmas': sigmas,
               'image': image
               }
        log = {'loss': loss, 'folding_loss': folding_loss,
               'reconstruction_loss': reconstruction_loss, 's_begin': self.sigma_begin.item(), 's_end': self.sigma_end.item()}
        prog_bar = log.copy()

        return {'loss': loss, 'out': out, 'log': log, 'progress_bar': prog_bar}

    def renormalize(self, data, center, scale):
        norm_vert = data * scale.view(-1, 1, 1) + center.view(-1, 1, 3)
        return norm_vert

    def validation_step(self, batch, batch_nb):
        self.update_sigmas()
        print(self.sigmas)
        if batch_nb == 0:
            image, input, all_points, _, _ = batch
            z, _ = self.encoder(image)
            if self.hparams.use_folding:
                prior_cloud, _, _ = self.folding_decoder(z)
            else:
                prior_cloud = get_prior(z.size(0), self.hparams.valid_points_count, 3)
            pred = self.langevin_dynamics(z, prior_cloud, self.hparams.valid_points_count, self.hparams.step_size_ratio,
                                          self.hparams.num_steps,
                                          self.hparams.weight)
            print(input.shape, all_points.shape, pred.shape)
            self.log_point_cloud('valid/' + 'input_point_cloud', all_points[0].unsqueeze(0))
            self.log_point_cloud('valid/' + 'pred_point_cloud', pred[0].unsqueeze(0))
            self.log_point_cloud('valid/' + 'folding_point_cloud', prior_cloud[0].unsqueeze(0))

            self.log_point_cloud('valid/' + 'input_small', input[0].unsqueeze(0))
            ids = torch.randperm(pred.shape[1])[:self.hparams.trainer.valid_chamfer_points]
            self.log_point_cloud('valid/' + 'folding_point_cloud_small', pred[:, ids, :][0].unsqueeze(0))

            # Log images
            out_image = image
            grid = torchvision.utils.make_grid(out_image)

            grid = grid * 0.5 + 0.5
            grid = torch.clamp(grid, 0.0, 1.0)

            self.logger.experiment.add_image('valid_image', grid, self.global_step)


        image, input, all_points, center, scale = batch
        z, _ = self.encoder(image)
        if self.hparams.use_folding:
            prior_cloud, _, _ = self.folding_decoder(z)
        else:
            prior_cloud = get_prior(z.size(0), self.hparams.valid_points_count, 3)
        pred = self.langevin_dynamics(z, prior_cloud, self.hparams.valid_points_count, self.hparams.step_size_ratio,
                                      self.hparams.num_steps,
                                      self.hparams.weight)

        renormalized_all_points, renormalized_pred = self.renormalize(all_points, center, scale), self.renormalize(pred, center, scale)
        loss = self.val_loss(renormalized_pred, renormalized_all_points)
        if not torch.all(torch.isfinite(loss)):
            loss = torch.ones(1).mean().to(pred.device)
        #print('valid/chamfer_distance', loss)
        self.log('valid_chamfer_distance', loss)

        small_input = input
        ids = torch.randperm(pred.shape[1])[:self.hparams.trainer.valid_chamfer_points]
        small_pred = pred[:, ids, :]

        renormalized_small_input, renormalized_small_pred = self.renormalize(small_input, center, scale), self.renormalize(small_pred,
                                                                                                                   center,
                                                                                                                   scale)
        small_loss = self.val_loss(renormalized_small_pred, renormalized_small_input)
        if not torch.all(torch.isfinite(small_loss)):
            small_loss = torch.ones(1).mean().to(small_pred.device)
        self.log('valid_chamfer_distance_small', small_loss)

        log = {'valid/chamfer_distance': loss}
        self.log_dict(log, prog_bar=False, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_nb):
        self.encoder.eval()
        self.decoder.eval()
        self.folding_decoder.eval()
        self.update_sigmas()
        print(self.sigmas)

        def log_point_cloud(name, point_cloud, step):
            self.logger.experiment.add_mesh(name, point_cloud, global_step=step)

        input, all_points, center, scale = batch
        z, _ = self.encoder(input)
        prior_cloud, _, _ = self.folding_decoder(z)
        pred = self.langevin_dynamics(z, prior_cloud, self.hparams.valid_points_count, self.hparams.step_size_ratio,
                                      self.hparams.num_steps,
                                      self.hparams.weight)
        print(input.shape, pred.shape)
        for i in range(pred.shape[0]):
            step = batch_nb * input.shape[0] + i
            log_point_cloud('valid/' + 'input_point_cloud', all_points[i].unsqueeze(0), step)
            log_point_cloud('valid/' + 'pred_point_cloud', pred[i].unsqueeze(0), step)
            #log_point_cloud('valid/' + 'folding_point_cloud', prior_cloud[i].unsqueeze(0), step)

            #log_point_cloud('valid/' + 'input_small', input[i].unsqueeze(0), step)
            #ids = torch.randperm(pred.shape[1])[:self.hparams.trainer.valid_chamfer_points]
            #log_point_cloud('valid/' + 'folding_point_cloud_small', pred[:, ids, :][i].unsqueeze(0), step)

        renormalized_all_points, renormalized_pred = self.renormalize(all_points, center, scale), self.renormalize(pred,
                                                                                                                   center,
                                                                                                                   scale)

        # assert  renormalized_pred.shape[-1] == 3 and renormalized_all_points.shape[-1] == 3
        # dl, dr = self.test_loss(renormalized_pred, renormalized_all_points)
        # loss = dl.mean(dim=1) + dr.mean(dim=1)

        output = {}
        # print('valid/chamfer_distance', loss)
        # output['valid_chamfer_distance'] = loss


        small_input = input
        ids = torch.randperm(pred.shape[1])[:self.hparams.trainer.valid_chamfer_points]
        small_pred = pred[:, ids, :]

        renormalized_small_input, renormalized_small_pred = self.renormalize(small_input, center,
                                                                             scale), self.renormalize(small_pred,
                                                                                                      center,
                                                                                                      scale)
        dl, dr = self.test_loss(renormalized_small_pred, renormalized_small_input)
        small_loss = dl.mean(dim=1) + dr.mean(dim=1)
        print('small_loss', small_loss)
        output['valid_chamfer_distance_small'] = small_loss

        os.makedirs('pred', exist_ok=True)
        os.makedirs('gt', exist_ok=True)
        for i in range(pred.shape[0]):
            with open(f'pred/batch_{self.hparams.val_batch_size * batch_nb + i}_loss_{small_loss[i].item():.6f}.npy',
                      'wb') as f:
                np.save(f, pred[i].detach().cpu().numpy())

            with open(f'gt/batch_{self.hparams.val_batch_size * batch_nb + i}_loss_{small_loss[i].item():.6f}.npy',
                      'wb') as f:
                np.save(f, all_points[i].detach().cpu().numpy())

        return output

    def test_epoch_end(self, outputs):
        def merge_dict(outputs):
            if not outputs:
                return {}
            keys = outputs[0].keys()
            result = {}
            for key in keys:
                merged_values = []
                for value in outputs:
                    merged_values.append(value[key])
                result[key] = torch.cat(merged_values, dim=0)

            return result


        outputs = merge_dict(outputs)
        for k,v in outputs.items():
            loss = v.mean()
            print(f'loss {k} has value {loss}')
            self.log(k, loss)


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
                step_size = sigma ** 2 * step_size_ratio
                for t in range(num_steps):
                    z_t = torch.randn_like(x) * weight
                    x += torch.sqrt(step_size) * z_t
                    grad = self.decoder(x, z_sigma)
                    grad = grad / sigma ** 2
                    x += step_size * grad
                #print('max/min for sigma', sigma, ':',x.max(),'/',x.min())
        return x


    def configure_optimizers(self):
        params = self.encoder.parameters()
        optimizer = registries.OPTIMIZERS.get_from_params(**{'params': params, **self.hparams.optimizer_params})

        return [optimizer], [self.get_scheduler(optimizer)]

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.val_batch_size,
                          shuffle=True,
                          drop_last=False,
                          num_workers=self.hparams.data_params.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.hparams.val_batch_size,
                          shuffle=True,
                          drop_last=False,
                          num_workers=self.hparams.data_params.num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.data_params.batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=self.hparams.data_params.num_workers)
