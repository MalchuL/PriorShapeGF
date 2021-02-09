# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# pointnet2
#
# Copyright (c) 2017, Geometric Computation Group of Stanford University
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Charles R. Qi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import kaolin.cuda as ext
import kaolin.cuda.ball_query
import kaolin.cuda.furthest_point_sampling
import kaolin.cuda.three_nn

from .resnet import BasicBlockReLU as BasicBlock


class PointNetFeatureExtractor(nn.Module):
    r"""PointNet feature extractor (extracts either global or local, i.e.,
    per-point features).
    Based on the original PointNet paper:.
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2016pointnet,
              title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
              author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
              journal={arXiv preprint arXiv:1612.00593},
              year={2016}
            }
    Args:
        in_channels (int): Number of channels in the input pointcloud
            (default: 3, for X, Y, Z coordinates respectively).
        feat_size (int): Size of the global feature vector
            (default: 1024)
        layer_dims (Iterable[int]): Sizes of fully connected layers
            to be used in the feature extractor (excluding the input and
            the output layer sizes). Note: the number of
            layers in the feature extractor is implicitly parsed from
            this variable.
        global_feat (bool): Extract global features (i.e., one feature
            for the entire pointcloud) if set to True. If set to False,
            extract per-point (local) features (default: True).
        activation (function): Nonlinearity to be used as activation
                    function after each batchnorm (default: F.relu)
        batchnorm (bool): Whether or not to use batchnorm layers
            (default: True)
        transposed_input (bool): Whether the input's second and third dimension
            is already transposed. If so, a transpose operation can be avoided,
            improving performance.
            See documentation for the forward method for more details.
    For example, to specify a PointNet feature extractor with 4 linear
    layers (sizes 6 -> 10, 10 -> 40, 40 -> 500, 500 -> 1024), with
    3 input channels in the pointcloud and a global feature vector of size
    1024, see the example below.
    Example:
        >>> pointnet = PointNetFeatureExtractor(in_channels=3, feat_size=1024,
                                           layer_dims=[10, 20, 40, 500])
        >>> x = torch.rand(2, 3, 30)
        >>> y = pointnet(x)
        print(y.shape)
    """

    def __init__(self,
                 in_channels: int = 3,
                 feat_size: int = 1024,
                 layer_dims=[64, 128],
                 global_feat: bool = True,
                 batchnorm: bool = True,
                 transposed_input: bool = False):
        super(PointNetFeatureExtractor, self).__init__()

        if not isinstance(in_channels, int):
            raise TypeError('Argument in_channels expected to be of type int. '
                            'Got {0} instead.'.format(type(in_channels)))
        if not isinstance(feat_size, int):
            raise TypeError('Argument feat_size expected to be of type int. '
                            'Got {0} instead.'.format(type(feat_size)))
        if not hasattr(layer_dims, '__iter__'):
            raise TypeError('Argument layer_dims is not iterable.')
        for idx, layer_dim in enumerate(layer_dims):
            if not isinstance(layer_dim, int):
                raise TypeError('Elements of layer_dims must be of type int. '
                                'Found type {0} at index {1}.'.format(
                    type(layer_dim), idx))
        if not isinstance(global_feat, bool):
            raise TypeError('Argument global_feat expected to be of type '
                            'bool. Got {0} instead.'.format(
                type(global_feat)))

        # Store feat_size as a class attribute
        self.feat_size = feat_size

        # Store activation as a class attribute
        #self.activation = activation

        # Store global_feat as a class attribute
        self.global_feat = global_feat

        # Add in_channels to the head of layer_dims (the first layer
        # has number of channels equal to `in_channels`). Also, add
        # feat_size to the tail of layer_dims.
        if not isinstance(layer_dims, list):
            layer_dims = list(layer_dims)
        layer_dims.insert(0, in_channels)
        layer_dims.append(feat_size)


        self.blocks = nn.ModuleList([BasicBlock(layer_dims[idx],
                                              layer_dims[idx + 1], norm_layer=batchnorm) for idx in range(len(layer_dims) - 1)])


        # Store whether or not to use batchnorm as a class attribute
        self.batchnorm = batchnorm

        self.transposed_input = transposed_input

    def forward(self, x: torch.Tensor):
        r"""Forward pass through the PointNet feature extractor.
        Args:
            x (torch.Tensor): Tensor representing a pointcloud
                (shape: :math:`B \times N \times D`, where :math:`B`
                is the batchsize, :math:`N` is the number of points
                in the pointcloud, and :math:`D` is the dimensionality
                of each point in the pointcloud).
                If self.transposed_input is True, then the shape is
                :math:`B \times D \times N`.
        """
        if not self.transposed_input:
            x = x.transpose(1, 2)

        # Number of points
        num_points = x.shape[2]

        # By default, initialize local features (per-point features)
        # to None.
        local_features = None

        # Apply a sequence of conv-batchnorm-nonlinearity operations

        # For the first layer, store the features, as these will be
        # used to compute local features (if specified).

        x = self.blocks[0](x)


        if self.global_feat is False:
            local_features = x

        # Pass through the remaining layers (until the penultimate layer).

        for idx in range(1, len(self.blocks)):
            x = self.blocks[idx](x)




        # Max pooling.
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.feat_size)

        # If extracting global features, return at this point.
        if self.global_feat:
            return x

        # If extracting local features, compute local features by
        # concatenating global features, and per-point features
        x = x.view(-1, self.feat_size, 1).repeat(1, 1, num_points)
        return torch.cat((x, local_features), dim=1)


class FurthestPointSampling(torch.autograd.Function):
    r"""
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    """

    @staticmethod
    def forward(ctx, xyz, num_points_out):
        """Uses iterative furthest point sampling to select a set of num_points_out features that have the largest minimum distance.
        Args:
            xyz (torch.Tensor): (B, N, 3) tensor where N > num_points_out
            num_points_out (int32): number of features in the sampled set
        Returns:
            (torch.Tensor): (B, num_points_out) tensor containing the set
        """
        return ext.furthest_point_sampling.furthest_point_sampling(xyz, num_points_out)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sampling = FurthestPointSampling.apply


class FPSGatherByIndex(torch.autograd.Function):
    r"""
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    """

    @staticmethod
    def forward(ctx, features, idx):
        """TODO: documentation (and the ones below)
        Args:
            features (torch.Tensor): (B, C, N) tensor
            idx (torch.Tensor): (B, npoint) tensor of the features to gather
        Returns:
            (torch.Tensor): (B, C, npoint) tensor
        """

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return ext.furthest_point_sampling.gather_by_index(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = ext.furthest_point_sampling.gather_by_index_grad(
            grad_out.contiguous(), idx, N)
        return grad_features, None


fps_gather_by_index = FPSGatherByIndex.apply


class ThreeNN(torch.autograd.Function):
    r"""
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    """

    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features
        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        dist2, idx = ext.three_nn.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(torch.autograd.Function):
    r"""
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    """

    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights
        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return ext.three_nn.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs
        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features
        None
        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_features = ext.three_nn.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupGatherByIndex(torch.autograd.Function):
    r"""
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    """

    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with
        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return ext.ball_query.gather_by_index(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward
        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        grad_features = ext.ball_query.gather_by_index_grad(
            grad_out.contiguous(), idx, N)

        return grad_features, None


group_gather_by_index = GroupGatherByIndex.apply


class BallQuery(torch.autograd.Function):
    r"""
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    """

    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz, use_random=False):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        TODO: documentation
        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query
        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        if use_random:
            return ext.ball_query.ball_random_query(
                torch.randint(int(1e9), ()).item(), new_xyz, xyz, radius,
                nsample)

        return ext.ball_query.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


# TODO: improvement: experiment with random sampling instead of current approach.


def separate_xyz_and_features(points):
    """Break up a point cloud into position vectors (first 3 dimensions) and feature vectors.
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    Args:
        points (torch.Tensor): shape = (batch_size, num_points, 3 + num_features)
            The point cloud to separate.
    Returns:
        xyz (torch.Tensor): shape = (batch_size, num_points, 3)
            The position vectors of the points.
        features (torch.Tensor|None): shape = (batch_size, num_features, num_points)
            The feature vectors of the points.
            If there are no feature vectors, features will be None.
    """
    assert (len(points.shape) == 3 and points.shape[2] >= 3), (
        'Expected shape of points to be (batch_size, num_points, 3 + num_features), got {}'
            .format(points.shape))

    xyz = points[:, :, 0:3].contiguous()
    features = (points[:, :, 3:].transpose(1, 2).contiguous()
                if points.shape[2] > 3 else None)

    return xyz, features


class PointNet2GroupingLayer(nn.Module):
    """
    TODO: documentation: if radius is None, then group everything
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    """

    def __init__(self, radius, num_samples, use_xyz_feature=True, use_random_ball_query=False):
        super(PointNet2GroupingLayer, self).__init__()
        self.radius = radius
        self.num_samples = num_samples
        self.use_xyz_feature = use_xyz_feature
        self.use_random_ball_query = use_random_ball_query

    def forward(self, xyz, new_xyz, features=None):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)
        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        if self.radius is None:
            grouped_xyz = xyz.transpose(1, 2)
            if features is not None:
                grouped_features = features
                if self.use_xyz_feature:
                    new_features = torch.cat(
                        [grouped_xyz, grouped_features], dim=1
                    )  # (B, 3 + C, 1, N)
                else:
                    new_features = grouped_features
            else:
                new_features = grouped_xyz

            return new_features

        else:
            idx = ball_query(self.radius, self.num_samples, xyz,
                             new_xyz, self.use_random_ball_query)
            xyz_trans = xyz.transpose(1, 2).contiguous()
            grouped_xyz = group_gather_by_index(
                xyz_trans, idx)  # (B, 3, npoint, nsample)
            grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

            if features is not None:
                grouped_features = group_gather_by_index(features, idx)
                if self.use_xyz_feature:
                    new_features = torch.cat(
                        [grouped_xyz, grouped_features], dim=1
                    )  # (B, C + 3, npoint, nsample)
                else:
                    new_features = grouped_features
            else:
                assert self.use_xyz_feature, "Must have at least one feature or set use_xyz_feature = True"
                new_features = grouped_xyz

            return new_features.transpose(1, 2).contiguous()


class PointNet2SetAbstraction(nn.Module):
    """A single set-abstraction layer for the PointNet++ architecture.
    Supports multi-scale grouping (MSG).
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    Args:
        num_points_out (int|None): The number of output points.
            If None, group all points together.
        pointnet_in_features (int): The number of features to input into pointnet.
            Note: if use_xyz_feature is true, this value will be increased by 3.
        pointnet_layer_dims_list (List[List[int]]): The pointnet MLP dimensions list for each scale.
            Note: the first (input) dimension SHOULD NOT be included in each list,
            while the last (output) dimension SHOULD be included in each list.
        radii_list (List[float]|None): The grouping radius for each scale.
            If num_points_out is None, this value is ignored.
        num_samples_list (List[int]|None): The number of samples in each ball query for each scale.
            If num_points_out is None, this value is ignored.
        batchnorm (bool): Whether or not to use batch normalization.
        use_xyz_feature (bool): Whether or not to use the coordinates of the
            points as feature.
        use_random_ball_query (bool): Whether or not to use random sampling when
            there are too many points per ball.
    """

    def __init__(self,
                 num_points_out,
                 pointnet_in_features,
                 pointnet_layer_dims_list,
                 radii_list=None,
                 num_samples_list=None,
                 batchnorm=True,
                 use_xyz_feature=True,
                 use_random_ball_query=False):

        super(PointNet2SetAbstraction, self).__init__()

        # TODO: Testing: test if the model works with each of the parameters

        if num_points_out is None:
            radii_list = [None]
            num_samples_list = [None]
        else:
            assert isinstance(radii_list, list) and isinstance(
                num_samples_list, list), 'radii_list and num_samples_list must be lists'

        assert (len(radii_list) == len(num_samples_list) == len(pointnet_layer_dims_list)), (
            'Dimension of radii_list ({}), num_samples_list ({}), pointnet_layer_dims_list ({}) must match'
                .format(len(radii_list), len(num_samples_list), len(pointnet_layer_dims_list)))

        self.num_points_out = num_points_out
        self.pointnet_layer_dims_list = pointnet_layer_dims_list

        self.groupers = []
        self.pointnets = []

        self.layers = []
        self.pointnet_in_channels = pointnet_in_features + \
                                    (3 if use_xyz_feature else 0)

        num_scales = len(radii_list)
        for i in range(num_scales):
            radius = radii_list[i]
            num_samples = num_samples_list[i]
            pointnet_layer_dims = pointnet_layer_dims_list[i]

            assert isinstance(pointnet_layer_dims,
                              list), 'Each pointnet_layer_dims must be a list, got {} instead'.format(
                pointnet_layer_dims)
            assert len(
                pointnet_layer_dims) > 0, 'Each pointnet_layer_dims must have at least one element'

            grouper = PointNet2GroupingLayer(
                radius, num_samples, use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query)

            # TODO: refactor: add dropout parameters
            pointnet = PointNetFeatureExtractor(
                in_channels=self.pointnet_in_channels,
                feat_size=pointnet_layer_dims[-1],
                layer_dims=pointnet_layer_dims[:-1],
                global_feat=True,
                batchnorm=batchnorm,
                transposed_input=True
            )

            # Register sub-modules
            self.groupers.append(grouper)
            self.pointnets.append(pointnet)

            self.layers.append((i, i, num_samples))

        self.groupers = nn.ModuleList(self.groupers)
        self.pointnets = nn.ModuleList(self.pointnets)


    def forward(self, xyz, features=None):
        """
        Args:
            xyz (torch.Tensor): shape = (batch_size, num_points_in, 3)
                The 3D coordinates of each point.
            features (torch.Tensor|None): shape = (batch_size, num_features, num_points_in)
                The features of each point.
        Returns:
            new_xyz (torch.Tensor|None): shape = (batch_size, num_points_out, 3)
                The new coordinates of the grouped points.
                If self.num_points_out is None, new_xyz will be None.
            new_features (torch.Tensor): shape = (batch_size, out_num_features, num_points_out)
                The features of each output point.
                If self.num_points_out is None, new_features will have shape:
                (batch_size, num_features_out)
        """
        batch_size = xyz.shape[0]


        new_xyz = None
        if self.num_points_out is not None:
            # TODO: implement: this is flipped here for some reason
            new_xyz_idx = furthest_point_sampling(xyz, self.num_points_out)
            new_xyz = fps_gather_by_index(
                xyz.transpose(1, 2).contiguous(), new_xyz_idx)
            new_xyz = new_xyz.transpose(1, 2).contiguous()

        new_features_list = []
        for grouper, pointnet, num_samples in self.layers:
            grouper = self.groupers[grouper]
            pointnet = self.pointnets[pointnet]

            new_features = grouper(xyz, new_xyz, features)
            # shape = (batch_size, num_points_out, self.pointnet_in_channels, num_samples)
            # if num_points_out is None:
            # shape = (batch_size, self.pointnet_in_channels, num_samples)

            if self.num_points_out is not None:
                new_features = new_features.view(-1,
                                                 self.pointnet_in_channels, num_samples)

            new_features = pointnet(new_features)
            # shape = (batch_size * num_points_out, feat_size)
            # if num_points_out is None:
            # shape = (batch_size, feat_size)

            # TODO: Optimization: avoid this packing and unpacking step by refactoring and generalizing pointnet
            if self.num_points_out is not None:
                new_features = new_features.view(
                    batch_size, self.num_points_out, -1).transpose(1, 2)
                # shape = (batch_size, feat_size, num_points_out)

            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1)
        # shape = (batch_size, num_features_out, num_points_out)
        # if num_points_out is None:
        # shape = (batch_size, num_features_out)

        return new_xyz, new_features

    def get_num_features_out(self):
        return sum([lst[-1] for lst in self.pointnet_layer_dims_list])


class PointNet2FeaturePropagator(nn.Module):
    """A single feature-propagation layer for the PointNet++ architecture.
    Used for segmentation.
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    Args:
        num_features (int): The number of features in the current layer.
            Note: this is the number of output features of the corresponding
            set abstraction layer.
        num_features_prev (int): The number of features from the previous
            feature propagation layer (corresponding to the next layer during
            feature extraction).
            Note: this is the number of output features of the previous feature
            propagation layer (or the number of output features of the final set
            abstraction layer, if this is the very first feature propagation
            layer)
        layer_dims (List[int]): Sizes of the MLP layer.
            Note: the first (input) dimension SHOULD NOT be included in the list,
            while the last (output) dimension SHOULD be included in the list.
        batchnorm (bool): Whether or not to use batch normalization.
    """

    def __init__(self, num_features, num_features_prev, layer_dims, batchnorm=True):
        super(PointNet2FeaturePropagator, self).__init__()

        self.layer_dims = layer_dims

        unit_pointnets = []
        in_features = num_features + num_features_prev
        for out_features in layer_dims:
            unit_pointnets.append(
                nn.Conv1d(in_features, out_features, 1))

            if batchnorm:
                unit_pointnets.append(nn.BatchNorm1d(out_features))

            unit_pointnets.append(nn.ReLU())
            in_features = out_features

        self.unit_pointnet = nn.Sequential(*unit_pointnets)

    def forward(self, xyz, xyz_prev, features=None, features_prev=None):
        """
        Args:
            xyz (torch.Tensor): shape = (batch_size, num_points, 3)
                The 3D coordinates of each point at current layer,
                computed during feature extraction (i.e. set abstraction).
            xyz_prev (torch.Tensor|None): shape = (batch_size, num_points_prev, 3)
                The 3D coordinates of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).
                This value can be None (i.e. for the very first propagator layer).
            features (torch.Tensor|None): shape = (batch_size, num_features, num_points)
                The features of each point at current layer,
                computed during feature extraction (i.e. set abstraction).
            features_prev (torch.Tensor|None): shape = (batch_size, num_features_prev, num_points_prev)
                The features of each point from the previous feature
                propagation layer (corresponding to the next layer during
                feature extraction).
        Returns:
            (torch.Tensor): shape = (batch_size, num_features_out, num_points)
        """
        num_points = xyz.shape[1]
        if xyz_prev is None:  # Very first feature propagation layer
            new_features = features_prev.expand(
                *(features.shape + [num_points]))

        else:
            dist, idx = three_nn(xyz, xyz_prev)
            # shape = (batch_size, num_points, 3), (batch_size, num_points, 3)
            inverse_dist = 1.0 / (dist + 1e-8)
            total_inverse_dist = torch.sum(inverse_dist, dim=2, keepdim=True)
            weights = inverse_dist / total_inverse_dist
            new_features = three_interpolate(features_prev, idx, weights)
            # shape = (batch_size, num_features_prev, num_points)

        if features is not None:
            new_features = torch.cat([new_features, features], dim=1)

        return self.unit_pointnet(new_features)

    def get_num_features_out(self):
        return self.layer_dims[-1]


class PointNet2Classifier(nn.Module):
    r"""PointNet++ classification network.
    Based on the original PointNet++ paper.
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    Args:
        in_features (int): Number of features (not including xyz coordinates) in
            the input point cloud (default: 0).
        num_classes (int): Number of classes (for the classification
            task) (default: 2).
        batchnorm (bool): Whether or not to use batch normalization.
            (default: True)
        use_xyz_feature (bool): Whether or not to use the coordinates of the
            points as feature.
        use_random_ball_query (bool): Whether or not to use random sampling when
            there are too many points per ball.
    TODO: Documentation: add example
    """

    # TODO: Implement: ssg

    def __init__(self,
                 in_features=0,
                 num_classes=2,
                 batchnorm=True,
                 use_xyz_feature=True,
                 use_random_ball_query=False):
        super(PointNet2Classifier, self).__init__()

        self.set_abstractions = nn.ModuleList()

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=512,
                pointnet_in_features=in_features,
                pointnet_layer_dims_list=[
                    [32, 32, 64],
                    [64, 64, 128],
                    [64, 96, 128],
                ],
                radii_list=[0.1, 0.2, 0.4],
                num_samples_list=[16, 32, 128],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=128,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [64, 64, 128],
                    [128, 128, 256],
                    [128, 128, 256],
                ],
                radii_list=[0.2, 0.4, 0.8],
                num_samples_list=[32, 64, 128],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=None,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [256, 512, 1024],
                ],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        final_layer_modules = [
            module for module in [
                nn.Linear(
                    self.set_abstractions[-1].get_num_features_out(), 512),
                nn.BatchNorm1d(512) if batchnorm else None,
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256) if batchnorm else None,
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            ] if module is not None
        ]
        self.final_layers = nn.Sequential(*final_layer_modules)

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): shape = (batch_size, num_points, 3 + in_features)
                The points to classify.
        Returns:
            (torch.Tensor): shape = (batch_size, num_classes)
                The score of the inputs being in each class.
                Note: no softmax or logsoftmax will be applied.
        """
        xyz, features = separate_xyz_and_features(points)

        for module in self.set_abstractions:
            xyz, features = module(xyz, features)

        return self.final_layers(features)


class PointNet2Featurizer(nn.Module):
    """PointNet++ classification network.
    .. note::
        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::
            @article{qi2017pointnet2,
                title = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
                author = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
                year = {2017},
                journal={arXiv preprint arXiv:1706.02413},
            }
    Args:
        in_features (int): Number of features (not including xyz coordinates) in
            the input point cloud (default: 0).
        num_classes (int): Number of classes (for the classification
            task) (default: 2).
        batchnorm (bool): Whether or not to use batch normalization.
            (default: True)
        use_xyz_feature (bool): Whether or not to use the coordinates of the
            points as feature.
        use_random_ball_query (bool): Whether or not to use random sampling when
            there are too many points per ball.
    TODO: Documentation: add example
    """

    def __init__(self,
                 in_features=0,
                 out_features=1900,
                 batchnorm=True,
                 use_xyz_feature=True,
                 use_random_ball_query=False):

        super(PointNet2Featurizer, self).__init__()

        self.set_abstractions = nn.ModuleList()

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=1024,
                pointnet_in_features=in_features,
                pointnet_layer_dims_list=[
                    [16, 16, 32],
                    [32, 32, 32],
                ],
                radii_list=[0.1, 0.13],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=256,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [64, 64, 64],
                    [64, 64, 64],
                ],
                radii_list=[0.16, 0.2],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=64,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [128, 128, 128],
                    [128, 128, 128],
                ],
                radii_list=[0.25, 0.33],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.set_abstractions.append(
            PointNet2SetAbstraction(
                num_points_out=16,
                pointnet_in_features=self.set_abstractions[-1].get_num_features_out(
                ),
                pointnet_layer_dims_list=[
                    [256, 256, 256],
                    [256, 256, 256],
                ],
                radii_list=[0.4, 0.5],
                num_samples_list=[16, 32],
                batchnorm=batchnorm,
                use_xyz_feature=use_xyz_feature,
                use_random_ball_query=use_random_ball_query
            )
        )

        self.sizes = [64, 128, 256, 512]

        self.masks = nn.ModuleList()



        self.out_feature = nn.Linear(sum(self.sizes), out_features)


        for size in self.sizes:
            self.masks.append(FeatureMasking(size + 3))

    def forward(self, points):
        """
        Args:
            points (torch.Tensor): shape = (batch_size,  3 + in_features, num_points)
                The points to perform segmentation on.
        Returns:
            (torch.Tensor): shape = (batch_size, num_points, num_classes)
                The score of each point being in each class.
                Note: no softmax or logsoftmax will be applied.
        """
        #points = points.transpose(2, 1)
        xyz, features = separate_xyz_and_features(points)

        xyz_list, features_list = [xyz], [features]

        for module in self.set_abstractions:
            #print(xyz.shape, features.shape)
            xyz, features = module(xyz, features)
            xyz_list.append(xyz)
            features_list.append(features)

        masked = []
        for i, features in enumerate(features_list[1:]):
            masked.append(self.masks[i](torch.cat([xyz_list[i+1].transpose(2,1),features], dim=1))[:,3:])

        masked = torch.cat(masked, dim=1)

        out = masked

        return self.out_feature(out), 1



class FeatureMax(nn.Module):
    def forward(self, x):
        out, _ = x.max(2)
        return out

class FeatureMasking(nn.Module):
    def __init__(self, features_size):
        super().__init__()

        self.features_preprocess = []
        for _ in range(3):
            self.features_preprocess += [BasicBlock(features_size, features_size, norm_layer=False)]
        self.features_preprocess = nn.Sequential(*self.features_preprocess)


        self.features_masking = []
        for _ in range(3):
            self.features_masking += [BasicBlock(features_size, features_size, norm_layer=False)]
        self.features_masking += [nn.Conv1d(features_size, features_size, 1)]
        self.features_masking = nn.Sequential(*self.features_masking)

        self.activation = nn.Sigmoid()
        self.eps = 1e-6

    def forward(self, x):
        x = self.features_preprocess(x)
        mask = self.activation(self.features_masking(x))
        out = (mask * x).sum(2) / ((mask).sum(2) + self.eps)
        return out


