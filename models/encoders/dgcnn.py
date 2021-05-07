#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
Modified by
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""
# https://github.com/AnTao97/dgcnn.pytorch
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from models.encoders.Identity import Identity


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels: int, eps: float = 1e-05, affine: bool = True, momentum=0.1,
                 divider=2):
        # assert num_channels % 2 == 0

        def find_best_divisor(size, low, high, step=1):
            minimal_truncation, best_divisor = min((size % divisor, divisor)
                                                   for divisor in range(low, high, step))
            return best_divisor

        def get_groups_count(num_channels, divider_coef=2):
            groups = math.sqrt(divider_coef * num_channels)
            groups_to_power_of_divider = divider_coef ** math.floor(math.log(groups) / math.log(divider_coef))
            if num_channels % groups_to_power_of_divider != 0:
                groups_to_power_of_divider = find_best_divisor(num_channels, groups_to_power_of_divider // 2,
                                                               groups_to_power_of_divider * 2)
            return groups_to_power_of_divider

        super().__init__(get_groups_count(num_channels), num_channels, eps, affine)


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

# def knn(x, k):
#     x = x.transpose(1, 2)
#     dist = torch.cdist(x, x)
#     idx = dist.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx



def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    with torch.no_grad():
        if idx is None:
            if dim9 == False:
                idx = knn(x, k=k)  # (batch_size, num_points, k)
            else:
                idx = knn(x[:, 6:], k=k)
        device = torch.device('cuda')

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1,
                     padding=0, groups=groups, bias=False)

class BasicBlockGN(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=True):
        super(BasicBlockGN, self).__init__()
        if norm_layer:
            norm_layer = lambda x: GroupNorm(x)
        else:
            norm_layer = Identity
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.downsample = downsample

        self.inplanes = inplanes
        self.planes = planes

        if self.downsample is None and inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion),
                #nn.InstanceNorm1d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        if self.conv1.weight.device != x.device:
            raise Exception(f"{self.conv1.weight.device}, {x.device}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)

        return out


class DGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40, pretrained=False):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn5 = GroupNorm(args.emb_dims)
        self.conv1 = BasicBlockGN(6, 64)
        self.conv2 = BasicBlockGN(64 * 2, 64)
        self.conv3 = BasicBlockGN(64 * 2, 128)
        self.conv4 = BasicBlockGN(128 * 2, 256)
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = GroupNorm(512)
        self.dp1 = lambda x: x #nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = GroupNorm(256)
        self.dp2 = lambda x: x  #nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        if pretrained:
            print('load from ', pretrained)
            ckpt = torch.load(pretrained, map_location='cpu')
            # del ckpt['linear3.weight']
            # del ckpt['linear3.bias']
            self.load_state_dict(ckpt, strict=False)


    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x, 0




class DGCNN_cls_old(nn.Module):
    def __init__(self, args, output_channels=40, pretrained=False):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = GroupNorm(64)
        self.bn2 = GroupNorm(64)
        self.bn3 = GroupNorm(128)
        self.bn4 = GroupNorm(256)
        self.bn5 = GroupNorm(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = GroupNorm(512)
        self.dp1 = lambda x: x #nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = GroupNorm(256)
        self.dp2 = lambda x: x  #nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

        if pretrained:
            print('load from ', pretrained)
            ckpt = torch.load(pretrained, map_location='cpu')
            # del ckpt['linear3.weight']
            # del ckpt['linear3.bias']
            self.load_state_dict(ckpt, strict=False)


    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.conv5(x)  # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x, 0