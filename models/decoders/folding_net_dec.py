import torch
import torch.nn as nn
import numpy as np
import math
from functools import lru_cache as cache

class FoldingNetDecFold(nn.Module):
    def __init__(self, input_features, point_dim=[2,3]):
        super(FoldingNetDecFold, self).__init__()
        self.conv1 = nn.Conv1d(input_features + point_dim[0], 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, point_dim[1], 1)

        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,514,45^2
        x = self.relu(self.conv1(x))  # x = batch,512,45^2
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x



def GridSamplingLayer(batch_size, meshgrid):
    '''
    output Grid points as a NxD matrix
    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    '''

    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)

    return g


class FoldingNetDec(nn.Module):
    def __init__(self, input_features):
        super(FoldingNetDec, self).__init__()
        self.fold1 = FoldingNetDecFold(input_features, [2,3])
        self.fold2 = FoldingNetDecFold(input_features, [3,3])

    def forward(self, x):  # input x = batch, 512
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, 45 ** 2, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2

        meshgrid = [[-1.0, 1.0, 45], [-1.0, 1.0, 45]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid)

        if x.is_cuda:
            grid = grid.cuda()

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2

        x = self.fold1(x)  # x = batch,3,45^2
        p1 = x  # to observe

        x = torch.cat((code, x), 1)  # x = batch,515,45^2

        x = self.fold2(x)  # x = batch,3,45^2
        x = x.transpose(2, 1)  # x = batch,45^2,3
        return x

class FoldingNetDec3d(nn.Module):
    def __init__(self, input_features):
        super(FoldingNetDec3d, self).__init__()
        self.fold1 = FoldingNetDecFold(input_features, [3,3])
        self.fold2 = FoldingNetDecFold(input_features, [3,3])

    def forward(self, x):  # input x = batch, 512
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, 13 ** 3, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2

        meshgrid = [[-1.0, 1.0, 13], [-1.0, 1.0, 13], [-1.0, 1.0, 13]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,13^3,2
        grid = torch.from_numpy(grid)

        if x.is_cuda:
            grid = grid.cuda()

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2

        x = self.fold1(x)  # x = batch,3,45^2
        p1 = x  # to observe

        x = torch.cat((code, x), 1)  # x = batch,515,45^2

        x = self.fold2(x)  # x = batch,3,45^2
        x = x.transpose(2, 1)  # x = batch,45^2,3
        return x



@cache(10)
def fibonacci_sphere(batch_size, samples=1):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    points = np.array(points, dtype=np.float32)
    points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

    return points


class FoldingNetDec3dSphere(nn.Module):
    def __init__(self, input_features):
        super(FoldingNetDec3dSphere, self).__init__()
        self.fold1 = FoldingNetDecFold(input_features, [3,3])
        self.fold2 = FoldingNetDecFold(input_features, [3,3])

    def forward(self, x):  # input x = batch, 512
        samples = 2048
        batch_size = x.size(0)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, samples, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2

        grid = fibonacci_sphere(batch_size, samples)  # grid = batch,13^3,2
        grid = torch.from_numpy(grid)

        if x.is_cuda:
            grid = grid.cuda()

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2

        x = self.fold1(x)  # x = batch,3,45^2
        p1 = x  # to observe

        x = torch.cat((code, x), 1)  # x = batch,515,45^2

        x = self.fold2(x)  # x = batch,3,45^2
        x = x.transpose(2, 1)  # x = batch,45^2,3
        return x