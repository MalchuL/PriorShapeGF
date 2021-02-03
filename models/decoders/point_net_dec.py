import torch
import torch.nn as nn
import numpy as np


class PointNetDecoder(nn.Module):
    def __init__(self, input_features, output_points):
        super(PointNetDecoder, self).__init__()
        self.mlp1 = nn.Linear(input_features, 512)
        self.mlp2 = nn.Linear(512, 512)
        self.mlp3 = nn.Linear(512, 512)

        self.bn1 = nn.BatchNorm1d(512)
        self.mlp4 = nn.Linear(512 + input_features, 512)
        self.mlp5 = nn.Linear(512, 512)
        self.mlp6 = nn.Linear(512, output_points * 3)
        self.output_points = output_points

        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,514,45^2
        input = x
        x = self.relu(self.mlp1(x))
        x = self.relu(self.mlp2(x))

        x = self.relu(self.bn1(self.mlp3(x)))
        x = torch.cat([x, input], dim=1)

        x = self.relu(self.mlp4(x))
        x = self.relu(self.mlp5(x))
        x = self.mlp6(x)

        x = x.view(-1, self.output_points, 3)

        return x, x, x

