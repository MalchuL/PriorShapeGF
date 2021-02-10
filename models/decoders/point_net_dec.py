import torch
import torch.nn as nn
import numpy as np


class PointNetDecoder(nn.Module):
    def __init__(self, input_features, output_points, hidden_size=512):
        super(PointNetDecoder, self).__init__()
        self.mlp1 = nn.Linear(input_features, hidden_size)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.mlp3 = nn.Linear(hidden_size, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.mlp4 = nn.Linear(hidden_size + input_features, hidden_size)
        self.mlp5 = nn.Linear(hidden_size, hidden_size)
        self.mlp6 = nn.Linear(hidden_size, output_points * 3)
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

