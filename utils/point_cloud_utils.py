import numpy as np
import torch


def get_offsets(point_cloud, batch_size=None):
    if batch_size is None:
        return np.random.randn(*point_cloud.shape).astype(np.float32)
    else:
        return np.random.randn(batch_size, *point_cloud.shape).astype(np.float32)


def get_sigmas(sigmas, batch_size):
        labels = np.random.choice(sigmas, batch_size)
        return labels


def get_prior(batch_size, num_points, inp_dim):
    # -1 to 1, uniform
    return (torch.rand(batch_size, num_points, inp_dim) * 2 - 1.) * 1.5
