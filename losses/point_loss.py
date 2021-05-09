import torch
import torch.nn as nn
import kaolin as kal
from kaolin.metrics import directed_distance, SidedDistance
import emd

from losses.emd.emd_module import emdModule


class RMSEPointLoss(nn.Module):
    def __init__(self, eps=0.0):
        super().__init__()
        self.eps = torch.ones(1) * float(eps)
        self.use_eps = eps > 0.0

    def forward(self, pred, target):
        assert len(pred.size()) == 3
        if self.use_eps:
            loss = torch.max((pred - target).abs(), self.eps.to(pred.device))[0].pow(2).sum(1).sqrt().mean()
        else:
            loss = (pred - target).pow(2).sum(1).sqrt().mean()
        return loss


class MaxPointLoss(nn.Module):
    def __init__(self, batchwise_weight=1):
        super().__init__()
        self.batchwise_weight = batchwise_weight

    def forward(self, pred, target):
        assert len(pred.size()) == 3
        pointwise_loss = (pred - target).pow(2).max(1)[0]
        batch_loss = pointwise_loss.max(1)[0].mean()
        pointwise_loss = pointwise_loss.mean()
        return (batch_loss * self.batchwise_weight + pointwise_loss)


class ShapeGFPointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, pred_offset, offset, sigmas):
        assert len(pred_offset.size()) == 3
        B, C, H = pred_offset.shape
        assert C == 3
        sigmas = sigmas[:, 0:1].view(B, 1, 1)
        target = - offset[..., 0] * sigmas
        lambda_sigma = 1. / sigmas
        loss = 0.5 * (self.loss(pred_offset, target) * lambda_sigma).sum(dim=1).mean()

        return loss

class ChamferDistance(nn.Module):
    def forward(self, x, y):  # for example, x = batch,M,3 y = batch,M,3
        #   compute chamfer distance between tow point clouds x and y

        x_size = x.size()
        y_size = y.size()
        assert (x_size[0] == y_size[0])
        assert (int(x_size[2]) == int(y_size[2]) == 3)

        chamfer_loss = []

        for i in range(x_size[0]):

            chamfer_loss.append(kal.metrics.point.chamfer_distance(x[i],y[i]))
        chamfer_loss = torch.stack(chamfer_loss).mean()

        return chamfer_loss


class MaxChamferDistance(nn.Module):

    def __init__(self, Z_index=0.5):
        super().__init__()
        self.Z_index = Z_index

    def mask_by_Z(self, x, Z_index):
        if Z_index is None:
            return x
        with torch.no_grad():
            std, mean = torch.std_mean(x)
            x_minus_mean = x - mean
            mask = x_minus_mean > -Z_index * std
        return x[mask]

    def chamfer_distance(self, S1: torch.Tensor, S2: torch.Tensor,
                         w1: float = 1., w2: float = 1.):
        # Nx3
        assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
        assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

        dist_to_S2 = directed_distance(S1, S2, mean=False)
        dist_to_S1 = directed_distance(S2, S1, mean=False)

        dist_to_S2 = self.mask_by_Z(dist_to_S2, self.Z_index)
        dist_to_S1 = self.mask_by_Z(dist_to_S1, self.Z_index)
        #print(dist_to_S1.shape, dist_to_S2.shape)

        return dist_to_S2.mean(), dist_to_S1.mean()

    def forward(self, x, y):  # for example, x = batch,M,3 y = batch,M,3
        #   compute chamfer distance between tow point clouds x and y

        x_size = x.size()
        y_size = y.size()
        assert (x_size[0] == y_size[0])
        assert (int(x_size[2]) == int(y_size[2]) == 3)

        chamfer_loss = []

        for i in range(x_size[0]):
            chamfer_loss.append(sum(self.chamfer_distance(x[i], y[i])))
        chamfer_loss = torch.stack(chamfer_loss).mean()

        return chamfer_loss



class MaxDistance(nn.Module):



    def chamfer_distance(self, S1: torch.Tensor, S2: torch.Tensor,
                         w1: float = 1., w2: float = 1.):
        # Nx3
        assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
        assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

        dist_to_S2 = directed_distance(S1, S2, mean=False)
        dist_to_S1 = directed_distance(S2, S1, mean=False)


        return torch.sqrt(torch.max(dist_to_S2)), torch.sqrt(torch.max(dist_to_S1))

    def forward(self, x, y):  # for example, x = batch,M,3 y = batch,M,3


        max_loss = []
        x_size = x.size()
        for i in range(x_size[0]):
            max_loss.append(max(self.chamfer_distance(x[i], y[i])))
        max_loss = torch.stack(max_loss).mean()

        return max_loss


def directed_sigma(S1: torch.Tensor, S2: torch.Tensor, mean: bool = True):
    r"""Computes the average distance from point cloud S1 to point cloud S2

    Args:
            S1 (torch.Tensor): point cloud
            S2 (torch.Tensor): point cloud
            mean (bool): if the distances should be reduced to the average

    Returns:
            torch.Tensor: ditance from point cloud S1 to point cloud S2

    Args:

    Example:
            >>> A = torch.rand(300,3)
            >>> B = torch.rand(200,3)
            >>> >>> directed_distance(A,B)
            tensor(0.1868)

    """

    if S1.is_cuda and S2.is_cuda:
        sided_minimum_dist = SidedDistance()
        closest_index_in_S2 = sided_minimum_dist(
            S1.unsqueeze(0), S2.unsqueeze(0))[0]
        closest_S2 = torch.index_select(S2, 0, closest_index_in_S2)

    else:
        from time import time
        closest_index_in_S2 = nnsearch(S1, S2)
        closest_S2 = S2[closest_index_in_S2]

    dist_to_S2 = S1 - closest_S2
    if mean:
        dist_to_S2 = dist_to_S2.mean()

    return dist_to_S2

class SigmaDistance(nn.Module):

    def sigma_distance(self, S1: torch.Tensor, S2: torch.Tensor,
                       w1: float = 1., w2: float = 1.):
        # Nx3
        assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
        assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

        dist_to_S2 = directed_sigma(S1, S2, mean=False)
        dist_to_S1 = directed_sigma(S2, S1, mean=False)

        std_S1 = torch.std(dist_to_S2)
        std_S2 = torch.std(dist_to_S1)

        # Returns in MSE value
        return std_S1, std_S2

    def forward(self, x, y):  # for example, x = batch,M,3 y = batch,M,3

        max_loss = []
        x_size = x.size()
        for i in range(x_size[0]):
            max_loss.append(max(self.sigma_distance(x[i], y[i])))

        max_loss = torch.stack(max_loss).mean()

        return max_loss

class RobustSigmaDistance(nn.Module):

    def mask_by_Z(self, x, q_begin=(0.05, 0.95), q_end=(0.25, 0.75)):
        with torch.no_grad():
            q = torch.tensor(q_begin + q_end).type_as(x)
            quantiles = torch.quantile(x, q)
            mask_begin = (x < quantiles[0]) | (x > quantiles[1])
            mask_end = (x > quantiles[2]) & (x < quantiles[3])

        return x[mask_begin], x[mask_end]

    def __init__(self, q_begin=(0.05, 0.95), q_end=(0.25, 0.75), min_elements=20):
        super(RobustSigmaDistance, self).__init__()
        self.min_elements = min_elements
        self.q_begin = q_begin
        self.q_end = q_end

    def sigma_distance(self, S1: torch.Tensor, S2: torch.Tensor):
        # Nx3
        assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
        assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

        dist_to_S2 = directed_sigma(S1, S2, mean=False)
        dist_to_S1 = directed_sigma(S2, S1, mean=False)

        S1_mask_begin, S1_mask_end = self.mask_by_Z(dist_to_S2, q_begin=self.q_begin, q_end=self.q_end)
        S2_mask_begin, S2_mask_end = self.mask_by_Z(dist_to_S1, q_begin=self.q_begin, q_end=self.q_end)

        begin = max(torch.std(S1_mask_begin), torch.std(S2_mask_begin))
        end = max(torch.std(S1_mask_end), torch.std(S2_mask_end))

        # Returns in MSE value
        return begin, end

    def forward(self, x, y):  # for example, x = batch,M,3 y = batch,M,3

        max_loss = []
        min_loss = []
        x_size = x.size()
        for i in range(x_size[0]):
            begin, end = self.sigma_distance(x[i], y[i])
            max_loss.append(begin)
            min_loss.append(end)

        max_loss = torch.stack(max_loss).mean()
        min_loss = torch.stack(min_loss).mean()

        return max_loss, min_loss





class EMDLoss(nn.Module):
    def __init__(self, iters=50, eps=0.005): # As in https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd
        super(EMDLoss, self).__init__()
        self.iters = iters
        self.eps = eps
        self.loss = emdModule()

    def forward(self, input1, input2):
        return self.loss(input1, input2, self.eps, self.iters)[0].mean()


def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]
