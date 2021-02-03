import torch
import torch.nn as nn
import kaolin as kal
from kaolin.metrics import directed_distance


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