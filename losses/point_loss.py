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

    def chamfer_distance(self, S1: torch.Tensor, S2: torch.Tensor,
                         w1: float = 1., w2: float = 1.):


        assert (S1.dim() == S2.dim()), 'S1 and S2 must have the same dimesionality'
        assert (S1.dim() == 2), 'the dimensions of the input must be 2 '

        dist_to_S2 = directed_distance(S1, S2)
        dist_to_S1 = directed_distance(S2, S1)


        return dist_to_S2, dist_to_S1

    def forward(self, x, y):  # for example, x = batch,M,3 y = batch,M,3
        #   compute chamfer distance between tow point clouds x and y

        x_size = x.size()
        y_size = y.size()
        assert (x_size[0] == y_size[0])
        assert (int(x_size[2]) == int(y_size[2]) == 3)

        chamfer_loss = []

        for i in range(x_size[0]):

            chamfer_loss.append(max(self.chamfer_distance(x[i],y[i])))
        chamfer_loss = torch.stack(chamfer_loss).mean()

        return chamfer_loss