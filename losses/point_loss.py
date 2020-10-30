import torch
import torch.nn as nn
import kaolin as kal

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
        assert (x_size[2] == y_size[2] == 3)

        chamfer_loss = []

        for i in range(x_size[0]):

            chamfer_loss.append(kal.metrics.point.chamfer_distance(x[i],y[i]))
        chamfer_loss = torch.stack(chamfer_loss).mean()

        return chamfer_loss