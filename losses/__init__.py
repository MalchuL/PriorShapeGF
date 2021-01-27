import registry.registries as registry
from .point_loss import MaxPointLoss, RMSEPointLoss, ChamferDistance, ShapeGFPointLoss, MaxChamferDistance

registry.Criterion(MaxPointLoss)
registry.Criterion(RMSEPointLoss)
registry.Criterion(ChamferDistance)
registry.Criterion(ShapeGFPointLoss)
registry.Criterion(MaxChamferDistance)