import registry.registries as registry
from .point_loss import MaxPointLoss, RMSEPointLoss, ChamferDistance, ShapeGFPointLoss


registry.Criterion(MaxPointLoss)
registry.Criterion(RMSEPointLoss)
registry.Criterion(ChamferDistance)
registry.Criterion(ShapeGFPointLoss)