import torchvision.models as models
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, out_classes):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_classes)

    def forward(self, x):
        return self.model(x), 1