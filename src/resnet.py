import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes=29, freeze_backbone=True):
    model = resnet18(weights='IMAGENET1K_V1')

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model