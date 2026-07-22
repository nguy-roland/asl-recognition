import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes=29, unfreeze_from=None):
    """
    unfreeze_from: None = everything frozen except the head.
                   'layer4' = layer4 + head trainable.
                   'layer3' = layer3, layer4 + head trainable.
                   'all' = everything trainable.
    """
    model = resnet18(weights='IMAGENET1K_V1')

    for param in model.parameters():
        param.requires_grad = False
    
    if unfreeze_from == 'all':
        for param in model.parameters():
            param.requires_grad = True
    elif unfreeze_from == 'layer4':
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif unfreeze_from == 'layer3':
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model