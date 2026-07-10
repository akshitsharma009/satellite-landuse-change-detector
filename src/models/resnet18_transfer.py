import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes=10, pretrained=True):
    """
    Build a ResNet18 model for transfer learning.
    """

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model