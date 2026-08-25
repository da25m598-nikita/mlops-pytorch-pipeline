import torch.nn as nn
import torchvision.models as models

def get_model(arch, num_classes):
    if arch != "resnet18":
        raise ValueError(f"unknown architecture: {arch}")
    model = models.resnet18(weights=None)
    in_size = model.fc.in_features
    model.fc = nn.Linear(in_size, num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
