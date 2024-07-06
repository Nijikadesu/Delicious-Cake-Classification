import torch.nn as nn
from torchvision import models


def ResNet50FT(classes):
    model = models.resnet50(pretrained=True)
    num_nodes = model.fc.in_features
    model.fc = nn.Linear(num_nodes, classes)
    return model
