import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES


class CNN3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, inputs, return_features=False):
        features = self.features(inputs).flatten(1)
        logits = self.fc(features)

        if return_features:
            return logits, features

        return logits


class ResNetWrapper(nn.Module):
    def __init__(self, name, num_classes=NUM_CLASSES):
        super().__init__()

        network = getattr(models, name)(weights=None)
        input_features = network.fc.in_features
        network.fc = nn.Identity()

        self.backbone = network
        self.fc = nn.Linear(input_features, num_classes)

    def forward(self, inputs, return_features=False):
        features = self.backbone(inputs)
        logits = self.fc(features)

        if return_features:
            return logits, features

        return logits


def build_model(name):
    if name == "cnn3":
        return CNN3()

    if name == "resnet18":
        return ResNetWrapper("resnet18")

    raise ValueError(f"Unknown model name: {name}")