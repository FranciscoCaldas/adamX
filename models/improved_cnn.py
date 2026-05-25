import torch
import torch.nn as nn


class ImprovedCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = None
        self.fc2 = None
        self._init_fc(input_size)

    def _forward_features(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        return x

    def _init_fc(self, input_size: int) -> None:
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, input_size, input_size)
            features = self._forward_features(dummy)
            feature_dim = features.view(1, -1).size(1)

        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
