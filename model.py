import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.groups = min(4, channels)

        self.conv1x1 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=1
        )

        self.conv3x3 = nn.Conv2d(
            channels // self.groups,
            channels // self.groups,
            kernel_size=3,
            padding=1
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        b, c, h, w = x.size()

        x_group = x.view(
            b * self.groups,
            -1,
            h,
            w
        )

        xn = self.avg_pool(x_group)

        x1 = self.conv1x1(xn)
        x2 = self.conv3x3(xn)

        out = torch.sigmoid(x1 + x2)

        return (x_group * out).view(b, c, h, w)


class CattleMuzzleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            EMA(16),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            EMA(32),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        self.embedding = nn.Sequential(

            nn.Linear(64, 256),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.35),

            nn.Linear(256, 256)
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.embedding(x)

        return F.normalize(
            x,
            p=2,
            dim=1
        )