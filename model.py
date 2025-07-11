import torch.nn as nn
import torch

class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        return self.gap(x).view(x.size(0), -1)

class MoEHead(nn.Module):
    def __init__(self, emb_dim=512, num_classes=5, k=10):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(emb_dim, num_classes), nn.ReLU(inplace=True))
            for _ in range(k)
        ])
        self.gate = nn.Linear(emb_dim, k)

    def forward(self, emb):
        g = torch.softmax(self.gate(emb), dim=1)
        e = torch.stack([exp(emb) for exp in self.experts], dim=2)
        return torch.sum(e * g.unsqueeze(1), dim=2)

class CNN_MoE(nn.Module):
    def __init__(self, num_classes=5, k=10):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = MoEHead(512, num_classes, k)

    def forward(self, x):
        emb = self.backbone(x)
        return self.head(emb)
