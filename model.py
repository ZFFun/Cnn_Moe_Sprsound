import torch
import torch.nn as nn

# 定义带 BatchNorm + Conv + ReLU + AvgPool 的标准卷积块
class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.BatchNorm2d(in_c),                   # 批归一化
            nn.Conv2d(in_c, out_c, 3, padding=1),   # 3×3 卷积
            nn.ReLU(inplace=True),                  # 非线性激活
            nn.AvgPool2d(2)                         # H,W 都 /2
        )

# 定义最后一层只做卷积不池化的块，避免把高度降到 0
class ConvNoPool(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
            # **不做 AvgPool2d**，保持特征图尺寸
        )

# CNN 特征提取骨干网络
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),        # 输入1通道（MFCC图），输出64通道，40×162 → 20×80
            ConvBlock(64, 128),      # 64×20×80 → 128×10×20
            ConvBlock(128, 256),     # 128×10×20 → 256×5×10
            ConvBlock(256, 256),     # 256×5×10 → 256×2×5
            ConvBlock(256, 512),     # 256×2×5 → 512×1×2
            ConvNoPool(512, 512)     # 512×1×2 → 512×1×2（不池化）
        )
        # 全局平均池化：任意 H×W → 1×1
        self.gap = nn.AdaptiveAvgPool2d(1)  # 输出 shape: (B,512,1,1)

    def forward(self, x):
        x = self.features(x)            # [B, 1,40,81] → [B,512,1,2]
        x = self.gap(x)                 # → [B,512,1,1]
        return x.view(x.size(0), -1)    # → [B,512]

# Mixture of Experts 头
class MoEHead(nn.Module):
    def __init__(self, emb_dim=512, num_classes=5, k=10):
        super().__init__()  # 初始化父类
        # k 个专家：每个都是 Linear→ReLU
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, num_classes),
                nn.ReLU(inplace=True)
            )
            for _ in range(k)
        ])
        # 门控网络：把 emb [B,512] → gate 分数 [B,k]
        self.gate = nn.Linear(emb_dim, k)

    def forward(self, emb):
        # emb: [B,512] → g: [B,k]（softmax 后）
        g = torch.softmax(self.gate(emb), dim=1)
        # 所有专家并行跑一遍：list of [B,C] → stack → [B,C,K]
        e = torch.stack([exp(emb) for exp in self.experts], dim=2)
        # 按照 gate 权重做加权和：e*[B,1,K] → sum(dim=2) → [B,C]
        return torch.sum(e * g.unsqueeze(1), dim=2)

# 整体模型：CNNBackbone + MoEHead
class CNN_MoE(nn.Module):
    def __init__(self, num_classes=5, k=10):
        super().__init__()                      # 初始化父类
        self.backbone = CNNBackbone()           # CNN 特征提取器
        self.head = MoEHead(512, num_classes, k)  # MoE 分类头

    def forward(self, x):
        emb = self.backbone(x)  # [B,1,40,81] → [B,512]
        return self.head(emb)   # → [B,num_classes]

