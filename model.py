import torch.nn as nn
import torch
#torch 用于张量操作、softmax 等

#定义卷积块
class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.BatchNorm2d(in_c),                       # 对输入进行批归一化，加快训练速度、提升稳定性
            nn.Conv2d(in_c, out_c, 3, padding=1),       # 卷积核大小为3×3，输入通道in_c，输出通道out_c，保持尺寸不变
            nn.ReLU(inplace=True),                      # ReLU 激活函数，增加非线性
            nn.AvgPool2d(2)                             # 平均池化，窗口大小为2×2，将特征图尺寸减半
        )

#CNN特征提取模块.把MFCC特征图，经过一系列卷积+池化，最终转化为一个固定长度的特征向量[B, 512]，作为模型的中间表示（embedding），给后面的MoE分类器使用
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),               # 输入1通道（MFCC图），输出64通道，40×81 → 20×40
            ConvBlock(64, 128),             # 输出128通道，20×40 → 10×20
            ConvBlock(128, 256),            # 输出256通道，10×20 → 5×10
            ConvBlock(256, 256),            # 加强非线性和语义提取能力,空间尺寸会继续减半,输出一个256通道的卷积块，5×10 → 2×5
            ConvBlock(256, 512),            # 输出512通道，2×5 → 1×2
            ConvBlock(512, 512)             # 再一个512通道卷积块，1×2 → 1×1
        )
        #：自适应地将任意尺寸的特征图，平均池化成1×1的特征图。去掉了全连接层对尺寸的依赖，大大减少了参数量， 保留了每个通道的全局信息。
        self.gap = nn.AdaptiveAvgPool2d(1)             # 全局平均池化，将每个通道池化为一个值，输出 shape: (B, C, 1, 1)

    #将输入的MFCC特征图（1×40×81）通过CNN提取成一个512维的特征向量，[1, 40, 81] → [512] ，为后续分类器MoE使用
    def forward(self, x):
        x = self.features(x)                           # 输入经过6个卷积块
        return self.gap(x).view(x.size(0), -1)         # 展平成向量[512]，作为特征嵌入输出

class MoEHead(nn.Module):
    # emb_dim=512: 输入特征维度（来自 CNN）
    # num_classes=5: 输出类别数量（如：normal、wheeze、crackle、both、other）
    # k=10: 使用的专家个数（每个专家是一个独立的 MLP）
    def __init__(self, emb_dim=512, num_classes=5, k=10):
        #调用父类的构造函数，不写很多PyTorch机制将无法正常工作（如参数注册、模型保存、反向传播等）。
        super().__init__()
        #nn.ModuleList([...])是PyTorch提供的模块列表容器，保存多个子模型（多个专家）。相比普通Python list，ModuleList 是“可训练”的
        self.experts = nn.ModuleList([
            # nn.Linear(emb_dim, num_classes)是一个全连接层,接收一个长度为emb_dim=512的向量，输出一个长度为num_classes=5的向量
            # nn.ReLU(inplace=True)，ReLU是一个非线性激活函数，它可以让模型学到复杂的决策边界。inplace = True，表示不新建内存，加快计算
            #nn.Sequential(...)是PyTorch提供的模块容器，它把多个层“顺序组合”成一个整体，表示这两个层会按顺序执行：input → Linear → ReLU → output
            nn.Sequential(nn.Linear(emb_dim, num_classes), nn.ReLU(inplace=True))
            # 构造k个专家网络，都是Linear→ReLU的结构。每次循环，创建一个新的专家（结构相同，参数独立）
            for _ in range(k)
        ])
        #网络为每个样本产生k个专家权重，[B,512] → [B,10]
        self.gate = nn.Linear(emb_dim, k)

    #前向传播函数，emb:是来自CNN的特征向量
    def forward(self, emb):
        #self.gate(emb)，[B,512] → [B,10]。通过softmax将专家的分数变成概率分布，所有专家权重之和为1
        g = torch.softmax(self.gate(emb), dim=1)
        #门控专家模型（MoE）中核心的一步，让所有专家分别对输入emb做预测，然后把每个专家的输出按维度堆叠起来，得到三维张量e[B, C, K]
        e = torch.stack([exp(emb) for exp in self.experts], dim=2)
        #对所有专家预测结果的加权融合
        return torch.sum(e * g.unsqueeze(1), dim=2)

#完整模型结构
class CNN_MoE(nn.Module):
    #初始化函数，构建模型结构。5种分类，10个专家
    def __init__(self, num_classes=5, k=10):
        #初始化父类（nn.Module），这是 PyTorch 的固定写法
        super().__init__()
        #创建一个CNN主体CNNBackbone，输入是[B,1,40,81]（即MFCC特征图），输出是[B,512]的特征向量。
        self.backbone = CNNBackbone()
        #创建MoE头部，输入是512维特征向量，输出是5类的打分。
        self.head = MoEHead(512, num_classes, k)

    #x是输入（MFCC特征图），shape通常为[B,1,40,81]
    def forward(self, x):
        #把图像送进 CNN 提取特征，输出 [B, 512]
        emb = self.backbone(x)
        #把512维特征送进MoE分类头，得到最终分类结果[B,5]
        return self.head(emb)
