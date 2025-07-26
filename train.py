import torch

# 训练函数，每次调用训练一轮（模型，训练集DataLoader，损失函数，优化器，模型运行的设备）
def train_epoch(model, loader, crit, opt, device):
    # 设置为训练模式，初始化本轮损失
    model.train(); total_loss = 0
    # 遍历DataLoader，每次得到一个batch的spec音频的特征图（[B,1,40,81]），label对应标签（整数：0~4）。
    for spec, label in loader:
        # 把数据移动到CPU或GPU上
        spec, label = spec.to(device), label.to(device)
        # 每个batch都要先清空梯度，否则梯度会累加
        opt.zero_grad()
        # 把音频特征输入模型，得到输出logits，形状为[B,num_classes]
        logits = model(spec)
        # 计算预测结果与真实标签的交叉熵损失
        loss = crit(logits, label)
        # loss.backward()：反向传播，计算梯度；opt.step()：更新模型权重
        loss.backward(); opt.step()
        # 累加batch的总损失。乘上样本数，是为了后面计算平均值
        total_loss += loss.item() * spec.size(0)
    # 返回本轮训练的平均损失
    return total_loss / len(loader.dataset)

# @torch.no_grad() 表示验证时关闭梯度计算，节省显存
# 和训练函数结构类似，但没有反向传播
@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    # model.eval()：切换到评估模式。初始化损失与准确数
    model.eval(); total_loss = 0; correct = 0
    for spec, label in loader:
        # 遍历验证集，每次处理一个batch
        spec, label = spec.to(device), label.to(device)
        # 模型预测logits，形状是[B,num_classes]
        logits = model(spec)
        # 累加每个样本的损失
        total_loss += crit(logits, label).item() * spec.size(0)
        # 比较模型预测的标签logits.argmax(1)和真实标签label。统计正确预测的数量
        correct += (logits.argmax(1) == label).sum().item()
    # 平均验证损失 、 准确率（正确数/总数）
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

