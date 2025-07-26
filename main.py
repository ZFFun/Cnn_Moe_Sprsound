import torch                                        # PyTorch 基础库
from torch.utils.data import DataLoader             # 用于批量加载数据
from dataset import SPRSoundDataset                 # 自定义的数据集类
from model import CNN_MoE                           # CNN + 门控专家MoE
from train import train_epoch, eval_epoch           # 每轮训练与验证逻辑
from config import CFG                              # 超参数配置类
from logger import init_logger                      # 日志记录器

def main():
    # 检查是否有可用GPU，否则使用 CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 初始化日志打印器（logger）用于控制台输出训练过程信息
    logger = init_logger()
    # 加载训练集与验证集，使用配置文件中指定的路径，通过SPRSoundDataset类读取
    train_set = SPRSoundDataset(CFG.data_dir, 'train')
    val_set = SPRSoundDataset(CFG.data_dir, 'val')
    # 使用PyTorch的DataLoader封装数据集，用于训练过程的批量加载和多线程加速。shuffle=True表示训练集每轮都会打乱，num_workers=4表示用4个进程加载数据
    tr_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    va_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False)
    # 实例化CNN-MoE模型并将其加载到CPU/GPU上
    model = CNN_MoE().to(device)
    # 定义损失函数为交叉熵损失，用于多分类任务
    crit = torch.nn.CrossEntropyLoss()
    # 使用AdamW优化器更新模型参数
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    # 使用余弦退火策略动态调整学习率，T_max为最大周期
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.epochs)

    # 初始化变量，用于记录目前为止验证集上的最佳准确率
    best_acc = 0
    # ep为当前轮数
    for ep in range(1, CFG.epochs + 1):
        # 每轮先训练一遍（返回训练损失），再验证一遍（返回验证损失和准确率）
        tr_loss = train_epoch(model, tr_loader, crit, opt, device)
        va_loss, va_acc = eval_epoch(model, va_loader, crit, device)
        # 更新学习率调度器
        sched.step()
        # 打印每轮训练结果（训练损失、验证损失、验证准确率）
        logger.info(f"[E{ep}] Train Loss: {tr_loss:.3f} | Val Loss: {va_loss:.3f} | Val Acc: {va_acc:.2%}")
        # 若当前验证准确率比历史最好值还高，则保存该模型的参数文件
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), 'cnn_moe_sprsound_best.pth')
    # 训练结束后输出最佳验证准确率
    logger.info(f"Best Validation Accuracy: {best_acc:.2%}")

    # 如果该脚本被直接运行，就会调用main()，开始整个训练流程
    if __name__ == '__main__':
        main()
