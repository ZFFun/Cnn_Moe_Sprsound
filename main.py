import torch
from torch.utils.data import DataLoader
from dataset import SPRSoundDataset      # 自定义的数据集类
from model import CNN_MoE               # 你的模型
from train import train_epoch, eval_epoch
from config import CFG
from logger import init_logger

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = init_logger()

    # 加载数据集
    train_set = SPRSoundDataset(CFG.data_dir, 'train')
    val_set   = SPRSoundDataset(CFG.data_dir, 'val')
    # 5 类
    num_classes = len(train_set.classes)

    tr_loader = DataLoader(train_set, batch_size=CFG.batch_size,
                           shuffle=True, num_workers=4)
    va_loader = DataLoader(val_set,   batch_size=CFG.batch_size,
                           shuffle=False)

    # 初始化模型，num_classes=5
    model = CNN_MoE(num_classes=num_classes).to(device)
    crit  = torch.nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=CFG.lr)

    best_acc = 0
    for ep in range(1, CFG.epochs+1):
        tr_loss    = train_epoch(model, tr_loader, crit, opt, device)
        va_loss, va_acc = eval_epoch(model, va_loader, crit, device)
        logger.info(f"[E{ep}] Train Loss: {tr_loss:.3f} | Val Loss: {va_loss:.3f} | Val Acc: {va_acc:.2%}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), 'cnn_moe_sprsound_best.pth')
    logger.info(f"Best Validation Accuracy: {best_acc:.2%}")

if __name__ == '__main__':
    main()
