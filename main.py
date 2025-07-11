import torch
from torch.utils.data import DataLoader
from dataset import SPRSoundDataset
from model import CNN_MoE
from train import train_epoch, eval_epoch
from config import CFG
from logger import init_logger

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = init_logger()

    train_set = SPRSoundDataset(CFG.data_dir, 'train')
    val_set = SPRSoundDataset(CFG.data_dir, 'val')
    tr_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=4)
    va_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False)

    model = CNN_MoE().to(device)
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.epochs)

    best_acc = 0
    for ep in range(1, CFG.epochs + 1):
        tr_loss = train_epoch(model, tr_loader, crit, opt, device)
        va_loss, va_acc = eval_epoch(model, va_loader, crit, device)
        sched.step()
        logger.info(f"[E{ep}] Train Loss: {tr_loss:.3f} | Val Loss: {va_loss:.3f} | Val Acc: {va_acc:.2%}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), 'cnn_moe_sprsound_best.pth')
    logger.info(f"Best Validation Accuracy: {best_acc:.2%}")

if __name__ == '__main__':
    main()
