import torch

def train_epoch(model, loader, crit, opt, device):
    model.train(); total_loss = 0
    for spec, label in loader:
        spec, label = spec.to(device), label.to(device)
        opt.zero_grad()
        logits = model(spec)
        loss = crit(logits, label)
        loss.backward(); opt.step()
        total_loss += loss.item() * spec.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval(); total_loss = 0; correct = 0
    for spec, label in loader:
        spec, label = spec.to(device), label.to(device)
        logits = model(spec)
        total_loss += crit(logits, label).item() * spec.size(0)
        correct += (logits.argmax(1) == label).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

