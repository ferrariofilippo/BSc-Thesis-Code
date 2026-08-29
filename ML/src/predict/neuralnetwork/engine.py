import torch

def train_one_epoch(model, loader, optimizer, device, loss_fn) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        predicted = model(x)
        loss = loss_fn(predicted, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, loss_fn) -> float:
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        predicted = model(x)
        loss = loss_fn(predicted, y)
        total_loss += loss.item() * x.size(0)
        
    return total_loss / len(loader.dataset)
