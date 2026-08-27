import torch

def train_one_epoch(model, loader, optimizer, device, loss_fn, clip_grad=False) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)

        optimizer.step()
        total_loss += loss.item() * batch.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, loss_fn) -> float:
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        recon, _ = model(batch)
        loss = loss_fn(recon, batch)
        total_loss += loss.item() * batch.size(0)
        
    return total_loss / len(loader.dataset)
