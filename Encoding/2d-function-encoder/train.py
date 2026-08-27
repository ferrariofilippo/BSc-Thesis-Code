import logging
import os
from sys import path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

path.append(os.path.join(os.path.dirname(__file__), '..', 'shared/neural-network'))

from dataset import build_dataloaders
from engine import evaluate, train_one_epoch
from model import Autoencoder

log = logging.getLogger(__name__)

DO_LOG = False

def pick_device() -> torch.device:
    if not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU.")
        return torch.device("cpu")

    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    if DO_LOG:
        log.info(f"Sweep job #{0} -> GPU {gpu_id}")
    return torch.device(f"cuda:{gpu_id}")

def build_optimizer(cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
    if cfg.training.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

def save_loss_plot(
    train_history: list, val_history: list, run_dir: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    plt.figure(figsize=(10, 6))
    plt.plot(train_history, label="Train Loss", color="#ff8945")
    plt.plot(val_history, label="Validation Loss", color="#006064")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plot_path = os.path.join(run_dir, "loss_curves.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device()
    if DO_LOG:
        log.info(f"Using device: {device}")

    run_dir = HydraConfig.get().runtime.output_dir
    if DO_LOG:
        log.info(f"Run directory: {run_dir}")

    train_loader, val_loader = build_dataloaders(cfg.data, cfg.seed, cfg.model.input_dim)

    model = Autoencoder(
        input_dim=cfg.model.input_dim,
        hidden_1_dim=cfg.model.hidden_1_dim,
        hidden_2_dim=cfg.model.hidden_2_dim,
        latent_dim=cfg.model.latent_dim,
        hidden_activation=cfg.model.hidden_activation,
        latent_activation=cfg.model.latent_activation,
        output_activation=cfg.model.output_activation,
        dropout_1=.0,
        dropout_2=.0,
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = build_optimizer(cfg, model)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    best_val = float("inf")
    patience_counter = 0
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    train_loss_history = []
    val_loss_history = []

    for epoch in range(cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn, True)
        val_loss = evaluate(model, val_loader, device, loss_fn)

        scheduler.step()

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if DO_LOG and epoch % cfg.training.log_every == 0 or epoch == cfg.training.epochs - 1:
            log.info(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_cfg": cfg.model,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                log.info(f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
                if DO_LOG:
                    log.info(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                break

    save_loss_plot(train_loss_history, val_loss_history, run_dir)

    cfg_save_path = os.path.join(run_dir, "resolved_config.yaml")
    with open(cfg_save_path, "w") as f:
        OmegaConf.save(config=cfg, f=f, resolve=True)

    log.info(f"Done. Best val_loss={best_val:.6f}. Checkpoint: {ckpt_path}")
    return best_val

if __name__ == "__main__":
    main()
