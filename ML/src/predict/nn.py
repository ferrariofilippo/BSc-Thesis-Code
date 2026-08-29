from .mlengine import IMLEngine

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .neuralnetwork.engine import evaluate, train_one_epoch
from .neuralnetwork.model import NeuralNetworkModel

log = logging.getLogger(__name__)

class NeuralNetwork(IMLEngine):
    PREDICTION_MODEL_PATH = "./conf/tolerance-prediction/nn.pt"
    
    DO_LOG = False

    def __init__(self, mode: str = "Predict"):
        super().__init__(mode)
        
    def _build_optimizer(self, cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
        if cfg.training.optimizer == "adam":
            return optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        elif cfg.training.optimizer == "sgd":
            return optim.SGD(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        elif cfg.training.optimizer == "adamw":
            return optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")

    def _pick_device(self) -> torch.device:
        if not torch.cuda.is_available():
            if self.DO_LOG:
                log.warning("CUDA not available, falling back to CPU.")
            return torch.device("cpu")

        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        if self.DO_LOG:
            log.info(f"Sweep job #{0} -> GPU {gpu_id}")
        return torch.device(f"cuda:{gpu_id}")
    
    def _get_model(self, cfg: DictConfig, device) -> NeuralNetworkModel:
        return NeuralNetworkModel(
            input_dim=cfg.model.input_dim,
            hidden_1_dim=cfg.model.hidden_1_dim,
            hidden_2_dim=cfg.model.hidden_2_dim,
            hidden_3_dim=cfg.model.hidden_3_dim,
            output_dim=cfg.model.output_dim,
            hidden_activation=cfg.model.hidden_activation,
            output_activation=cfg.model.output_activation,
        ).to(device)

    def predict(self, inputs) -> float:
        device = self._pick_device()
        ckpt = torch.load(self.PREDICTION_MODEL_PATH, map_location=device, weights_only=False)
        self._load_standardizations()

        self.model = self._get_model(ckpt["model_cfg"], device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        stdized = (inputs - self.means[1:]) / self.stds[1:]
        with torch.no_grad():
            x = torch.as_tensor(stdized, dtype=torch.float32, device=device) 
            tol = self.model.forward(x)

        return tol.cpu().numpy()[0] * self.stds[0] + self.means[0]
        
    def train(self, cfg: DictConfig) -> float:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        device = self._pick_device()
        if self.DO_LOG:
            log.info(f"Using device: {device}")

        run_dir = HydraConfig.get().runtime.output_dir

        train_loader, val_loader = self._build_dataloaders(cfg.data, cfg.seed)

        model = self._get_model(cfg, device)

        loss_fn = nn.MSELoss()
        optimizer = self._build_optimizer(cfg, model)

        best_val = float("inf")
        patience_counter = 0
        ckpt_path = os.path.join(run_dir, "best_model.pt")

        train_loss_history = []
        val_loss_history = []

        for epoch in range(cfg.training.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
            val_loss = evaluate(model, val_loader, device, loss_fn)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if self.DO_LOG and epoch % cfg.training.log_every == 0 or epoch == cfg.training.epochs - 1:
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
                    if self.DO_LOG:
                        log.info(f"Early stopping at epoch {epoch} (best val_loss={best_val:.6f})")
                    break

        self._save_loss_plot(train_loss_history, val_loss_history, run_dir)

        cfg_save_path = os.path.join(run_dir, "resolved_config.yaml")
        with open(cfg_save_path, "w") as f:
            OmegaConf.save(config=cfg, f=f, resolve=True)

        log.info(f"Done. Best val_loss={best_val:.6f}. Checkpoint: {ckpt_path}")

        return best_val
