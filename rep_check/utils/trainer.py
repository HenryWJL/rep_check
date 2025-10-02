import os
import hydra
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from rep_check.utils.logger import create_logger
from rep_check.utils.visualizer import plot_curves


class Trainer:

    def __init__(self, cfg: OmegaConf) -> None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.logger = create_logger()
        self.device = torch.device(cfg.device)
        self.num_epochs = cfg.num_epochs
        self.val_freq = cfg.val_freq
        self.model = hydra.utils.instantiate(cfg.model)
        self.model.to(self.device)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_dataloader = hydra.utils.instantiate(cfg.train_dataloader)
        self.val_dataloader = None if cfg.val_dataloader is None \
            else hydra.utils.instantiate(cfg.val_dataloader)

    def run(self) -> None:
        ckpt_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        for epoch in tqdm(range(1, self.num_epochs + 1), desc="Model Training"):
            message = f"Epoch [{epoch}/{self.num_epochs}]\n"
            self.model.train()
            total_loss, correct, total = 0.0, 0.0, 0.0
            for (poses, labels) in self.train_dataloader:
                poses, labels = poses.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(poses)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * poses.size(0)
                pred_labels = logits.max(dim=1)[1]
                correct += pred_labels.eq(labels).sum().item()
                total += poses.size(0)

            avg_loss = total_loss / total
            accuracy = 100.0 * correct / total
            train_loss.append(avg_loss)
            train_acc.append(accuracy)
            message += f"Train Loss: {avg_loss} | Train Accuracy: {accuracy}\n"
            
            if self.val_dataloader is not None and epoch % self.val_freq == 0:
                self.model.eval()
                total_loss, correct, total = 0.0, 0.0, 0.0
                with torch.no_grad():
                    for poses, labels in self.val_dataloader:
                        poses, labels = poses.to(self.device), labels.to(self.device)
                        logits = self.model(poses)
                        loss = self.criterion(logits, labels)

                        total_loss += loss.item() * poses.size(0)
                        pred_labels = logits.max(dim=1)[1]
                        correct += pred_labels.eq(labels).sum().item()
                        total += poses.size(0)
                avg_loss = total_loss / total
                accuracy = 100.0 * correct / total
                val_loss.append(avg_loss)
                val_acc.append(accuracy)
                message += f"Validation Loss: {avg_loss} | Validation Accuracy: {accuracy}\n"
                # Save checkpoints
                ckpt_path = os.path.join(ckpt_dir, f"epoch={epoch}-val_acc={accuracy}.pth")
                torch.save(self.model.state_dict(), ckpt_path)
            # Logging
            self.logger.info(message)
        
        # Plot loss and accuracy curves
        plot_curves(self.num_epochs, train_loss, train_acc)
        if val_loss and val_acc:
            plot_curves(self.num_epochs, val_loss, val_acc, train=False)
