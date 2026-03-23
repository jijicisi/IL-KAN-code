import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.pykan.kan import MultKAN  # Ensure this package is installed: pip install MultKAN

# Define model-building function
class RealKANModel(pl.LightningModule):
    def __init__(self, input_size, output_size, focus_indices, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.focus_indices = torch.tensor(focus_indices)

        # Learnable sensitivity weights for skewed hydrological parameters
        min_w, max_w = 1.0, 50.0
        self.sensitive_weights = nn.Parameter(
            torch.full((len(focus_indices),), (min_w + max_w) / 2),
            requires_grad=True
        )

        # Initialize KAN Model
        self.model = MultKAN(
            width=[input_size] + hparams['width'] + [output_size],
            grid=hparams['grid'],
            k=hparams['k'],
            mult_arity=hparams.get('mult_arity', 2)
        )

        self.criterion = nn.MSELoss()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # Neuron "Wake-up" mechanism to prevent dead zones during incremental learning
        if self.training:
            mask = (torch.rand_like(x) < 0.1).float()
            noise = torch.randn_like(x) * 0.01
            x = x + mask * noise

        x = self.dropout(x)
        y_pred, activations = self.model(x)
        return y_pred, activations

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, _ = self(x)
        base_loss = self.criterion(y_pred, y)

        focused_pred = y_pred[:, self.focus_indices]
        focused_true = y[:, self.focus_indices]
        mse_vector = (focused_pred - focused_true) ** 2

        clipped_weights = torch.clamp(self.sensitive_weights, 1.0, 50.0)
        weighted_loss = torch.mean(clipped_weights * mse_vector.mean(dim=0))

        l1_reg = self.model.reg(scale=1.0)
        total_loss = base_loss + weighted_loss + 0.1 * l1_reg
        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred, _ = self(x)
        val_mse = self.criterion(y_pred, y)
        self.log("val_focused_weighted_MSE", val_mse, prog_bar=True)
        return val_mse

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_focused_weighted_MSE"}
        }




