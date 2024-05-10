# import copy
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import torchvision.models as models
# from torch import optim
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import config
#
#
# class ModelGrouperInitial(pl.LightningModule):
#     def __init__(self, n_slices, n_features):
#         super(ModelGrouperInitial, self).__init__()
#         self.n_features = n_features
#         self.n_slices = n_slices
#         # self.grouper_model = nn.Sequential(
#         #     nn.Linear(self.n_features, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, self.n_slices),
#         #     nn.Softmax(dim=1)  # Apply Softmax along dimension 1 (features dimension)
#         # )
#         self.grouper_model = nn.Sequential(
#                     nn.Linear(self.n_features, 128),
#                     nn.ReLU(),
#                     nn.Linear(128, 64),
#                     nn.ReLU(),
#                     nn.Linear(64, 32),
#                     nn.ReLU(),
#                     nn.Linear(32, 16),
#                     nn.ReLU(),
#                     nn.Linear(16, self.n_slices),
#                     nn.Softmax()
#                 )
#         self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
#
#     def forward(self, x):
#         return self.grouper_model(x)
#
#     def training_step(self, batch, batch_idx):
#         x, y, e, g,c  = batch
#         preds = self(e)
#         preds_log = torch.log_softmax(preds, dim=1)  # Log probabilities of predictions
#
#         # Convert g to one-hot encoding if it's not already probabilities
#         g_one_hot = torch.nn.functional.one_hot(g, num_classes=self.n_slices).float()
#
#         # Compute the KL divergence loss
#         loss = self.kl_div_loss(preds_log, g_one_hot)
#
#         # Calculate accuracy
#         preds_acc = torch.argmax(preds, dim=1)
#         accuracy_2 = (preds_acc == g).float().mean()
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#
#         # You should return a single output from training_step for logging purposes
#         self.log('train_accuracy', accuracy_2, on_step=False, on_epoch=True, prog_bar=True)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = optim.SGD(self.grouper_model.parameters(), lr=config.grouper_lr, weight_decay=config.weight_decay)
#         scheduler = CosineAnnealingLR(optimizer, T_max=10)
#
#         return [optimizer], [scheduler]

import copy
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config

class ModelGrouperInitial(pl.LightningModule):
    def __init__(self, n_slices, n_features):
        super(ModelGrouperInitial, self).__init__()
        self.n_features = n_features
        self.n_slices = n_slices
        self.grouper_model = nn.Sequential(
            nn.Linear(self.n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.n_slices),
            nn.Softmax()
        )
        # Changed from KLDivLoss to CrossEntropyLoss
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.grouper_model(x)

    def training_step(self, batch, batch_idx):
        x, y, e, g , c= batch
        preds = self(e)  # Output from the model
        # Compute cross-entropy loss directly
        loss = self.cross_entropy_loss(preds, g)

        # Calculate accuracy
        preds_acc = torch.argmax(preds, dim=1)
        accuracy_2 = (preds_acc == g).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy_2, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.grouper_model.parameters(), lr=config.grouper_lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]




# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# from torch import optim
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import config
#
# class ModelGrouperInitial(pl.LightningModule):
#     def __init__(self, n_slices, n_features, hidden_size=64):
#         super(ModelGrouperInitial, self).__init__()
#         self.n_features = n_features
#         self.n_slices = n_slices
#         self.hidden_size = hidden_size
#
#         # Define a single LSTM layer with input_size set to 1
#         self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
#
#         # Fully connected layers after LSTM
#         self.fc_layers = nn.Sequential(
#             nn.Linear(hidden_size, 16),
#             nn.ReLU(),
#             nn.Linear(16, n_slices),
#             nn.Softmax(dim=1)
#         )
#
#         # Loss function
#         self.cross_entropy_loss = nn.CrossEntropyLoss()
#
#     def forward(self, x):
#         # Reshape x to have each feature as a separate timestep: (batch_size, n_features, 1)
#         x = x.unsqueeze(-1)  # or x.view(x.size(0), self.n_features, 1)
#
#         # Pass through LSTM
#         lstm_out, _ = self.lstm(x)
#         # Take only the final output from the last LSTM cell
#         lstm_out = lstm_out[:, -1, :]
#         # Pass through fully connected layers
#         return self.fc_layers(lstm_out)
#
#     def training_step(self, batch, batch_idx):
#         x, y, e, g, c = batch
#         # Forward pass through the model
#         preds = self(e)
#         # Compute the loss
#         loss = self.cross_entropy_loss(preds, g)
#
#         # Calculate accuracy
#         preds_acc = torch.argmax(preds, dim=1)
#         accuracy_2 = (preds_acc == g).float().mean()
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('train_accuracy', accuracy_2, on_step=False, on_epoch=True, prog_bar=True)
#         return loss
#
#     def configure_optimizers(self):
#         optimizer = optim.SGD(self.parameters(), lr=config.grouper_lr, weight_decay=config.weight_decay)
#         scheduler = CosineAnnealingLR(optimizer, T_max=10)
#         return [optimizer], [scheduler]
