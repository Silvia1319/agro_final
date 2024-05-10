import copy
import os

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
from transformers import BeitModel, BeitConfig

class Grouper_Model(pl.LightningModule):
    def __init__(self, n_features,n_slices,task_model):
        super(Grouper_Model, self).__init__()
        self.n_features = n_features
        self.n_slices = n_slices
        self.training = True
        self.task_model = copy.deepcopy(task_model)
        self.processed_data_counts = torch.zeros(self.n_slices)
        self.avg_group_loss = torch.zeros(self.n_slices)
        self.avg_group_acc = torch.zeros(self.n_slices)
        # tr_loss_step_primary = torch.tensor(0.0).to(args.device)
        self.avg_per_sample_loss = torch.tensor(0.0)
        self.avg_actual_loss = torch.tensor(0.0)
        self.avg_acc = torch.tensor(0.0)
        self.worst_group = None
        self.batch_count = 0
        self.all_group_losses = torch.zeros(4)
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

    def forward(self, x):
        return self.grouper_model(x)

    def label_loss(self, batch, batch_idx):
        x, y, _,_,_ = batch
        loss = nn.CrossEntropyLoss(reduction="none")
        pred = self.task_model(x)
        return loss(pred, y)
    def compute_soft_group_loss(self, losses, group_prob):
        group_wise_loss = group_prob * losses.unsqueeze(1)
        if self.training:
            self.all_group_losses =self.all_group_losses.to(device = config.device) + (group_wise_loss.sum(0)).to(device = config.device)
            self.worst_group = (torch.argmax(self.all_group_losses)).item()
        return group_wise_loss.sum(0)

    def compute_adversary_loss_greedy(self, groups_losses, alpha, W):
        # Sort the losses in descending order and get corresponding indices
        sorted_losses, sorted_idx = torch.sort(groups_losses, descending=True)

        # Calculate the number of elements to assign weight alpha
        num_alpha = int(round(alpha * len(sorted_losses)))  # Round to nearest integer

        # Initialize output losses and weights arrays with zeros
        output_losses = torch.zeros_like(groups_losses)
        weights = torch.zeros_like(groups_losses)

        # Assign weights based on the top alpha percent
        if num_alpha > 0:
            # Assign weights and losses for the top alpha percent
            output_losses[sorted_idx[:num_alpha]] = sorted_losses[:num_alpha] * alpha
            weights[sorted_idx[:num_alpha]] = alpha

        # Assign weights for the remaining (1 - alpha) percent
        if num_alpha < len(sorted_losses):
            output_losses[sorted_idx[num_alpha:]] = sorted_losses[num_alpha:] * W
            weights[sorted_idx[num_alpha:]] = W

        # Sum up the output losses and return along with the weights
        total_loss = output_losses.sum()
        return total_loss, weights

    def training_step(self, batch, batch_idx):
        x, y, e, g ,c = batch

        losses = self.label_loss(batch,batch_idx)
        final_loss =0
        self.group_distribution  = self(e)
        group_losses = self.compute_soft_group_loss(losses,self.group_distribution)

        final_loss,weights = self.compute_adversary_loss_greedy(group_losses,config.alpha,config.W)
        self.log('train_loss', final_loss, on_step=True, on_epoch=True,prog_bar=True)

        minibatch_group_acc, minibatch_group_count = self.compute_group_avg((torch.argmax(self.task_model(x), 1) == y).float(),
                                                                            self.group_distribution)
        self.update_stats(final_loss, group_losses, minibatch_group_acc, minibatch_group_count, weights)

        return -final_loss
    def compute_group_avg(self, losses, group_distribution):
        # Find argmax for groups
        group_idx = torch.argmax(group_distribution, dim=1)
        group_map = (group_idx == (torch.arange(self.n_slices).unsqueeze(1).long()).to(device = config.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_stats(self, actual_loss, group_loss, group_acc, group_count, weights=None):
        # avg group loss
        denom = self.processed_data_counts.to(config.device) + group_count.to(config.device)
        denom += (denom == 0).float()
        denom = denom.to(config.device)
        prev_weight = self.processed_data_counts.to(config.device) / denom
        curr_weight = group_count.to(config.device) / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss.to(config.device) + curr_weight * group_loss.to(config.device)

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc.to(config.device) + curr_weight * group_acc.to(config.device)

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count / denom) * self.avg_actual_loss.to(config.device) + (1 / denom) * actual_loss.to(config.device)

        # counts
        self.processed_data_counts =self.processed_data_counts.to(config.device)+ group_count

        # avg per-sample quantities
        group_frac = self.processed_data_counts.to(config.device) / (self.processed_data_counts.sum()).to(config.device)
        self.avg_per_sample_loss = group_frac.to(config.device) @ self.avg_group_loss.to(config.device)
        self.avg_acc = group_frac.to(config.device) @ self.avg_group_acc.to(config.device)
    def on_train_epoch_end(self):

        self.log('train_avg_loss', torch.tensor(self.avg_actual_loss), on_epoch=True, prog_bar=True)
        self.processed_data_counts = torch.zeros(self.n_slices)
        self.avg_group_loss = torch.zeros(self.n_slices)
        self.avg_group_acc = torch.zeros(self.n_slices)
        # tr_loss_step_primary = torch.tensor(0.0).to(args.device)
        self.avg_per_sample_loss = torch.tensor(0.0)
        self.avg_actual_loss = torch.tensor(0.0)
        self.avg_acc = torch.tensor(0.0)
        self.batch_count = 0
        self.all_group_losses = torch.zeros(4)


    def configure_optimizers(self):
        optimizer = optim.SGD(self.grouper_model.parameters(), lr=config.second_lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        return [optimizer], [scheduler]