# import copy
# import torch
# import torch.nn as nn
# import pytorch_lightning as pl
# import torchvision.models as models
# from torch import optim
# from torch.optim.lr_scheduler import CosineAnnealingLR
# import config
# class ModelErm(pl.LightningModule):
#     def __init__(self, num_classes=2):
#         super(ModelErm, self).__init__()
#         self.model = models.resnet50(pretrained=True)
#         in_features = self.model.fc.in_features
#         self.model.fc = nn.Linear(in_features, num_classes)
#         self.criterion = nn.CrossEntropyLoss()
#
#     def forward(self, x):
#         return self.model(x)
#
#     def get_embeddings(self, x):
#         # This method modifies the forward pass to stop at the penultimate layer
#         # Extract features from the last layer before the fully connected layer
#         with torch.no_grad():
#
#             def hook(module, input, output):
#                 self.embeddings = output.detach()
#
#             # Register hook to the last layer before the fully connected layer
#             handle = self.model.avgpool.register_forward_hook(hook)
#             # Forward pass to get embeddings
#             self.model(x)
#             # Remove the hook after getting the embeddings
#             handle.remove()
#         return self.embeddings
#
#     def training_step(self, batch, batch_idx):
#         loss, _ = self.common_step(batch, batch_idx)
#         self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss, _ = self.common_step(batch, batch_idx)
#         self.log('validation_loss', loss, on_step=False, on_epoch=True)
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         loss, accuracy_erm = self.common_step(batch, batch_idx)
#         self.log('test_loss', loss, on_step=False, on_epoch=True)
#         self.log('test_accuracy_erm', accuracy_erm, on_step=False, on_epoch=True, prog_bar=True)
#         return loss
#
#     def common_step(self, batch, batch_idx):
#         x, y, c = batch
#         preds = self(x)
#         preds_acc = torch.argmax(preds, dim=1)
#         accuracy_2 = (preds_acc == y).float().mean()
#         loss = self.criterion(preds, y)
#
#         return loss, accuracy_2
#
#     def configure_optimizers(self):
#         optimizer = optim.SGD(self.model.parameters(), lr=config.first_lr, weight_decay=config.weight_decay)
#         scheduler = CosineAnnealingLR(optimizer, T_max=10)
#
#         return [optimizer], [scheduler]
#
#
#
#
# # Function to create a model instance and optionally load weights
# def get_model(load_weights=False, weights_path=None):
#     # Instantiate the model
#     model = ModelErm(num_classes=2)  # Adjust num_classes as needed
#
#     # Optionally load weights
#     if load_weights and weights_path:
#         # Load the model weights from a file
#         model.load_state_dict(torch.load(weights_path))
#
#     return model
import copy
import torch
from water_birds_dataset import CustomizedWaterbirdsDataset
from water_birds_module import WaterBirdsDataModule
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
class ModelErm(pl.LightningModule):
    def __init__(self, num_classes=2, hidden_layer = 128):
        super(ModelErm, self).__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.fc1 = nn.Linear(in_features, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Extract features using the feature extractor part of ResNet50
        features = self.model(x)

        # Apply the intermediate layer
        intermediate_output = self.fc1(features.view(features.size(0), -1))

        # Final classification
        output = self.fc2(intermediate_output)
        return output


    def get_embeddings(self, x):
        # This function will return embeddings of shape [batch_size, 256]
        with torch.no_grad():
            features = self.model(x)
            embeddings = self.fc1(features)
        return embeddings

    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch, batch_idx)
        self.log('validation_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy_erm = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_accuracy_erm', accuracy_erm, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def common_step(self, batch, batch_idx):
        x, y, c = batch
        preds = self(x)
        preds_acc = torch.argmax(preds, dim=1)
        accuracy_2 = (preds_acc == y).float().mean()
        loss = self.criterion(preds, y)

        return loss, accuracy_2

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=config.first_lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        return [optimizer], [scheduler]



def get_model(load_weights=False, weights_path=None):
    # Instantiate the model
    model = ModelErm(num_classes=2)  # Adjust num_classes as needed

    # Optionally load weights
    if load_weights and weights_path:
        # Load the model weights from a file
        model.load_state_dict(torch.load(weights_path))

    return model
# from slicer import DominoSlicer,DominoMixture
# from initial_grouper_model import ModelGrouperInitial
#
# #getting group labels
#
# domino = DominoSlicer(n_slices=4, n_mixture_components=4, max_iter=1000,n_pca_components=5)
#
# pretrained_model = get_model()
# fine_tuned_model = get_model(load_weights=True, weights_path='model_0_seed_final.pth')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pretrained_model.to(device)
# fine_tuned_model.to(device)
# datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
#                                   input_size=config.input_size,
#                                   batch_size=config.batch_size,
#                                   num_workers=config.num_workers)
#
# first_trainer = pl.Trainer(max_epochs=config.final_epochs, accelerator=config.accelerator, devices=[0])
# first_trainer.test(fine_tuned_model, datamodule=datamodule)
# embeds, groups = datamodule.getting_embeddings(pretrained_model,fine_tuned_model,domino)

# erm = ModelErm()  # Initialize ModelErm
# CustomizedWaterbirdsDataset.weights = None
# datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
#                                           input_size=config.input_size,
#                                           batch_size=config.batch_size,
#                                           num_workers=config.num_workers)
# first_trainer = pl.Trainer(max_epochs=config.final_epochs, accelerator=config.accelerator, devices=[0])
# first_trainer.fit(erm, datamodule)
# first_trainer.test(erm,datamodule)
# torch.save(erm.state_dict(), f'model_erm_with_reduction.pth')