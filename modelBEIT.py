import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import BeitModel, BeitConfig
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import config
from water_birds_dataset import CustomizedWaterbirdsDataset
from water_birds_module import WaterBirdsDataModule,TrainingStage

class ModelBeit(pl.LightningModule):
    def __init__(self, num_classes=2):
        super(ModelBeit, self).__init__()
        # Load the pretrained BEiT model from Hugging Face
        beit_config = BeitConfig.from_pretrained('microsoft/beit-base-patch16-224', output_hidden_states=True)
        self.model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224', config=beit_config)

        # Classification head
        self.fc = nn.Linear(beit_config.hidden_size, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        outputs = self.model(x)
        hidden_states = outputs.hidden_states[-1]
        logits = self.fc(hidden_states[:, 0])
        return logits

    def get_embeddings(self, x):
        with torch.no_grad():
            outputs = self.model(x)
            last_hidden_layer = outputs.hidden_states[-1]
            classifier_representations = last_hidden_layer[:, 0, :].detach()
        return classifier_representations

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, accuracy_erm = self.common_step(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        loss, accuracy_erm = self.common_step(x, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_accuracy_erm', accuracy_erm, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def common_step(self, x, y):
        preds = self(x)
        preds_acc = torch.argmax(preds, dim=1)
        accuracy_2 = (preds_acc == y).float().mean()
        loss = self.criterion(preds, y)
        return loss, accuracy_2

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=config.first_lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

# Function to create a model instance and optionally load weights
def get_model(load_weights=False, weights_path=None):
    model_beit = ModelBeit(num_classes=2)  # Adjust num_classes as needed

    if load_weights and weights_path:
        model_beit.load_state_dict(torch.load(weights_path))

    return model_beit

#
# beit = ModelBeit()  # Initialize ModelErm
# datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
#                                   input_size=config.input_size,
#                                   batch_size=config.batch_size,
#                                   num_workers=config.num_workers)
# datamodule.setup()
# first_trainer = pl.Trainer(max_epochs=config.final_epochs, accelerator=config.accelerator, devices=[0])
# first_trainer.fit(beit, datamodule)
# torch.save(beit.state_dict(), f'model_beit.pth')



# beit = get_model(load_weights=True,weights_path="model_beit.pth")
# datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
#                                   input_size=config.input_size,
#                                   batch_size=config.batch_size,
#                                   num_workers=config.num_workers)
# datamodule.setup()
#
# datamodule._stage = TrainingStage.REWEIGTING
#
# data_iter = iter(datamodule.train_dataloader())  # Use validation dataloader for testing
# images, labels, _ = next(data_iter)
# beit.eval()
#
# # # Convert images to tensor and move to the appropriate device
#
# # Get the embeddings
# embeddings = beit.get_embeddings(images)
# print(embeddings)
# print(embeddings.shape)