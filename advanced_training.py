import torch
import torch.nn
from pytorch_lightning.callbacks import ModelCheckpoint

from domino_slicer import DominoSlicer
from water_birds_module import WaterBirdsDataModule
from advanced_grouper import Grouper_Model
import config
import pytorch_lightning as pl
import numpy as np
import random
from pytorch_lightning import seed_everything
from water_birds_dataset import CustomizedWaterbirdsDataset
from modelERM import ModelErm, get_model
# from  modelBEIT import ModelBeit,get_model

pretrained_model = get_model()
fine_tuned_model = get_model(load_weights=True, weights_path='model_erm_with_reduction.pth')
# beit = get_model(load_weights=True,weights_path="model_beit.pth")
datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
                                  input_size=config.input_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers)

first_trainer = pl.Trainer(max_epochs=config.final_epochs, accelerator=config.accelerator, devices=[0])
first_trainer.test(fine_tuned_model, datamodule=datamodule)
# first_trainer.test(beit, datamodule=datamodule)
domino = DominoSlicer(n_slices=4, n_mixture_components=4,max_iter=100,n_pca_components=100)#,max_iter=100,n_pca_components=100

grouper = Grouper_Model(n_features=config.n_features, n_slices=config.n_slices,task_model=fine_tuned_model)
# grouper = Grouper_Model(n_features=config.n_features, n_slices=config.n_slices,task_model=beit)
# grouper.grouper_model.load_state_dict(torch.load(f'model_initial_grouper_beit.pth'))
grouper.grouper_model.load_state_dict(torch.load(f'model_initial.pth'))

embeds, groups = datamodule.getting_embeddings(pretrained_model, fine_tuned_model, domino)
# embeds, groups = datamodule.getting_embeddings(pretrained_model, beit, domino)
final_trainer = pl.Trainer(max_epochs=config.final_epochs, accelerator=config.accelerator, devices=[0])

final_trainer.fit(grouper, datamodule)
torch.save(grouper.state_dict(), f'grouper_ERM_with_W1.pth')
# torch.save(grouper.state_dict(), f'grouper_trained_with_bert.pth')
CustomizedWaterbirdsDataset.training = False
test_embeds, test_groups = datamodule.get_embeds_for_test(pretrained_model, fine_tuned_model, domino)
# test_embeds, test_groups = datamodule.get_embeds_for_test(pretrained_model, beit, domino)
print(grouper.worst_group)
datamodule.model_predictions(grouper)