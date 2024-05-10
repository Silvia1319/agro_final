import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from water_birds_dataset import CustomizedWaterbirdsDataset
# from modelERM import ModelErm, get_model
from  modelBEIT import ModelBeit,get_model
from  water_birds_module import WaterBirdsDataModule
import  config
import pytorch_lightning as pl
#from domino_slicer import DominoSlicer,DominoMixture,unpack_args
from slicer import DominoSlicer,DominoMixture
from initial_grouper_model import ModelGrouperInitial

#getting group labels

domino = DominoSlicer(n_slices=4, n_mixture_components=5)

pretrained_model = get_model()
fine_tuned_model = get_model(load_weights=True, weights_path=f'model_beit.pth')#model_erm_with_reduction.pth
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_model.to(device)
fine_tuned_model.to(device)

 # Initialize ModelErm
CustomizedWaterbirdsDataset.embeds = None
CustomizedWaterbirdsDataset.groups = None

datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
                                  input_size=config.input_size,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers)

# first_trainer = pl.Trainer(max_epochs=config.final_epochs, accelerator=config.accelerator, devices=[0])
# first_trainer.test(fine_tuned_model, datamodule=datamodule)
datamodule.setup()
embeds, groups = datamodule.getting_embeddings(pretrained_model,fine_tuned_model,domino)

#second step learn grouper to predict these groups for 10 epochs
grouper_initial_model = ModelGrouperInitial(n_slices=config.n_slices, n_features=config.n_features)
initial_grouper_trainer = pl.Trainer(max_epochs=config.initial_grouper_epochs, accelerator=config.accelerator, devices=[0])
initial_grouper_trainer.fit(grouper_initial_model, datamodule)
torch.save(grouper_initial_model.grouper_model.state_dict(), f'model_initial_grouper_beit.pth')