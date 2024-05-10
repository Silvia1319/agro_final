import csv
import os
from torchvision.utils import save_image
import numpy as np
import torch
import pytorch_lightning as pl
from domino_slicer import DominoSlicer,DominoMixture
from enum import Enum
from torch.utils.data import DataLoader

from torchvision.transforms import transforms
from water_birds_dataset import CustomizedWaterbirdsDataset as WaterbirdsDataset
import config


class TrainingStage(Enum):
    ERM = 1
    REWEIGTING = 2


class WaterBirdsDataModule(pl.LightningDataModule):
    def __init__(self, name: str, root_dir: str, input_size: int, batch_size: int, num_workers: int, **kwargs):
        super().__init__()
        self._name = name
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._stage = TrainingStage.ERM
        self._transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._to_tensor_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor()
        ])

        self._train_erm_data = None
        self._train_rw_data = None
        self._train_rw_original = None
        self._train_erm_original = None
        self._val_data = None
        self._test_data = None

    def setup(self, *args, **kwargs):
        dataset = WaterbirdsDataset(root_dir=self._root_dir, download=True)
        self._train_erm_data = dataset.get_subset("train", transform=self._transform)
        # self._train_rw_data = dataset.get_subset("train_rw", transform=self._transform)
        self._train_rw_data = dataset.get_subset("train_rw", transform=self._transform)
        self._train_rw_original = dataset.get_subset("train_rw", transform=self._to_tensor_transform)  # No advanced transform for saving original images
        self._train_erm_original = dataset.get_subset("train", transform=self._to_tensor_transform)  # No advanced transform for saving original images
        self._val_data = dataset.get_subset("val", transform=self._transform)
        self._test_data = dataset.get_subset("test", transform=self._transform)
        self._test_data_original = dataset.get_subset("test", transform=self._to_tensor_transform)

    def train_dataloader(self, shuffle=True, include_original=False) -> DataLoader:
        if include_original:
            transformed_loader = DataLoader(self._train_rw_data, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=shuffle)
            original_loader = DataLoader(self._train_rw_original, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=shuffle)
            return zip(transformed_loader, original_loader)
        else:
            data = self._train_erm_data if self._stage == TrainingStage.ERM else self._train_rw_data
            return DataLoader(data, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=shuffle)

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(self._val_data,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers,
                                shuffle=False)
        return val_loader

    def test_dataloader(self, shuffle=False, include_original=False) -> DataLoader:
        test_loader = DataLoader(self._test_data,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers,
                                shuffle=False)
        return test_loader

    def getting_embeddings(self, pretrained_model, fine_tuned_model,domino):
        pretrained_model.eval()
        pretrained_model.cuda()

        fine_tuned_model.eval()
        fine_tuned_model.cuda()

        logits = []
        concatenated_embeddings = []
        ys = []

        for x, y, c in self.train_dataloader(shuffle=False, include_original=False):
            pre_embeddings = (pretrained_model.get_embeddings(x.to("cuda")).detach().cpu()).squeeze()
            embeddings = (fine_tuned_model.get_embeddings(x.to("cuda")).detach().cpu()).squeeze()
            concat_embeddings = torch.cat((pre_embeddings, embeddings), dim=1)
            concatenated_embeddings.append(concat_embeddings)
            y_hat = fine_tuned_model(x.to("cuda")).detach().cpu()
            y_hat = torch.argmax(y_hat, dim=1)
            logits.append(y_hat)
            ys.append(y)

        concatenated_embeddings = torch.cat(concatenated_embeddings)
        logits = torch.cat(logits)
        ys = torch.cat(ys)

        domino.fit(None, np.array(concatenated_embeddings), np.array(ys, dtype=float), (logits.detach().numpy()).astype(float))
        clusters = domino.transform(None, np.array(concatenated_embeddings), np.array(ys, dtype=float), (logits.detach().numpy()).astype(float))
        clusters_tensor = torch.tensor(clusters)

        # Apply softmax along the appropriate axis (axis=1 for row-wise softmax)
        softmax_clusters = torch.nn.functional.softmax(clusters_tensor, dim=1)
        clusters = torch.argmax(softmax_clusters, dim=1)
        clusters = clusters.numpy()
        WaterbirdsDataset.groups = {self._train_erm_data.indices[i]: clusters[i] for i in
                                     range(len(self._train_erm_data))}
        WaterbirdsDataset.embeds = {self._train_erm_data.indices[i]: concatenated_embeddings[i] for i in
                                     range(len(self._train_erm_data))}

        with open('groups.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header (optional)
            writer.writerow(['Index', 'Value'])  # Assuming 'Index' and 'Value' as column headers
            # Write each key-value pair as a row in the CSV file
            for key, value in WaterbirdsDataset.groups.items():
                writer.writerow([key, value])
        with open('embeddings.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header (optional)
            writer.writerow(['Index', 'Value'])  # Assuming 'Index' and 'Value' as column headers
            # Write each key-value pair as a row in the CSV file
            for key, value in WaterbirdsDataset.embeds.items():
                writer.writerow([key, value])
        return WaterbirdsDataset.embeds, WaterbirdsDataset.groups

    def get_embeds_for_test(self, pretrained_model, fine_tuned_model,domino):

        self._stage = TrainingStage.REWEIGTING
        pretrained_model.eval()
        pretrained_model.cuda()

        fine_tuned_model.eval()
        fine_tuned_model.cuda()
        logits = []
        concatenated_embeddings = []
        ys = []
        cs = []
        for x, y, c in self.train_dataloader(shuffle=False, include_original=False):
            pre_embeddings = (pretrained_model.get_embeddings(x.to("cuda")).detach().cpu()).squeeze()
            embeddings = (fine_tuned_model.get_embeddings(x.to("cuda")).detach().cpu()).squeeze()
            concat_embeddings = torch.cat((pre_embeddings, embeddings), dim=1)
            concatenated_embeddings.append(concat_embeddings)
            y_hat = fine_tuned_model(x.to("cuda")).detach().cpu()
            y_hat = torch.argmax(y_hat, dim=1)
            logits.append(y_hat)
            ys.append(y)
            cs.append(c)

        concatenated_embeddings = torch.cat(concatenated_embeddings)
        logits = torch.cat(logits)
        ys = torch.cat(ys)
        cs = torch.cat(cs)

        domino.fit(None, np.array(concatenated_embeddings), np.array(ys, dtype=float), (logits.detach().numpy()).astype(float))
        clusters = domino.transform(None, np.array(concatenated_embeddings), np.array(ys, dtype=float), (logits.detach().numpy()).astype(float))
        clusters_tensor = torch.tensor(clusters)

        # Apply softmax along the appropriate axis (axis=1 for row-wise softmax)
        softmax_clusters = torch.nn.functional.softmax(clusters_tensor, dim=1)
        weights,_ = torch.max(softmax_clusters, dim=1)
        weights = weights.numpy()
        clusters = torch.argmax(softmax_clusters, dim=1)
        clusters = clusters.numpy()
        WaterbirdsDataset.test_groups = {self._train_rw_data.indices[i]: clusters[i] for i in
                                     range(len(self._train_rw_data))}
        WaterbirdsDataset.test_embeds = {self._train_rw_data.indices[i]: concatenated_embeddings[i] for i in
                                     range(len(self._train_rw_data))}

        with open('dominoW.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header (optional)
            writer.writerow(['Index', 'Value','Group','Label','Weights'])  # Assuming 'Index' and 'Value' as column headers
            # Write each key-value pair as a row in the CSV file
            for index, (key, value) in enumerate(WaterbirdsDataset.test_groups.items()):
                writer.writerow([key, value, cs[index], ys[index], weights[index]])

        return WaterbirdsDataset.test_embeds, WaterbirdsDataset.test_groups

    def model_predictions(self,model):

        self._stage = TrainingStage.REWEIGTING
        # Disable training mode and move model to GPU
        model.eval()
        model.cuda()

        # Prepare directories for saving images
        os.makedirs('test_images_W', exist_ok=True)
        for j in range(4):
            os.makedirs(f'test_images_W/test_images{j}', exist_ok=True)

        # Initialize a list to collect CSV data
        results = []
        index = 0
        # Process each batch in the test dataloader
        for batch, original in self.train_dataloader(shuffle=False, include_original=True):
            (x, y, e, g, c), (x_original, _, _, _,c) = batch, original
            group_distribution = model(e.to("cuda"))
            predicted_groups = torch.argmax(group_distribution, dim=1)
            probabilities = torch.max(group_distribution, dim=1)[0].detach().cpu().numpy()

            # Iterate over each example in the batch
            for i in range(x_original.shape[0]):
                ind = self._train_rw_data.indices[index]
                index+=1
                pred_group = predicted_groups[i].item()
                probability = probabilities[i]
                results.append([ind, pred_group, probability, c[i], y[i]])

                # Save image if the predicted group matches the worst group
                save_path = f'test_images_W/test_images{pred_group}/image_{ind}.png'
                save_image(x_original[i], save_path)

        # Write all results to the CSV file at once
        with open('trained_grouper_W.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Predicted Group', 'Weights','Real Group','Real Label'])
            writer.writerows(results)

        # Re-enable training mode for the dataset
        model.train()