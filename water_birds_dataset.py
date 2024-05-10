import numpy as np
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
import config
import os
from PIL import Image


class CustomizedWaterbirdsDataset(WaterbirdsDataset):
    embeds = None
    groups = None
    test_embeds = None
    test_groups = None
    training = True
    selected_indices_file = "selected_indices.npy"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cache = {}
        self._make_reweigting_set()

    def _make_reweigting_set(self):
        """
        Creates a reweighting set from the training set for use in the CustomizedWaterbirdsDataset.

        This method selects a subset of the training set and assigns it to a new reweighting split. To fully
        understand the implementation, it's recommended to review the implementation of the WaterbirdsDataset class.

        The method identifies training samples, randomly selects a specified proportion of them
        (20% by default), and reassigns these selected indices to a new split category named 'train_rw'.
        """
        # validation_indices = np.where(self._split_array == self.split_dict['val'])[0]
        # num_of_val = int(len(validation_indices)-len(self._split_array)*0.005)
        # select_for_not_val = np.random.choice(validation_indices, num_of_val, replace=False)
        # self._split_array[select_for_not_val] = 0

        train_indices = np.where(self._split_array == self.split_dict['train'])[0]
        num_to_change = int(len(train_indices) * config.rw_ratio)  # TODO 0.2 is a magic number, pass from configs
        selected_indices=None
        if os.path.exists(CustomizedWaterbirdsDataset.selected_indices_file):
            selected_indices = np.load(self.selected_indices_file)
        else:
            selected_indices = np.random.choice(train_indices, num_to_change, replace=False)
            np.save(CustomizedWaterbirdsDataset.selected_indices_file, selected_indices)
        self._split_array[selected_indices] = 3
        self._split_names = {'train': 'Train For ERM',
                             'val': 'Validation',
                             'test': 'Test',
                             'train_rw': 'Train for reweighting'}
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'train_rw': 3}
    def __getitem__(self, idx):

        x, y, metadata = super().__getitem__(idx)
        # if idx==3 or idx==186 or idx==543 or idx==325:
        #    x.save(f"image{idx}.png")
        if  self.training:
            x, y, c = x, y, metadata[0]
            if (self.embeds is None and self.groups is None) or idx not in self.embeds:
                return x, y, c
            elif self.embeds is None and self.groups is not None:
                g = self.groups[idx]
                return x, y, g

            else:
                e = self.embeds[idx]
                g = self.groups[idx]
                return x, y, e, g, c
        else:
            x, y, c = x, y, metadata[0]
            if (self.test_embeds is None and self.test_groups is None) or idx not in self.test_embeds:
                return x, y, c
            else:
                e = self.test_embeds[idx]
                g = self.test_groups[idx]
                return x, y, e, g,c