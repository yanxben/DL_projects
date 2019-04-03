import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd


class Scaler(object):
    def __init__(self):
        self._pixel_span = 255

    def scale_image(self, image_array):
        return image_array / self._pixel_span


class ImageDataset(Dataset):
    """Image dataset handler."""

    def __init__(self, data, labels):
        self._data = data.copy()
        self._labels = labels.copy()

        #self._scaler = Scaler()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        #image_array = self._scaler.scale_image(self._data[idx + self._base])
        image_array = self._data[int(idx) + self._base]
        labels = self._labels[[int(idx)]]

        sample = {
            'images': torch.from_numpy(image_array),
            'labels': torch.from_numpy(labels)
        }
        return sample


class TorchFileDataset(ImageDataset):
    """Image dataset handler, assuming images are stored in torchfile.hashable_uniq_dict"""

    def __init__(self, data, labels):
        super(TorchFileDataset, self).__init__(data, labels)
        self._base = 1  # keys in a torchfile.hashable_uniq_dict are one-based

class ArrayDataset(ImageDataset):
    """Image dataset handler, assuming images are stored in an array"""

    def __init__(self, data, labels):
        super(ArrayDataset, self).__init__(data, labels)
        self._base = 0  # keys in an array zero-based


# example use case
# positive_data = fetch_data.fetch_positive_samples_12()
# positive_trainset = TorchFileDataset(positive_data, labels=np.ones((len(positive_data),)))
# positive_trainloader = DataLoader(positive_trainset, batch_size=32, shuffle=True, num_workers=2)
#
# n_negative = 100 # should be n_negative = len(positive_data)
# negative_data = fetch_data.fetch_negative_samples(size=12, n=n_negative)
# negative_trainset = ArrayDataset(negative_data, labels=np.zeros((len(negative_data),)))
# negative_trainloader = DataLoader(negative_trainset, batch_size=32, shuffle=True, num_workers=2)
