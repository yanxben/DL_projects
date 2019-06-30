import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision


class Scaler(object):
    def __init__(self):
        self._pixel_span = 255

    def scale_image(self, image_array):
        return image_array / self._pixel_span


class ImageDataset(Dataset):
    """Image dataset handler."""

    def __init__(self, data, labels, do_transform):
        self._data = data.clone()
        self._labels = labels.clone()
        self.do_transform = do_transform
        # self._scaler = Scaler()

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomAffine(degrees=0, scale=(0.8, 1.0), fillcolor=0),
            torchvision.transforms.RandomRotation(180.)
        ])

    def transform(self, image):
        image = torchvision.transforms.functional.to_pil_image(image)
        if self.do_transform:
            image = self.transforms(image)
        image = torchvision.transforms.functional.to_tensor(image)
        return image

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        image_array = self.transform(self._data[int(idx) + self._base]) / 255
        labels = self._labels[[int(idx)]]

        sample = {
            'images': image_array,
            'labels': labels
        }
        return sample


# class TorchFileDataset(ImageDataset):
#     """Image dataset handler, assuming images are stored in torchfile.hashable_uniq_dict"""
#
#     def __init__(self, data, labels):
#         super(TorchFileDataset, self).__init__(data, labels)
#         self._base = 1  # keys in a torchfile.hashable_uniq_dict are one-based

class ArrayDataset(ImageDataset):
    """Image dataset handler, assuming images are stored in an array"""

    def __init__(self, data, labels, do_transform):
        super(ArrayDataset, self).__init__(data, labels, do_transform)
        self._base = 0  # keys in an array zero-based
