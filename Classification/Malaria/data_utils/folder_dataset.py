import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision


class FolderDataset(Dataset):
    """Folder dataset handler."""

    def __init__(self, path, labels, do_transform):
        self._data = sorted(os.listdir(path))
        self._labels = labels.clone()
        self.path = path
        self.do_transform = do_transform

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomAffine(degrees=0, scale=(0.8, 1.0), fillcolor=0),
            torchvision.transforms.RandomRotation(180.)
        ])

    def load_image(self, image_path):
        return Image.open(os.path.join(self.path, image_path))

    def transform(self, image):
        if self.do_transform:
            image = self.transforms(image)
        image_transformed = torchvision.transforms.functional.to_tensor(image)
        return image_transformed

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        image_array = self.transform(self.load_image(self._data[int(idx)])) / 255
        labels = self._labels[[int(idx)]]

        sample = {
            'images': image_array,
            'labels': labels
        }
        return sample

