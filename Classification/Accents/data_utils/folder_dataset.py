import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision

import numpy as np
import pandas


class FolderDataset(Dataset):
    """Folder dataset handler."""

    def __init__(self, path, labels_csv, do_transform, crop_time=True, width=None):
        self.path = path
        self.data = sorted(os.listdir(path))
        if labels_csv is not None:
            labels_df = pandas.read_csv(labels_csv)
            try:
                self.labels = torch.from_numpy(labels_df['accent'].to_numpy()).long()
            except AttributeError:
                self.labels = torch.from_numpy(np.array(labels_df['accent'])).long()
        else:
            self.labels = torch.zeros((len(self.data),))
        self.do_transform = do_transform

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomAffine(degrees=0, scale=(0.8, 1.0), fillcolor=0),
            torchvision.transforms.RandomRotation(180.)
        ])

        self.crop_time = crop_time
        self.width = width

    def load_image(self, image_path):
        return Image.open(os.path.join(self.path, image_path))

    def transform(self, image):
        if self.do_transform:
            image = self.transforms(image)
        image_transformed = torchvision.transforms.functional.to_tensor(image)
        return image_transformed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pil_image = self.load_image(self.data[int(idx)])
        image = self.transform(pil_image) / 255
        label = self.labels[[int(idx)]]

        if self.crop_time:
            idx = random.randint(0, 173-self.width)
            delta = self.width
        else:
            idx = 0
            delta = self.width
        sample = {
            'images': image[:1, :128, idx:idx+delta],
            'labels': label
        }
        return sample

