import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    """Image dataset handler."""

    def __init__(self, data, labels):
        self._data = data.copy()
        self._labels = labels.copy()

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


class ArrayDataset(ImageDataset):
    """Image dataset handler, assuming images are stored in an array"""

    def __init__(self, data, labels):
        super(ArrayDataset, self).__init__(data, labels)
        self._base = 0  # keys in an array zero-based
