import os
import pandas

import torch
import torch.utils
import skimage

from .dataset import ArrayDataset
from torch.utils.data import DataLoader


def load_data(root, train_val_rate, batch_size):
    labels = pandas.read_csv(os.path.join(root, 'train_labels.csv'))
    #train_file = sorted(os.listdir(os.path.join(root, 'train')))

    data = torch.empty(size=[len(labels), 3, 128, 128], dtype=torch.float)
    for i in range(len(labels)):
        image_path = os.path.join(root, 'train', labels['filename'][i])
        data[i] = torch.FloatTensor(skimage.io.imread(image_path)).permute([2,0,1])

    dataset = ArrayDataset(data, labels=torch.from_numpy(labels['infected'].to_numpy()))
    train_length, validation_length = int(train_val_rate*len(labels)), len(labels) - int(train_val_rate*len(labels))
    trainset, validationset = torch.utils.data.random_split(dataset, [train_length, validation_length])

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, validationloader
