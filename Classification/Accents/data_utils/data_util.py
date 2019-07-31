import os

import torch
import torch.utils

import numpy as np
import pandas

from .folder_dataset import FolderDataset
from torch.utils.data import DataLoader


def load_folder_data(root, train_val_rate, batch_size, test=False, train_width=None, test_width=None):
    # Load data
    print('1. Loading train data')
    train_path = os.path.join(root, 'train')
    train_labels_file = os.path.join(root, 'train_labels.csv')

    # Create datasets
    print('2. Creating datasets')
    dataset = FolderDataset(path=train_path, labels_csv=train_labels_file, do_transform=False, crop_time=True, width=test_width)
    train_length, validation_length = int(train_val_rate*len(dataset)), len(dataset) - int(train_val_rate*len(dataset))
    trainset, validationset = torch.utils.data.random_split(dataset, [train_length, validation_length])
    validationset.width = test_width

    # Create dataloaders
    print('3. Creating dataloaders')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=4)

    if test:
        # Get test data from images
        print('4. Loading test data')
        test_path = os.path.join(root, 'test')

        # Create datasets
        print('5. Creating test datasets')
        testset = FolderDataset(path=test_path, labels_csv=None, do_transform=False, crop_time=True, width=test_width)

        # Create dataloader
        print('6. Creating test dataloader')
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Load submission dataframe
        # submission_df = pandas.read_csv(os.path.join(root, 'submission_format.csv'))
    else:
        testloader = None
        # submission_df = None

    return trainloader, validationloader, testloader


def save_outputs(test_outputs, root, suffix):
    submission_df = pandas.read_csv(os.path.join(root, 'submission_format.csv'))
    submission_df['accent'] = test_outputs.astype(np.int64)
    submission_df.to_csv(os.path.join(root, suffix), index=False)
