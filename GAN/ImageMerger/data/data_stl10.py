import os
import json
import numpy as np
import skimage
import torch.utils.data
import torchvision
from data.data_utils import ArrayDataset


def create_dataset_stl10_bird(opt, start=0, end=None):
    with open(os.path.join(opt.stl_mask_path, 'train.json'), 'r') as f:
        train_idx = json.load(f)
    stl10_train = torchvision.datasets.STL10(opt.stl_path, split='train', download=True)
    stl10_bird_train = (stl10_train.data[[label == stl10_train.classes.index('bird') for label in stl10_train.labels]])[train_idx]

    N, C, H, W = stl10_bird_train.shape
    if opt.use_mask:
        stl10_data_train = np.zeros([N, C+1, H, W])
        stl10_data_train[:, :C, :, :] = stl10_bird_train
        for n in range(N):
            stl10_data_train[n, C, :, :] = skimage.io.imread(os.path.join(opt.stl_mask_path, 'train', 'image_{}.png'.format(train_idx[n])))
    else:
        stl10_data_train = stl10_bird_train

    with open(os.path.join(opt.stl_mask_path, 'test.json'), 'r') as f:
        test_idx = json.load(f)
    stl10_test = torchvision.datasets.STL10(opt.stl_path, split='test', download=True)
    stl10_bird_test = (stl10_test.data[[label == stl10_test.classes.index('bird') for label in stl10_test.labels]])[test_idx[:143]]

    N, C, H, W = stl10_bird_test.shape
    if opt.use_mask:
        stl10_data_test = np.zeros([N, C+1, H, W])
        stl10_data_test[:, :C, :, :] = stl10_bird_test
        for n in range(N):
            stl10_data_test[n, C, :, :] = skimage.io.imread(os.path.join(opt.stl_mask_path, 'test', 'image_{}.png'.format(test_idx[n])))
    else:
        stl10_data_test = stl10_bird_test

    stl10_data = np.concatenate((stl10_data_train, stl10_data_test), axis=0)

    # Scale dataset
    stl10_data = (stl10_data / 255).astype(np.float32)

    # Create DataLoader
    dataset = ArrayDataset(stl10_data, labels=np.ones(stl10_data.shape[0]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return dataloader, stl10_data