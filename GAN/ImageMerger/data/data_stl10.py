import os
import numpy as np
import skimage
import torch.utils.data
import torchvision
from data.data_utils import ArrayDataset


def create_dataset_stl10_bird(opt, start=0, end=None):
    stl10_train = torchvision.datasets.STL10(opt.stl_path, split='train', download=True)
    stl10_bird = (stl10_train.data[[label == stl10_train.classes.index('bird') for label in stl10_train.labels]])[start:end]

    N, C, H, W = stl10_bird.shape
    if opt.use_mask:
        stl10_data = np.zeros([N, C+1, H, W])
        stl10_data[:, :C, :, :] = stl10_bird
        for n in range(N):
            stl10_data[n, C, :, :] = skimage.io.imread(os.path.join(opt.stl_mask_path, 'image_{}.png'.format(n + start)))
    else:
        stl10_data = stl10_bird

    # Scale dataset
    stl10_data = (stl10_data / 255).astype(np.float32)

    # Create DataLoader
    dataset = ArrayDataset(stl10_data, labels=np.ones(N))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    return dataloader, stl10_data