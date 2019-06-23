import os
import pandas

import torch
import torch.utils
import skimage

from .image_dataset import ArrayDataset
from .folder_dataset import FolderDataset
from torch.utils.data import DataLoader


def load_image_data(root, train_val_rate, batch_size, test=False):
    print('1. Loading data')
    # Get labels
    labels_df = pandas.read_csv(os.path.join(root, 'train_labels.csv'))
    labels = torch.from_numpy(labels_df['infected'].to_numpy()).float()
    # print(labels.shape)

    # Get data from images
    data = torch.empty(size=[len(labels_df), 3, 128, 128], dtype=torch.float)
    for i in range(len(labels_df)):
        image_path = os.path.join(root, 'train', labels_df['filename'][i])
        data[i] = torch.FloatTensor(skimage.io.imread(image_path)).permute([2,0,1])

    # Create datasets
    print('2. Creating datasets')
    dataset = ArrayDataset(data, labels=labels, do_transform=True)
    train_length, validation_length = int(train_val_rate*len(labels_df)), len(labels_df) - int(train_val_rate*len(labels_df))
    trainset, validationset = torch.utils.data.random_split(dataset, [train_length, validation_length])

    # Create dataloader
    print('3. Creating dataloader')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=4)

    if test:
        # Get test data from images
        print('4. Loading test data')
        testfiles = sorted(os.listdir(os.path.join(root, 'test')))
        test_data = torch.empty(size=[len(testfiles), 3, 128, 128], dtype=torch.float)
        for i in range(len(testfiles)):
            image_path = os.path.join(root, 'test', testfiles[i])
            test_data[i] = torch.FloatTensor(skimage.io.imread(image_path)).permute([2,0,1])

        # Create datasets
        print('5. Creating test datasets')
        testset = ArrayDataset(test_data, labels=torch.zeros((len(testfiles),)), do_transform=True)

        # Create dataloader
        print('6. Creating test dataloader')
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

        # # Load submission dataframe
        # submission_df = pandas.read_csv(os.path.join(root, 'submission_format.csv'))
    else:
        testloader = None
        # submission_df = None

    return trainloader, validationloader, testloader


def load_folder_data(root, train_val_rate, batch_size, test=False):
    print('1. Loading data')
    # Get labels
    labels_df = pandas.read_csv(os.path.join(root, 'train_labels.csv'))
    labels = torch.from_numpy(labels_df['infected'].to_numpy()).float()
    # print(labels.shape)

    # Get data from images
    # data = labels_df['filename'].tolist()

    # Create datasets
    print('2. Creating datasets')
    dataset = FolderDataset(os.path.join(root, 'train'), labels=labels, do_transform=True)
    train_length, validation_length = int(train_val_rate*len(labels_df)), len(labels_df) - int(train_val_rate*len(labels_df))
    trainset, validationset = torch.utils.data.random_split(dataset, [train_length, validation_length])

    # Create dataloader
    print('3. Creating dataloader')
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validationloader = DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=4)

    if test:
        # Get test data from images
        print('4. Loading test data')
        testfiles = sorted(os.listdir(os.path.join(root, 'test')))
        # test_data = torch.empty(size=[len(testfiles), 3, 128, 128], dtype=torch.float)
        # for i in range(len(testfiles)):
        #     image_path = os.path.join(root, 'test', testfiles[i])
        #     test_data[i] = torch.FloatTensor(skimage.io.imread(image_path)).permute([2,0,1])

        # Create datasets
        print('5. Creating test datasets')
        # testset = ArrayDataset(test_data, labels=torch.zeros((len(testfiles),)), root=root, do_transform=True)
        testset = FolderDataset(os.path.join(root, 'test'), labels=torch.zeros((len(testfiles),)), do_transform=True)

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
    submission_df['infected'] = test_outputs
    submission_df.to_csv(os.path.join(root, suffix))
