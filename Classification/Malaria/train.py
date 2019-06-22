import os

import numpy as np
import torch
import torch.nn

import torchvision.datasets.folder

from models.networks import VGG
from data_utils.data_util import load_data


## Parameters
train_val_rate = 0.8
batch_size = 64
epochs = 100

# Define model
print('Building model')
model = VGG(32, num_classes=1).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# Load data_utils
print('Loading data_utils')
data_train, data_validation = load_data('/media/yaniv/data2/datasets/Malaria', train_val_rate, batch_size)

# Train
print('Training')
for t in range(1, epochs+1):

    running_loss = 0
    train_size = 0
    for batch_idx, batch_data in enumerate(data_train, 1):
        images, labels = batch_data['images'].cuda(), batch_data['labels'].cuda()
        if labels.shape[0] < batch_size:
            continue
        outputs = model(images)

        #outputs = outputs.view(-1, 2)
        #labels = labels.view(-1, ).long()
        loss = criterion(outputs, labels)  # Compute the loss
        running_loss += loss.items() * labels.shape[0]
        train_size += labels.shape[0]

    if t % 10:
        validation_loss = 0
        validation_size = 0
        for batch_idx, batch_data in enumerate(data_validation, 1):
            images, labels = batch_data['images'].cuda(), batch_data['labels']

            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute the loss
            validation_loss += loss.item() * labels.shape[0]
            validation_size += labels.shape[0]

        print('Epoch {}: train loss: {} validation_loss: {}'.format(t, running_loss) / train_size, validation_loss / validation_size)
