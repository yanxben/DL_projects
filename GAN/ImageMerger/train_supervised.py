"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
import time
import random

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn

from models.encoder_decoder import DiscriminatorPair, DiscriminatorTriplet, Classifier200, Generator, AutoEncoder2
from options.train_options import TrainOptions
from data.data_caltech_ucsd import create_dataset_caltech_ucsd
from models import create_model

#from util.visualizer import Visualizer


def tensor2im(t):
    return (t / 2) + 0.5


def im2tensor(i):
    return (i - 0.5) * 2


testset = list(range(10)) # [0, 4, 5, 6, 206, 210, 213, 405, 407, 435]
testlen = len(testset)
batch_size = 64
imsize = 128
depth = 6

#mode = 'classification'
mode = 're-identification'
#mode = 'autoencoder'


if __name__ == '__main__':
    t0 = time.time()
    #opt = TrainOptions().parse()   # get training options
    dataset, caltech_data, caltech_labels = create_dataset_caltech_ucsd('C:/Datasets/Caltech-UCSD-Birds-200', batch_size, imsize=imsize)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training epochs = %d' % dataset_size)
    print('The number of training images = %d' % caltech_data.shape[0])

    plt.figure('test merge')


    # Declare model, optimizer, loss
    if mode == 'classification':
        model = Classifier200(caltech_data.shape[1], 512, imsize, depth=depth)
        criterion = nn.CrossEntropyLoss()
    if mode=='re-identification':
        model = DiscriminatorTriplet(caltech_data.shape[1], 512, imsize, depth=depth)
        criterion = nn.TripletMarginLoss()
    if mode=='autoencoder':
        model = AutoEncoder2(caltech_data.shape[1], 3, 512, 256, imsize, depth=depth, preprocess=False)
        criterion = nn.MSELoss()

    model.cuda()
    optimizer = optim.Adam(model.parameters())

    print('End of initialization. Time Taken: %d sec' % (time.time() - t0))

    total_iters = 0  # the total number of training iterations
    for epoch in range(1, 100):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        num_iterations = 0
        running_loss = 0
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if data['images'].shape[0] < batch_size:
                continue  # ignore tail of data if tail is smaller than batch_size

            if mode=='re-identification':
                images_a = data['images']
                images_p = torch.zeros_like(images_a)
                images_n = torch.zeros_like(images_a)
                for k in range(batch_size):
                    label_a = data['labels'][k]
                    #if random.randint(0,1) < 0.5:
                    indices = (caltech_labels == label_a).nonzero()
                    idx = indices[torch.randint(indices.numel(), (1,))]
                    images_p[k] = caltech_data[idx]
                    # data['labels'][k] = 1
                    #else:
                    indices = (caltech_labels != label_a).nonzero()
                    idx = indices[torch.randint(indices.numel(), (1,))]
                    images_n[k] = caltech_data[idx]
                    #data['labels'][k] = 0

                    #idx = indices[torch.randint(indices.numel(), (1,))]
                    #images_b[k] = caltech_data[idx]
                    data['images'] = torch.cat([images_a.unsqueeze(1), images_p.unsqueeze(1), images_n.unsqueeze(1)], dim=1)
                    data['images'] = im2tensor(data['images'])
                    # for m in range(6):
                    #     plt.subplot(2, 6, m + 1)
                    #     plt.imshow(images_a[m].permute([1, 2, 0]))
                    #     plt.subplot(2, 6, m + 7)
                    #     plt.imshow(images_b[m].permute([1, 2, 0]))

            if mode=='autoencoder':
                N, C, H, W = data['images'].shape
                data['labels'] = data['images'][:, :C-1, :, :]
                data['mask'] = data['images'][:, C-1, :, :].unsqueeze(1).expand([N, 2, H, W])
                data['images'] = data['images'][:, :C-1, :, :].unsqueeze(1).expand([N, 2, C-1, H, W])
                data['images'] = im2tensor(data['images'])

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Run train iteration
            if mode == 're-identification':
                yhat = model(data['images'].type(torch.cuda.FloatTensor))
                loss = criterion(yhat[:,0], yhat[:,1], yhat[:,2])
            if mode == 'autoencoder':
                yhat = model(data['images'].type(torch.cuda.FloatTensor),
                             mask_in=data['mask'].type(torch.cuda.FloatTensor), mode=0)
                loss = criterion(yhat, data['labels'].type(torch.cuda.FloatTensor))
            loss.backward()
            optimizer.step()

            total_iters += 1
            epoch_iter += 1
            running_loss += loss.item()
            num_iterations += 1

            # Keep last examples
            last_data_images = tensor2im(data['images'])
            last_yhat = tensor2im(yhat.detach().cpu())

        # cache our model every <save_epoch_freq> epochs
        if epoch % 1 == 0:
            # print statistics
            print('Epoch {:d}: loss {:.4f}'.format(epoch, running_loss / num_iterations))

            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            #save_suffix = 'epoch_%d' % epoch
            #model.save_networks(save_suffix)
            if epoch % 1 == 0:
                if mode=='autoencoder':
                    for m in range(6):
                         plt.subplot(2, 6, m + 1)
                         plt.imshow(last_data_images[m,0].permute([1, 2, 0]))
                         plt.subplot(2, 6, m + 7)
                         plt.imshow(last_yhat[m].permute([1, 2, 0]))
                    plt.draw()
                    plt.pause(0.001)
                    plt.show(block=False)

    print('DONE')

