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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn

from models.encoder_decoder import Discriminator2
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
if __name__ == '__main__':
    t0 = time.time()
    opt = TrainOptions().parse()   # get training options
    dataset, caltech_data = create_dataset_caltech_ucsd(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training epochs = %d' % dataset_size)
    print('The number of training images = %d' % caltech_data.shape[0])

    stl10_data_test = caltech_data[testset]
    # model_test_input = {'real_G': torch.Tensor(stl10_data_test[:, :3, :, :]),
    #                     'mask_G': torch.Tensor(stl10_data_test[:, 3, :, :]),
    #                     'real_D': torch.Tensor(stl10_data_test[:, :3, :, :]),
    #                     'mask_D': torch.Tensor(stl10_data_test[:, 3, :, :])}
    #
    # plt.figure('test merge', figsize=(1920 / 100, 1080 / 100), dpi=100)
    # plt.figure('test reflect', figsize=(1920 / 100, 1080 / 100), dpi=100)
    #
    # plt.figure('test merge')
    # for i in range(testlen//2):
    #     plt.subplot(testlen//2, 6, i * 6 + 1)
    #     plt.imshow(model_test_input['real_G'][2*i].permute([1, 2, 0]))
    #     plt.subplot(testlen//2, 6, i * 6 + 2)
    #     plt.imshow(model_test_input['real_G'][2*i + 1].permute([1, 2, 0]))
    #     plt.subplot(testlen//2, 6, i * 6 + 3)
    #     plt.subplot(testlen//2, 6, i * 6 + 4)
    #     plt.subplot(testlen//2, 6, i * 6 + 5)
    #     plt.subplot(testlen//2, 6, i * 6 + 6)
    if not os.path.isdir(os.path.join(opt.plots_dir, opt.name)):
        os.mkdir(os.path.join(opt.plots_dir, opt.name))

    # Declare model, optimizer, loss
    model = Discriminator2(3, 512, 96)      # create a model given opt.model and other options
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()

    print('End of initialization. Time Taken: %d sec' % (time.time() - t0))

    total_iters = 0  # the total number of training iterations
    for epoch in range(1, 100):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        num_iterations = 0
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if data['image'].shape[0] < opt.batch_size:
                continue  # ignore tail of data if tail is smaller than batch_size

            images_a = data['images']
            images_b = torch.zeros_like(images_a)
            for k in range(opt.batch_size):
                label_a = data['labels'][k]
                if random.randint(0,1) < 0.5:
                    indices = (caltech_data['labels'] == label_a).non_zero().squeeze()
                    data['labels'][k] = 1
                else:
                    indices = (caltech_data['labels'] != label_a).non_zero().squeeze()
                    data['labels'][k] = 0
                idx = indices[torch.randint(indices.numel(), (1,))]
                images_b[k] = caltech_data['images'][idx]

            data['images'] = torch.cat([images_a.unsqueeze(1), images_b.unsqueeze(1)], dim=1)
            total_iters += 1
            epoch_iter += 1

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Run train iteration
            yhat = model(data['images'])
            loss = criterion(yhat, data['labels'])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_iterations += 1

        # cache our model every <save_epoch_freq> epochs
        if epoch % 1 == 0:
            # print statistics
            print('Epoch {:d}: loss {:.4f}'.format(epoch, running_loss / num_iterations))
            running_loss = 0

            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            #save_suffix = 'epoch_%d' % epoch
            #model.save_networks(save_suffix)

    print('DONE')

