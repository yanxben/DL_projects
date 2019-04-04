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

from models.encoder_decoder import DiscriminatorReID, DiscriminatorTriplet, Classifier200, Generator, AutoEncoder2
from options.train_options import TrainOptions
from data.data_caltech_ucsd import create_dataset_caltech_ucsd
from models import create_model

#from util.visualizer import Visualizer


def tensor2im(t):
    return (t / 2) + 0.5


def im2tensor(i):
    return (i - 0.5) * 2


testset = ['Red_winged_Blackbird_0017_583846699', 'Yellow_headed_Blackbird_0009_483173184',
     'Lazuli_Bunting_0010_522399154', 'Painted_Bunting_0006_2862481106',
     'Gray_Catbird_0031_148467783', 'Purple_Finch_0006_2329434675', 'American_Goldfinch_0004_155617438',
     'Blue_Grosbeak_0008_2450854752', 'Green_Kingfisher_0002_228927324', 'Pied_Kingfisher_0002_1020026028']
testlen = len(testset)
batch_size = 64
imsize = 128
depth = 6
plot = True
#mode = 'classification'
#mode = 're-identification'
mode = 'autoencoder'



if __name__ == '__main__':
    t0 = time.time()
    #opt = TrainOptions().parse()   # get training options
    _, caltech_data, caltech_labels, testset = create_dataset_caltech_ucsd('C:/Datasets/Caltech-UCSD-Birds-200', batch_size, imsize=imsize, size=256, testset=testset)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    #print('The number of training epochs = %d' % dataset_size)
    print('The number of training images = %d' % caltech_data.shape[0])

    if mode == 'autoencoder':
        plt.figure(mode, figsize=(1920 / 100, 1080 / 100), dpi=100)


    # Declare model, optimizer, loss
    if mode == 'classification':
        model = Classifier200(caltech_data.shape[1], 512, imsize, depth=depth)
        criterion = nn.CrossEntropyLoss()
    if mode=='re-identification':
        caltech_data = caltech_data.type(torch.FloatTensor)
        model = DiscriminatorReID(caltech_data.shape[2], 512, imsize, depth=depth)
        criterion = nn.TripletMarginLoss()
    if mode=='autoencoder':
        model = Generator(caltech_data.shape[1], 3, 512, 256, imsize, depth=depth, preprocess=False)
        criterion = nn.MSELoss()

    model.cuda()
    optimizer = optim.Adam(model.parameters())

    print('End of initialization. Time Taken: %d sec' % (time.time() - t0))

    total_iters = 0  # the total number of training iterations
    for epoch in range(1, 400):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        num_iterations = 0
        running_loss = 0
        for i, batch in enumerate(torch.randperm(caltech_data.shape[0]).split(batch_size)): # enumerate(dataset):  # inner loop within one epoch
            if batch.shape[0] < batch_size:
                continue  # ignore tail of data if tail is smaller than batch_size

            if mode=='re-identification':
                N, C, H, W = caltech_data.shape
                batch_p = torch.zeros_like(batch)
                batch_n = torch.zeros_like(batch)
                for k in range(batch_size):
                    label_a = caltech_labels[batch[k]]
                    indices = (caltech_labels == label_a).nonzero()
                    batch_p[k] = indices[torch.randint(indices.numel(), (1,))]
                    indices = (caltech_labels != label_a).nonzero()
                    batch_n[k] = indices[torch.randint(indices.numel(), (1,))]

                #images = torch.cat([caltech_data[batch, :C - 1], caltech_data[batch_p, :C - 1], caltech_data[batch_n, :C - 1]], dim=1)
                images_a, images_p, images_n = caltech_data[batch, :C-1], caltech_data[batch_p, :C-1], caltech_data[batch_n, :C-1]
                images = im2tensor(images)

            if mode=='autoencoder':
                N, C, H, W = caltech_data.shape
                labels = caltech_data[batch, :C-1, :, :]
                mask = caltech_data[batch, C-1, :, :].unsqueeze(1).expand([batch_size, 2, H, W])
                images = caltech_data[batch, :C-1, :, :].unsqueeze(1).expand([batch_size, 2, C-1, H, W])
                images = im2tensor(images)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Run train iteration
            if mode == 're-identification':
                embed_a, embed_p, embed_n = model(images_a.cuda()), model(images_p.cuda()), model(images_n.cuda())
                loss = criterion(embed_a, embed_p, embed_n)
            if mode == 'autoencoder':
                yhat = model(images.cuda(),
                             mask_in=mask.cuda(), mode=0)
                loss = criterion(yhat, labels.cuda())

            # Run backward
            loss.backward()
            optimizer.step()

            total_iters += 1
            epoch_iter += 1
            running_loss += loss.item()
            num_iterations += 1

            # Keep last examples
            last_data_images = tensor2im(images)
            last_yhat = tensor2im(yhat.detach().cpu())

        # cache our model every <save_epoch_freq> epochs
        if epoch % 1 == 0:
            # print statistics
            print('Epoch {:d}: loss {:.4f}'.format(epoch, running_loss / num_iterations))

            #print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            #save_suffix = 'epoch_%d' % epoch
            #model.save_networks(save_suffix)
            if plot:
                if epoch % 1 == 0:
                    if mode=='autoencoder':
                        for m in range(6):
                            plt.subplot(3, 6, m + 1)
                            plt.imshow(last_data_images[m,0].permute([1, 2, 0]))
                            plt.subplot(3, 6, m + 7)
                            plt.imshow(last_yhat[m,0].permute([1, 2, 0]))
                            plt.subplot(3, 6, m + 13)
                            plt.imshow(last_yhat[m,1].permute([1, 2, 0]))
                    if mode == 're-identification':
                        for m in range(4):
                            plt.subplot(3, 4, m + 1)
                            plt.imshow(last_data_images[m, 0].permute([1, 2, 0]))
                            plt.subplot(3, 4, m + 5)
                            plt.imshow(last_data_images[m, 1].permute([1, 2, 0]))
                            plt.subplot(3, 4, m + 9)
                            plt.imshow(last_data_images[m, 2].permute([1, 2, 0]))
                    plt.suptitle('Epoch {:d}',format(epoch))
                    plt.draw()
                    plt.pause(0.001)
                    plt.show(block=False)

    print('DONE')

