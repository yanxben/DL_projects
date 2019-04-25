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
import socket
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn

from models.encoder_decoder import DiscriminatorReID, DiscriminatorTriplet, Classifier200, Generator, GeneratorHeavy, AutoEncoder2, EHeavy, DecoderHeavy
from data.data_caltech_ucsd import create_dataset_caltech_ucsd, crop_data
from models.utils_save import save_model


def tensor2im(t):
    return (t / 2) + 0.5


def im2tensor(i):
    return (i - 0.5) * 2


testset = ['Red_winged_Blackbird_0017_583846699', 'Yellow_headed_Blackbird_0009_483173184',
     'Lazuli_Bunting_0010_522399154', 'Painted_Bunting_0006_2862481106',
     'Gray_Catbird_0031_148467783', 'Purple_Finch_0006_2329434675', 'American_Goldfinch_0004_155617438',
     'Blue_Grosbeak_0008_2450854752', 'Green_Kingfisher_0002_228927324', 'Pied_Kingfisher_0002_1020026028']
testlen = len(testset)
batch_size = 16
imsize = 96
depth = 4
extract = [1, depth]
epochs = 500

#model_mode = 'classification'
#model_mode = 're-identification'
#model_mode = 'autoencoder'
model_mode = 'encoderdecoder'
data_mode = 'cropped'
save_dir = './checkpoints/encoder4BatchL1'
save_filename = 'encoder.pth.tar'


if __name__ == '__main__':
    t0 = time.time()
    #opt = TrainOptions().parse()   # get training options
    if socket.gethostname() == 'YABENN-P50':
        plot = True
        caltech_path = 'C:/Datasets/Caltech-UCSD-Birds-200'
    elif os.sys.platform == 'linux':
        plot = False
        caltech_path = '/home/' + os.getlogin() + '/Datasets/Caltech-UCSD-Birds-200'
    _, caltech_data, caltech_meta, testset = create_dataset_caltech_ucsd(caltech_path, batch_size, size=None, imsize=imsize, mode=data_mode, testset=testset)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    #print('The number of training epochs = %d' % dataset_size)
    caltech_labels = caltech_meta['labels']
    caltech_bboxes = caltech_meta['bboxes']
    print('The number of training images = %d' % caltech_data.shape[0])

    validationset = torch.zeros(2 * len(set(caltech_labels.numpy())), dtype=torch.long)
    trainset = torch.zeros(caltech_labels.shape[0] - validationset.shape[0], dtype=torch.long)
    validationsize = 0
    trainsize = 0
    label_dict = dict()
    for label in set(caltech_labels.numpy()):
        indices_p = ((caltech_labels == label).nonzero()[:, 0]).type(torch.long)
        label_dict[label] = {1: indices_p[:-2]}
        validationset[validationsize:validationsize+2] = indices_p[-2:]
        trainset[trainsize:trainsize + indices_p.numel() - 2] = indices_p[:-2]
        trainsize += indices_p.numel() - 2
        validationsize += 2
    for label in set(caltech_labels.numpy()):
        indices_n = list(trainset.numpy())
        for idx in list(label_dict[label][1].numpy()):
            indices_n.remove(idx)
        label_dict[label][0] = torch.tensor(indices_n, dtype=torch.long)
    assert validationset.shape[0] == validationsize, "Bad validation generation"

    if model_mode == 'autoencoder':
        plt.figure(model_mode, figsize=(1920 / 100, 1080 / 100), dpi=100)
    if model_mode == 're-identification':
        plt.figure(model_mode, figsize=(1920 / 100, 1080 / 100), dpi=100)

    # Declare model, optimizer, loss
    if model_mode == 'classification':
        model = Classifier200(caltech_data.shape[1], 512, imsize, depth=depth)
        criterion = nn.CrossEntropyLoss()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
    if model_mode=='re-identification':
        caltech_data = caltech_data.type(torch.FloatTensor)
        model = DiscriminatorReID(caltech_data.shape[1] - 1, 512, imsize, depth=depth, out_features=64, dropout=0.1)
        criterion = nn.TripletMarginLoss()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
    if model_mode=='autoencoder':
        model = GeneratorHeavy(5, 3, 512, 512, 512, imsize, depth=depth, preprocess=False, extract=extract)
        criterion = nn.MSELoss()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
    if model_mode=='encoderdecoder':
        #model = GeneratorHeavy(5, 3, 512, 512, 512, imsize, depth=depth, preprocess=False, extract=extract)
        encoder = EHeavy(5, 512, imsize, depth=depth)
        decoder = DecoderHeavy(3, 512, imsize, depth=depth, extract=extract)
        criterion = nn.L1Loss()
        encoder.cuda()
        decoder.cuda()
        optimizerE = optim.Adam(encoder.parameters(), lr=0.0002)
        optimizerD = optim.Adam(decoder.parameters(), lr=0.0002)

    print('End of initialization. Time Taken: %d sec' % (time.time() - t0))

    total_iters = 0  # the total number of training iterations
    plot_start_time = time.time()
    for epoch in range(1, epochs):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        num_iterations = 0
        running_loss = 0
        for i, batch_idx in enumerate(torch.randperm(trainset.shape[0]).split(batch_size)): # enumerate(dataset):  # inner loop within one epoch
            if batch_idx.shape[0] < batch_size:
                continue  # ignore tail of data if tail is smaller than batch_size
            batch = trainset[batch_idx]

            if model_mode == 're-identification':
                _, C, H, W = caltech_data.shape
                # Collect positive anc negative
                batch_p = torch.zeros_like(batch)
                batch_n = torch.zeros_like(batch)
                for k in range(batch_size):
                    label_a = int(caltech_labels[batch[k]])
                    indices = label_dict[label_a][1]
                    batch_p[k] = indices[torch.randint(indices.numel(), (1,))]
                    indices = label_dict[label_a][0]
                    batch_n[k] = indices[torch.randint(indices.numel(), (1,))]

                images_a, images_p, images_n = caltech_data[batch, :C-1], caltech_data[batch_p, :C-1], caltech_data[batch_n, :C-1]

                # Preprocess
                images_a = caltech_data[batch][:, :3]
                images_p = caltech_data[batch_p][:, :3]
                images_n = caltech_data[batch_n][:, :3]
                if data_mode == 'range':
                    bboxes = [caltech_bboxes[b] for b in batch]
                    images_a = crop_data(images_a, bboxes, imsize)
                    bboxes = [caltech_bboxes[b] for b in batch_p]
                    images_p = crop_data(images_p, bboxes, imsize)
                    bboxes = [caltech_bboxes[b] for b in batch_n]
                    images_n = crop_data(images_n, bboxes, imsize)

                flip = torch.randint(2, (batch_size, 1, 1, 1)).expand_as(images_a)
                images_a = torch.where(flip == 1, images_a.flip(3), images_a)
                flip = torch.randint(2, (batch_size, 1, 1, 1)).expand_as(images_p)
                images_p = torch.where(flip == 1, images_p.flip(3), images_p)
                flip = torch.randint(2, (batch_size, 1, 1, 1)).expand_as(images_a)
                images_n = torch.where(flip == 1, images_n.flip(3), images_n)
                images_a, images_p, images_n = im2tensor(images_a), im2tensor(images_p), im2tensor(images_n)

            if model_mode == 'autoencoder':
                _, C, H, W = caltech_data.shape
                images = caltech_data[batch, :, :, :]
                if data_mode == 'range':
                    bboxes = [caltech_bboxes[b] for b in batch]
                    images = crop_data(images, bboxes, imsize)

                flip = torch.randint(2, (batch_size, 1, 1, 1)).expand_as(images)
                images = torch.where(flip == 1, images.flip(3), images)
                labels = images[:, :C-1, :, :]
                mask = images[:, C-1, :, :].unsqueeze(1).expand([batch_size, 2, H, W])
                images = images[:, :C-1, :, :].unsqueeze(1).expand([batch_size, 2, C-1, H, W])
                images = im2tensor(images)
                # Particularly in this test we want to mask the object in image A to achieve recreation using deep features of image B
                images[:,0] = torch.where(mask[:,0].unsqueeze(1).expand_as(images[:,0]) > .5, torch.zeros_like(images[:,0]), images[:,0])

            if model_mode=='encoderdecoder':
                _, C, H, W = caltech_data.shape
                images = caltech_data[batch, :, :, :]
                if data_mode == 'range':
                    bboxes = [caltech_bboxes[b] for b in batch]
                    images = crop_data(images, bboxes, imsize)

                flip = torch.randint(2, (batch_size, 1, 1, 1)).expand_as(images)
                images = torch.where(flip == 1, images.flip(3), images)
                labels = images[:, :C - 1, :, :]
                images = torch.cat((images, 1 - images[:, C-1, :, :].unsqueeze(1)), dim=1)
                images[:, :C-1, :, :] = im2tensor(images[:, :C-1, :, :])

            # Run train iteration
            if model_mode == 're-identification':
                optimizer.zero_grad()  # Zero the parameter gradients
                embed_a, embed_p, embed_n = model(images_a.cuda()), model(images_p.cuda()), model(images_n.cuda())
                loss = criterion(embed_a, embed_p, embed_n)
                # Run backward
                loss.backward()
                optimizer.step()
            if model_mode == 'autoencoder':
                optimizer.zero_grad()  # Zero the parameter gradients
                yhat = model(images.cuda(), mask_in=mask.cuda(), mode=0, extract=extract, use_activation=True)
                loss = criterion(yhat, labels.cuda())
                # Run backward
                loss.backward()
                optimizer.step()
            if model_mode == 'encoderdecoder':
                optimizerE.zero_grad()  # Zero the parameter gradients
                optimizerD.zero_grad()  # Zero the parameter gradients
                z = encoder(images.cuda(), extract=extract)
                z[1] = torch.where(images[:, -1, ::2, ::2].cuda().unsqueeze(1).expand_as(z[1]) >= 0.5, z[1], torch.zeros_like(z[1]))
                yhat = decoder(z)
                loss = criterion(yhat, labels.cuda())
                # Run backward
                loss.backward()
                optimizerE.step()
                optimizerD.step()

            total_iters += 1
            epoch_iter += 1
            running_loss += loss.item()
            num_iterations += 1

            # Keep last examples
            if model_mode in ['autoencoder', 'encoderdecoder']:
                last_data_images = tensor2im(images)
                last_yhat = tensor2im(yhat.detach().cpu())
            if model_mode == 're-identification':
                last_data_images = tensor2im(torch.cat([images_a.unsqueeze(1), images_p.unsqueeze(1), images_n.unsqueeze(1)], dim=1).detach().cpu())

        # cache our model every <save_epoch_freq> epochs
        if epoch % 10 == 0:
            # print statistics
            with torch.no_grad():
                if model_mode == 're-identification':
                    _, C, H, W = caltech_data.shape
                    batch_p = torch.zeros_like(validationset)
                    batch_n = torch.zeros_like(validationset)
                    for k in range(validationsize):
                        label_a = int(caltech_labels[validationset[k]])
                        # indices = (caltech_labels == label_a).nonzero()
                        indices = label_dict[label_a][1]
                        batch_p[k] = indices[torch.randint(indices.numel(), (1,))]
                        # indices = (caltech_labels != label_a).nonzero()
                        indices = label_dict[label_a][0]
                        batch_n[k] = indices[torch.randint(indices.numel(), (1,))]

                    images_a, images_p, images_n = caltech_data[validationset, :C - 1], caltech_data[batch_p, :C - 1], caltech_data[batch_n, :C - 1]
                    images_a, images_p, images_n = im2tensor(images_a), im2tensor(images_p), im2tensor(images_n)

                    embed_a, embed_p, embed_n = model(images_a.cuda()), model(images_p.cuda()), model(images_n.cuda())
                    val_loss = criterion(embed_a, embed_p, embed_n)

                if model_mode == 'autoencoder':
                    _, C, H, W = caltech_data.shape
                    images = caltech_data[validationset]
                    labels = images[:, :C - 1, :, :]
                    mask = images[:, C - 1, :, :].unsqueeze(1).expand([validationset.shape[0], 2, H, W])
                    images = images[:, :C - 1, :, :].unsqueeze(1).expand([validationset.shape[0], 2, C - 1, H, W])
                    images = im2tensor(images)
                    # Mask image A in the same manner as is done in training
                    images[:, 0] = torch.where(mask[:, 0].unsqueeze(1).expand_as(images[:, 0]) > .5,
                                               torch.zeros_like(images[:, 0]), images[:, 0])

                    yhat = model(images.cuda(), mask_in=mask.cuda(), mode=0, extract=extract, use_activation=True)
                    val_loss = criterion(yhat, labels.cuda())

                if model_mode == 'encoderdecoder':
                    _, C, H, W = caltech_data.shape
                    images = caltech_data[validationset]
                    images = torch.cat((images, 1 - images[:, C-1, :, :].unsqueeze(1)), dim=1)
                    labels = images[:, :C - 1, :, :]
                    images[:, :C - 1, :, :] = im2tensor(images[:, :C - 1, :, :])

                    z = encoder(images.cuda(), extract=extract)
                    z[1] = torch.where(images[:, -1, ::2, ::2].cuda().unsqueeze(1).expand_as(z[1]) >= 0.5, z[1], torch.zeros_like(z[1]))
                    yhat = decoder(z)
                    val_loss = criterion(yhat, labels.cuda())

            print('Epoch {:d}: train loss {:.4f} --- val loss {:.4f}'.format(epoch, running_loss / num_iterations, val_loss))

            if plot:
                if model_mode == 'autoencoder':
                    for m in range(6):
                        plt.subplot(4, 6, m + 1)
                        plt.imshow(last_data_images[m, 1].permute([1, 2, 0]))
                        plt.subplot(4, 6, m + 7)
                        plt.imshow(last_yhat[m].permute([1, 2, 0]))
                        plt.subplot(4, 6, m + 13)
                        plt.imshow(tensor2im(images)[5*m, 1].permute([1, 2, 0]))
                        plt.subplot(4, 6, m + 19)
                        plt.imshow(tensor2im(yhat.detach().cpu())[5*m].permute([1, 2, 0]))
                if model_mode == 'encoderdecoder':
                    for m in range(6):
                        plt.subplot(4, 6, m + 1)
                        plt.imshow(last_data_images[m, :C-1].permute([1, 2, 0]))
                        plt.subplot(4, 6, m + 7)
                        plt.imshow(last_yhat[m].permute([1, 2, 0]))
                        plt.subplot(4, 6, m + 13)
                        plt.imshow(tensor2im(images)[5*m, :C-1].permute([1, 2, 0]))
                        plt.subplot(4, 6, m + 19)
                        plt.imshow(tensor2im(yhat.detach().cpu())[5*m].permute([1, 2, 0]))
                if model_mode == 're-identification':
                    for m in range(4):
                        plt.subplot(3, 4, m + 1)
                        plt.imshow(last_data_images[m, 0].permute([1, 2, 0]))
                        plt.subplot(3, 4, m + 5)
                        plt.imshow(last_data_images[m, 1].permute([1, 2, 0]))
                        plt.subplot(3, 4, m + 9)
                        plt.imshow(last_data_images[m, 2].permute([1, 2, 0]))
                plt.suptitle('Epoch {:d}'.format(epoch))
                plt.draw()
                plt.pause(0.001)
                plt.show(block=False)

            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_path = os.path.join(save_dir, save_filename)
            save_model(encoder, save_path, optimizer=optimizerE)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, epochs, time.time() - plot_start_time))
            plot_start_time = time.time()

    print('DONE')

