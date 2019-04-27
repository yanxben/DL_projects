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

import matplotlib.pyplot as plt
import torch

from models.encoder_decoder import Generator
from data.data_caltech_ucsd import create_dataset_caltech_ucsd
from models.utils_save import load_model


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
depth = 5
extract = [1, 3, depth]
epochs = 1
data_mode = 'cropped'

load_dir1 = './checkpoints/final/Light_Background_5_ReID_02_Disc_02_range_512_wd_gc_mask'
load_filename1 = 'net_Gen_save_6_epoch_660.pth.tar'
load_dir2 = './checkpoints/final/Light_Background_5_ReID_02_Disc_02_range_512_wd_gc_mask'
load_filename2 = 'net_Gen_save_2_epoch_720.pth.tar'
load_dir3 = './checkpoints/final/Light_ReID_02_Disc_02_background_2_range_mask'
load_filename3 = 'net_Gen_save_5_epoch_450.pth.tar'
load_dir4 = './checkpoints/final/ReID_01_Disc_01_range_512_mask'
load_filename4 = 'net_Gen_save_3_epoch_430.pth.tar'


def run(model, images):

    mask = images[:, :, C - 1, :, :].view([-1, 2, 1, H, W])
    images = images[:, :, :C - 1, :, :].view([-1, 2, C - 1, H, W])
    images = im2tensor(images)

    # Run train iteration
    output = model(images.cuda(), mask_in=mask.cuda())

    # Keep last examples
    images = tensor2im(images)
    output = tensor2im(output.detach().cpu())

    # print statistics
    for m in range(images.shape[0]):
        plt.subplot(images.shape[0], 4, 4 * m + 1)
        plt.imshow(images[m, 0].permute([1, 2, 0]))
        plt.subplot(images.shape[0], 4, 4 * m + 2)
        plt.imshow(images[m, 1].permute([1, 2, 0]))
        plt.subplot(images.shape[0], 4, 4 * m + 3)
        plt.imshow(output[m, 0].permute([1, 2, 0]))
        plt.subplot(images.shape[0], 4, 4 * m + 4)
        plt.imshow(output[m, 1].permute([1, 2, 0]))
    plt.draw()
    plt.pause(0.001)
    plt.show(block=False)


if __name__ == '__main__':

    # Load model
    model1 = Generator(5, 3, 512, 512, 512, imsize, depth=depth, extract=extract)
    model2 = Generator(5, 3, 512, 512, 512, imsize, depth=depth, extract=extract)
    model3 = Generator(5, 3, 512, 512, 512, imsize, depth=depth, extract=extract)
    model4 = Generator(5, 3, 512, 512, 512, imsize, depth=depth, extract=extract)
    # model = GeneratorHeavy(5, 3, 512, 512, 512, imsize, depth=depth, preprocess=False, extract=extract)

    load_path1 = os.path.join(load_dir1, load_filename1)
    load_model(model1, load_path1)
    model1.cuda()
    load_path2 = os.path.join(load_dir2, load_filename2)
    load_model(model2, load_path2)
    model2.cuda()
    load_path3 = os.path.join(load_dir3, load_filename3)
    load_model(model3, load_path3)
    model3.cuda()
    load_path4 = os.path.join(load_dir4, load_filename4)
    load_model(model4, load_path4)
    model4.cuda()

    if socket.gethostname() == 'YABENN-P50':
        caltech_path = 'C:/Datasets/Caltech-UCSD-Birds-200'
    elif os.sys.platform == 'linux':
        caltech_path = '/home/' + os.getlogin() + '/Datasets/Caltech-UCSD-Birds-200'
    _, caltech_data, caltech_meta, testset = create_dataset_caltech_ucsd(caltech_path, batch_size, size=None, imsize=imsize, mode=data_mode, testset=testset)
    caltech_labels = caltech_meta['labels']
    caltech_bboxes = caltech_meta['bboxes']
    print('The number of training images = %d' % caltech_data.shape[0])

    plt.figure('test1', figsize=(1920 / 100, 1080 / 100), dpi=100)
    plt.figure('test2', figsize=(1920 / 100, 1080 / 100), dpi=100)
    plt.figure('test3', figsize=(1920 / 100, 1080 / 100), dpi=100)
    plt.figure('test4', figsize=(1920 / 100, 1080 / 100), dpi=100)

    print('End of initialization.')

    _, C, H, W = testset['images'].shape
    for i in range(testlen):
        images = torch.zeros([testlen, 2, C, H, W])
        images[:, 0] = testset['images'][i].unsqueeze(0).expand(testlen, C, H, W)
        images[:, 1] = torch.cat([testset['images'][j].unsqueeze(0) for j in range(testlen)], dim=0)

        plt.figure('test1')
        run(model1, images)
        plt.figure('test2')
        run(model2, images)
        plt.figure('test3')
        run(model3, images)
        plt.figure('test4')
        run(model4, images)

    print('DONE')
