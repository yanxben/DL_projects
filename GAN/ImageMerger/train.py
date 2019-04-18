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
from options.train_options import TrainOptions
from data.data_stl10 import create_dataset_stl10_bird
from data.data_caltech_ucsd import create_dataset_caltech_ucsd, crop_data
from models import create_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
#from util.visualizer import Visualizer


def tensor2im(t):
    return (t / 2) + 0.5


def im2tensor(i):
    return (i - 0.5) * 2


testset = ['Red_winged_Blackbird_0017_583846699', 'Yellow_headed_Blackbird_0009_483173184',
           'Lazuli_Bunting_0010_522399154', 'Painted_Bunting_0006_2862481106',
           'Gray_Catbird_0031_148467783', 'Purple_Finch_0006_2329434675',
           'American_Goldfinch_0004_155617438', 'Blue_Grosbeak_0008_2450854752',
           'Green_Kingfisher_0002_228927324', 'Pied_Kingfisher_0002_1020026028']

if __name__ == '__main__':
    t0 = time.time()
    opt = TrainOptions().parse()   # get training options
    #dataset, stl10_data = create_dataset_stl10_bird(opt)  # create a dataset given opt.dataset_mode and other options
    if socket.gethostname() == 'YABENN-P50':
        caltech_path = 'C:/Datasets/Caltech-UCSD-Birds-200'
    elif os.sys.platform == 'linux':
        caltech_path = '/home/' + os.getlogin() + '/Datasets/Caltech-UCSD-Birds-200'
    _, caltech_data, caltech_meta, testset = create_dataset_caltech_ucsd(caltech_path, opt.batch_size, size=opt.data_size ,mode=opt.data_mode, imsize=opt.input_size, testset=testset)  # create a dataset given opt.dataset_mode and other options

    caltech_labels = caltech_meta['labels']
    caltech_bboxes = caltech_meta['bboxes']

    #dataset_size = caltech_data.shape[0]  #len(dataset)    # get the number of images in the dataset.
    dataset_size, C, H, W = caltech_data.shape
    print('The number of training epochs = %d' % (dataset_size // opt.batch_size))
    print('The number of training images = %d' % dataset_size)

    if opt.data_mode == 'range':
        testset['images'] = crop_data(testset['images'], testset['bboxes'], opt.input_size)
    testlen = testset['images'].shape[0]

    model_test_input = {'real_G': testset['images'][:, :3, :, :].reshape([-1, 2, 3, opt.input_size, opt.input_size]),
                        'mask_G': testset['images'][:, 3, :, :].unsqueeze(1).unsqueeze(1).reshape([-1, 2, 1, opt.input_size, opt.input_size]),
                        'real_D': testset['images'][:, :3, :, :],
                        'mask_D': testset['images'][:, 3, :, :].unsqueeze(1),
                        'real_a': testset['images'][:testlen//2, :3, :, :],                 # Unused
                        'mask_a': testset['images'][:testlen // 2, 3, :, :].unsqueeze(1),   # Unused
                        'real_n': testset['images'][:testlen//2, :3, :, :],                 # Unused
                        'mask_n': testset['images'][:testlen//2, 3, :, :].unsqueeze(1)      # Unused
                        }

    #caltech_data = caltech_data.cuda()
    plt.figure('test merge', figsize=(1920 / 100, 1080 / 100), dpi=100)
    plt.figure('test reflect', figsize=(1920 / 100, 1080 / 100), dpi=100)

    plt.figure('test merge')
    for i in range(testlen//2):
        plt.subplot(testlen//2, 6, i * 6 + 1)
        plt.subplot(testlen//2, 6, i * 6 + 2)
        plt.subplot(testlen//2, 6, i * 6 + 3)
        plt.subplot(testlen//2, 6, i * 6 + 4)
        plt.subplot(testlen//2, 6, i * 6 + 5)
        plt.subplot(testlen//2, 6, i * 6 + 6)
    if not os.path.isdir(opt.plots_dir):
        os.mkdir(opt.plots_dir)
    if not os.path.isdir(os.path.join(opt.plots_dir, opt.name)):
        os.mkdir(os.path.join(opt.plots_dir, opt.name))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    save_count = 0
    print('End of initialization. Time Taken: %d sec' % (time.time() - t0))
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, batch in enumerate(torch.randperm(dataset_size).split(opt.batch_size)): # enumerate(dataset):  # inner loop within one epoch
            if batch.shape[0] < opt.batch_size:
                continue  # ignore tail of data if tail is smaller than batch_size
            iter_start_time = time.time()  # timer for computation per iteration
            #if total_iters % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            total_iters += 1
            epoch_iter += 1

            # Run train iteration
            batch_a = torch.LongTensor(opt.batch_size//2)
            batch_n = torch.LongTensor(opt.batch_size//2)
            for k in range(opt.batch_size//2):
                label_a = caltech_labels[batch[2*k+1]]
                indices = (caltech_labels == label_a).nonzero()
                batch_a[k] = indices[torch.randint(indices.numel(), (1,))]
                indices = (caltech_labels != label_a).nonzero()
                batch_n[k] = indices[torch.randint(indices.numel(), (1,))]

            if opt.data_mode=='range':
                bboxes = [caltech_bboxes[b] for b in batch]
                images = crop_data(caltech_data[batch], bboxes, opt.input_size)
                bboxes = [caltech_bboxes[b] for b in batch_a]
                images_a = crop_data(caltech_data[batch_a], bboxes, opt.input_size)
                bboxes = [caltech_bboxes[b] for b in batch_n]
                images_n = crop_data(caltech_data[batch_n], bboxes, opt.input_size)
            if opt.data_mode=='cropped':
                images = caltech_data[batch]
                images_a = caltech_data[batch_a]
                images_n = caltech_data[batch_n]

            model_input = {'real_G': images[:, :3, :, :].reshape([-1, 2, 3, opt.input_size, opt.input_size]),  # image pairs for generator
                           'mask_G': images[:, 3, :, :].unsqueeze(1).unsqueeze(1).reshape([-1, 2, 1, opt.input_size, opt.input_size]),  # mask for real_G
                           'real_D': images[:, :3, :, :],  # real images for discriminator
                           'mask_D': images[:, 3, :, :].unsqueeze(1),  # mask for real_D
                           'real_a': images_a[:, :3, :, :],  # anchor images for ReID
                           'mask_a': images_a[:, 3, :, :].unsqueeze(1),  # mask for ReID
                           'real_n': images_n[:, :3, :, :],  # negative images for ReID
                           'mask_n': images_n[:, 3, :, :].unsqueeze(1),  # mask for ReID
                           }

            model.set_input(model_input, 'mix' if i % 1 == 0 else 'reflection', load=True)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(reid=i % opt.reid_freq == 0)  # calculate loss functions, get gradients, update network weights

            # plt.figure('intermediate')
            # #model.set_input(model_input, 'mix', load=True)
            # #model.forward()
            # test_results, iden_results = model.fake_G, model.rec_G_1
            # test_results = test_results.detach().cpu()
            # iden_results = iden_results.detach().cpu()
            # for i in range(testlen // 2):
            #     plt.subplot(testlen // 2, 6, 6 * i + 1)
            #     plt.imshow(tensor2im(model.real_G[i, 0].detach().cpu().permute([1, 2, 0])))
            #     plt.subplot(testlen // 2, 6, 6 * i + 2)
            #     plt.imshow(tensor2im(model.real_G[i, 1].detach().cpu().permute([1, 2, 0])))
            #     plt.subplot(testlen // 2, 6, 6 * i + 3)
            #     plt.imshow(tensor2im(test_results[i, 0].permute([1, 2, 0])))
            #     plt.subplot(testlen // 2, 6, 6 * i + 4)
            #     plt.imshow(tensor2im(test_results[i, 1].permute([1, 2, 0])))
            #     plt.subplot(testlen // 2, 6, 6 * i + 5)
            #     plt.imshow(tensor2im(iden_results[i, 0].permute([1, 2, 0])))
            #     plt.subplot(testlen // 2, 6, 6 * i + 6)
            #     plt.imshow(tensor2im(iden_results[i, 1].permute([1, 2, 0])))
            # display images on visdom and save images to a HTML file
            #if total_iters % opt.display_freq == 0:
            #    save_result = total_iters % opt.update_html_freq == 0
            #    model.compute_visuals()
            #    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # print training losses and save logging information to the disk
            #if total_iters % opt.print_freq == 0:
            #    losses = model.get_current_losses()
            #    t_comp = (time.time() - iter_start_time) / opt.batch_size
            #    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            #    if opt.display_id > 0:
            #        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            iter_data_time = time.time()

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            save_count = (save_count + 1) % 5
            save_suffix = 'save_{}' .format(save_count)
            model.save_networks(save_suffix)

            # Plot intermediate results
            plt.figure('test merge')
            model.set_input(model_test_input, 'mix', load=True)
            test_results, iden_results = model.runG()
            test_results = test_results.detach().cpu()
            iden_results = iden_results.detach().cpu()
            for i in range(testlen // 2):
                plt.subplot(testlen // 2, 6, 6 * i + 1)
                plt.imshow(tensor2im(model.real_G[i, 0].detach().cpu().permute([1, 2, 0])))
                plt.subplot(testlen // 2, 6, 6 * i + 2)
                plt.imshow(tensor2im(model.real_G[i, 1].detach().cpu().permute([1, 2, 0])))
                plt.subplot(testlen // 2, 6, 6 * i + 3)
                plt.imshow(tensor2im(test_results[i, 0].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 6, 6 * i + 4)
                plt.imshow(tensor2im(test_results[i, 1].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 6, 6 * i + 5)
                plt.imshow(tensor2im(iden_results[2 * i].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 6, 6 * i + 6)
                plt.imshow(tensor2im(iden_results[2 * i + 1].permute([1, 2, 0])))
            #plt.get_current_fig_manager().resize(1920, 1080)
            plt.suptitle('Merge epoch %d' % epoch, fontsize=12)
            #plt.pause(0.01)
            plt.savefig(os.path.join(opt.plots_dir, opt.name, 'epoch_%d.png' % epoch))

            # plt.figure('test reflect')
            # model.set_input(model_test_input, 'mix', load=True)
            # test_results, iden_results = model.runG()
            # test_results = test_results.detach().cpu()
            # iden_results = iden_results.detach().cpu()
            # for i in range(testlen):
            #     plt.subplot(testlen, 6, 6 * i + 1)
            #     plt.imshow(tensor2im(model.real_G[i, 0].detach().cpu().permute([1, 2, 0])))
            #     plt.subplot(testlen, 6, 6 * i + 2)
            #     plt.imshow(tensor2im(model.real_G[i, 1].detach().cpu().permute([1, 2, 0])))
            #     plt.subplot(testlen, 6, 6 * i + 3)
            #     plt.imshow(tensor2im(test_results[i, 0].permute([1, 2, 0])))
            #     plt.subplot(testlen, 6, 6 * i + 4)
            #     plt.imshow(tensor2im(test_results[i, 1].permute([1, 2, 0])))
            #     plt.subplot(testlen, 6, 6 * i + 5)
            #     plt.imshow(tensor2im(iden_results[2 * i].permute([1, 2, 0])))
            #     plt.subplot(testlen, 6, 6 * i + 6)
            #     plt.imshow(tensor2im(iden_results[2 * i + 1].permute([1, 2, 0])))
            # #plt.get_current_fig_manager().resize(1920, 1080)
            # plt.suptitle('Reflection epoch %d' % epoch, fontsize=12)
            # #plt.pause(0.01)
            # plt.savefig(os.path.join(opt.plots_dir, opt.name, 'reflection_epoch_%d.png' % epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        #model.update_learning_rate()                     # update learning rates at the end of every epoch.

    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
    save_suffix = 'save_{}_last'.format(save_count)
    model.save_networks(save_suffix)
    print('DONE')

