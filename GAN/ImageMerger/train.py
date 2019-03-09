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
from options.train_options import TrainOptions
from data.data_stl10 import create_dataset_stl10_bird
from models import create_model
import matplotlib.pyplot as plt
import torch
#from util.visualizer import Visualizer

testset = [0, 4, 5, 6, 206, 210, 213, 405, 407, 435]
if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset, stl10_data = create_dataset_stl10_bird(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    stl10_data_test = stl10_data[testset]
    model_test_input = {'real_G': torch.Tensor(stl10_data_test[:, :3, :, :]),
                        'mask_G': torch.Tensor(stl10_data_test[:, 3, :, :])}
    testset_figure = plt.figure()
    for i in range(len(testset)//2):
        plt.subplot(5, 6, i * 6 + 1)
        plt.imshow(model_test_input['real_G'][2*i].permute([1, 2, 0]))
        plt.subplot(5, 6, i * 6 + 2)
        plt.imshow(model_test_input['real_G'][2*i + 1].permute([1, 2, 0]))
        plt.subplot(5, 6, i * 6 + 3)
        plt.subplot(5, 6, i * 6 + 4)
        plt.subplot(5, 6, i * 6 + 5)
        plt.subplot(5, 6, i * 6 + 6)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            if data['image'].shape[0] < opt.batch_size:
                continue  # ignore tail of data if tail is smaller than batch_size
            iter_start_time = time.time()  # timer for computation per iteration
            #if total_iters % opt.print_freq == 0:
            #    t_data = iter_start_time - iter_data_time
            #visualizer.reset()
            total_iters += 1
            epoch_iter += 1

            # Run train iteration
            model_input = {'real_G': data['image'][:, :3, :, :],
                           'real_D': data['image'][:, :3, :, :],
                           'mask_G': data['image'][:, 3, :, :],
                           'mask_D': data['image'][:, 3, :, :]}
            model.set_input(model_input)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

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
            save_suffix = 'epoch_%d' % epoch
            model.save_networks(save_suffix)
            # Plot intermediate results
            test_results, iden_results = model.runG(model_test_input)
            test_results = test_results.detach()
            iden_results = iden_results.detach()
            for i in range(len(testset) // 2):
                plt.subplot(5, 6, 6 * i + 3)
                plt.imshow(test_results[i, 0].permute([1, 2, 0]))
                plt.subplot(5, 6, 6 * i + 4)
                plt.imshow(test_results[i, 1].permute([1, 2, 0]))
                plt.subplot(5, 6, 6 * i + 5)
                plt.imshow(iden_results[2 * i].permute([1, 2, 0]))
                plt.subplot(5, 6, 6 * i + 6)
                plt.imshow(iden_results[2 * i + 1].permute([1, 2, 0]))
            plt.savefig(os.path.join(opt.plots_dir, opt.name, 'epoch_%d.png' % epoch))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    print('DONE')
