"""
This code initiates the training routine.
To start the training, make sure the dataset is accessible and mention the right location in row 40.
"""

import os
import socket
import time
from options.train_options import TrainOptions
from data.data_caltech_ucsd import create_dataset_caltech_ucsd, crop_data
from models import create_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch



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

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if socket.gethostname() == 'YABENN-P50':
        caltech_path = 'C:/Datasets/Caltech-UCSD-Birds-200'
    elif os.sys.platform == 'linux':
        caltech_path = '/home/' + os.getlogin() + '/Datasets/Caltech-UCSD-Birds-200'
    _, caltech_data, caltech_meta, testset = create_dataset_caltech_ucsd(caltech_path, opt.batch_size, size=opt.data_size ,mode=opt.data_mode, imsize=opt.input_size, testset=testset)  # create a dataset given opt.dataset_mode and other options

    caltech_labels = caltech_meta['labels']
    caltech_bboxes = caltech_meta['bboxes']

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
                        'mask_n': testset['images'][:testlen//2, 3, :, :].unsqueeze(1),     # Unused
                        'real_a_labels': testset['labels'],  # anchor images for ReID
                        }

    plt.figure('test merge', figsize=(1920 / 100, 1080 / 100), dpi=100)
    plt.figure('test reflect', figsize=(1920 / 100, 1080 / 100), dpi=100)

    plt.figure('test merge')
    for i in range(testlen//2):
        plt.subplot(testlen//2, 8, i * 8 + 1)
        plt.subplot(testlen//2, 8, i * 8 + 2)
        plt.subplot(testlen//2, 8, i * 8 + 3)
        plt.subplot(testlen//2, 8, i * 8 + 4)
        plt.subplot(testlen//2, 8, i * 8 + 5)
        plt.subplot(testlen//2, 8, i * 8 + 6)
        plt.subplot(testlen//2, 8, i * 8 + 7)
        plt.subplot(testlen//2, 8, i * 8 + 8)
    if not os.path.isdir(opt.plots_dir):
        os.mkdir(opt.plots_dir)
    if not os.path.isdir(os.path.join(opt.plots_dir, opt.name)):
        os.mkdir(os.path.join(opt.plots_dir, opt.name))

    total_iters = 0                # the total number of training iterations
    save_count = 0
    print('End of initialization. Time Taken: %d sec' % (time.time() - t0))
    for epoch in range(opt.epoch_start, opt.epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, batch in enumerate(torch.randperm(dataset_size).split(opt.batch_size)): # enumerate(dataset):  # inner loop within one epoch
            if batch.shape[0] < opt.batch_size:
                continue  # ignore tail of data if tail is smaller than batch_size
            iter_start_time = time.time()  # timer for computation per iteration
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

            if opt.data_mode == 'range':
                bboxes = [caltech_bboxes[b] for b in batch]
                images = crop_data(caltech_data[batch], bboxes, opt.input_size)
                bboxes = [caltech_bboxes[b] for b in batch_a]
                images_a = crop_data(caltech_data[batch_a], bboxes, opt.input_size)
                bboxes = [caltech_bboxes[b] for b in batch_n]
                images_n = crop_data(caltech_data[batch_n], bboxes, opt.input_size)
            if opt.data_mode == 'cropped':
                images = caltech_data[batch]
                images_a = caltech_data[batch_a]
                images_n = caltech_data[batch_n]
            labels_a = caltech_labels[batch_a]

            model_input = {'real_G': images[:, :3, :, :].reshape([-1, 2, 3, opt.input_size, opt.input_size]),  # image pairs for generator
                           'mask_G': images[:, 3, :, :].unsqueeze(1).unsqueeze(1).reshape([-1, 2, 1, opt.input_size, opt.input_size]),  # mask for real_G
                           'real_D': images[:, :3, :, :],  # real images for discriminator
                           'mask_D': images[:, 3, :, :].unsqueeze(1),  # mask for real_D
                           'real_a': images_a[:, :3, :, :],  # anchor images for ReID
                           'mask_a': images_a[:, 3, :, :].unsqueeze(1),  # mask for ReID
                           'real_n': images_n[:, :3, :, :],  # negative images for ReID
                           'mask_n': images_n[:, 3, :, :].unsqueeze(1),  # mask for ReID
                           'real_a_labels': labels_a,  # anchor images for ReID
                           }

            model.set_input(model_input)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(reid=i % opt.reid_freq == 0)  # calculate loss functions, get gradients, update network weights

            iter_data_time = time.time()

        # cache our model every <save_epoch_freq> epochs
        if epoch % opt.save_epoch_freq == 0:
            save_count = (save_count + 1) % 10
            save_suffix = 'save_{}' .format(save_count)
            print('{} - {} - saving the model at the end of epoch {}, iters {}'.format(opt.name, save_suffix, epoch, total_iters))
            model.save_networks(save_suffix)

            # Plot intermediate results
            plt.figure('test merge')
            model.set_input(model_test_input)
            test_results, recon_results, iden_results = model.runG()
            test_results = test_results.detach().cpu()
            recon_results = recon_results.detach().cpu()
            iden_results = iden_results.detach().cpu()
            for i in range(testlen // 2):
                plt.subplot(testlen // 2, 8, 8 * i + 1)
                plt.imshow(tensor2im(model.real_G[i, 0].detach().cpu().permute([1, 2, 0])))
                plt.subplot(testlen // 2, 8, 8 * i + 2)
                plt.imshow(tensor2im(model.real_G[i, 1].detach().cpu().permute([1, 2, 0])))
                plt.subplot(testlen // 2, 8, 8 * i + 3)
                plt.imshow(tensor2im(test_results[i, 0].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 8, 8 * i + 4)
                plt.imshow(tensor2im(test_results[i, 1].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 8, 8 * i + 5)
                plt.imshow(tensor2im(recon_results[i, 0].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 8, 8 * i + 6)
                plt.imshow(tensor2im(recon_results[i, 1].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 8, 8 * i + 7)
                plt.imshow(tensor2im(iden_results[2 * i].permute([1, 2, 0])))
                plt.subplot(testlen // 2, 8, 8 * i + 8)
                plt.imshow(tensor2im(iden_results[2 * i + 1].permute([1, 2, 0])))
            plt.suptitle('Merge epoch %d' % epoch, fontsize=12)
            plt.savefig(os.path.join(opt.plots_dir, opt.name, 'epoch_%d.png' % epoch))

        print('End of epoch {} / {} \t Time Taken: {:d} sec'.format(epoch, opt.epochs, int(time.time() - epoch_start_time)))

    print('saving the model at the end of epoch {}, iters {}'.format(epoch, total_iters))
    save_suffix = 'save_{}_last'.format(save_count)
    model.save_networks(save_suffix)
    print('DONE')

