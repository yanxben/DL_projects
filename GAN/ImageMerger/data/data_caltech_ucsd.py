import os
import pickle
import random

import PIL
from scipy.io import loadmat
import numpy as np
import skimage.io
import skimage.transform
import scipy.misc
import torch.utils.data
import torchvision

from data.data_utils import ArrayDataset
from util.caltech_images import fit_square


def crop_data(images, bboxes, imsize):
    old_imsize = images.shape[2]
    images_out = 0.5 * torch.ones(size=(images.shape[0], images.shape[1], imsize, imsize))
    for k in range(images.shape[0]):
        left, top, delta = bboxes[k]['left'], bboxes[k]['top'], bboxes[k]['delta']
        delta2 = random.randint(delta, old_imsize)
        top2 = random.randint(max(0, top + delta - delta2), min(top, old_imsize - delta2))
        left2 = random.randint(max(0, left + delta - delta2), min(left, old_imsize - delta2))

        image_pil = torchvision.transforms.functional.to_pil_image(images[k, :3].type(torch.float32))
        image_pil_cropped = torchvision.transforms.functional.resized_crop(image_pil, top2, left2, delta2, delta2,
                                                                           [imsize, imsize], interpolation=PIL.Image.BICUBIC)
        image_cropped = torchvision.transforms.functional.to_tensor(image_pil_cropped)
        images_out[k, :3] = image_cropped
        if images.shape[1] == 4:
            image_pil = torchvision.transforms.functional.to_pil_image(images[k, 3:].type(torch.float32))
            image_pil_cropped = torchvision.transforms.functional.resized_crop(image_pil, top2, left2, delta2, delta2,
                                                                               [imsize, imsize], interpolation=PIL.Image.BICUBIC)
            image_cropped = torchvision.transforms.functional.to_tensor(image_pil_cropped)
            images_out[k, 3] = torch.where(image_cropped > 0.5, torch.ones_like(image_cropped), torch.zeros_like(image_cropped))

    return images_out


def load_caltech_data(path, imsize=128, type='pickle', mode='cropped', min_count=5, size=None, testset=None, testonly=False):
    assert type == 'pickle' or mode == 'original', 'caltech ucsd dataset in type mat cannot be cropped'
    caltech_images_path = os.path.join(path, 'images')
    caltech_annotations_path = os.path.join(path, 'annotations-' + type)

    N = 0
    for species in os.listdir(caltech_annotations_path):
        count = len(os.listdir(os.path.join(caltech_annotations_path, species)))
        if count >= min_count:
            N += len(os.listdir(os.path.join(caltech_annotations_path, species)))

    if size is not None:
        N = min(N, size)
    if mode == 'cropped':
        modesize = imsize
        images = 0.5 * np.ones((N, 3, modesize, modesize))
        masks = np.zeros((N, 1, modesize, modesize))
        labels = np.zeros((N,))
        bboxes = []

    if mode == 'range':
        modesize = imsize + 20
        images = 0.5 * np.ones((N, 3, modesize, modesize))
        masks = np.zeros((N, 1, modesize, modesize))
        labels = np.zeros((N,))
        bboxes = []

    if testset is not None:
        test_images = np.zeros((len(testset), 3, modesize, modesize))
        test_masks = np.zeros((len(testset), 1, modesize, modesize))
        test_labels = np.zeros((len(testset), ))
        test_bboxes = []
    n = 0
    for species in os.listdir(caltech_annotations_path):
        count = len(os.listdir(os.path.join(caltech_annotations_path, species)))
        print('{} {}'.format(species, count))
        if count < min_count:
            continue
        species_images_path = os.path.join(caltech_images_path, species)
        species_annotations_path = os.path.join(caltech_annotations_path, species)
        for image in os.listdir(species_annotations_path):
            if n >= N:
                if n == N:
                    print('quitting with reduced dataset')
                    n += 1
                continue

            image_name = image.split('.')[0]
            if not testonly or image_name in testset:
                image_path = os.path.join(species_images_path, image_name + '.jpg')
                annotation_path = os.path.join(species_annotations_path, image_name + '.' + type)

                image = skimage.io.imread(image_path)
                H, W, _ = image.shape

                if type=='mat':
                    annotation = loadmat(annotation_path)
                else:
                    with open(annotation_path, 'rb') as f:
                        annotation = pickle.load(f)

                bbox = annotation['bbox']
                seg = annotation['seg']
                if type == 'mat':
                    left, top, height, width = bbox['left'], bbox['top'], bbox['height'], bbox['width']
                else:
                    left, top, height, width = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                if mode=='cropped':
                    delta = max(height, width)
                if mode=='range':
                    delta = max(height, width)
                    delta += int(20 * delta / imsize)

                rect = fit_square(left - (delta - width)//2,
                                     top - (delta - height)//2,
                                     delta, H, W, allow_shrink=False)
                left, top, height, width = int(rect.get_x()), int(rect.get_y()), int(rect.get_height()), int(rect.get_width())
                image_delta = 255. * np.ones([delta, delta, 3]) / 2
                mask_delta = np.zeros([delta, delta])
                image_delta[max(-top,0):min(delta, H-top), max(-left,0):min(delta, W-left), :] = \
                    image[max(0,top):min(top+delta, H), max(0,left):min(left+delta, W), :]
                mask_delta[max(-top,0):min(delta, H-top), max(-left,0):min(delta, W-left)] = \
                    seg[max(0,top):min(top+delta, H), max(0,left):min(left+delta, W)]

                #images[n] = scipy.misc.imresize(image_delta, [modesize, modesize, 3]).transpose(2, 0, 1) / 255
                images[n] = skimage.transform.resize(mask_delta, [modesize, modesize], anti_aliasing=True, preserve_range=True)
                masks[n] = skimage.transform.resize(mask_delta, [modesize, modesize], anti_aliasing=True, preserve_range=True)
                masks[n] = np.where(masks[n] > 0.5, np.ones_like(masks[n]), np.zeros_like(masks[n]))
                # plt.subplot(2, 2, 1)
                # plt.imshow(image_delta)
                # plt.subplot(2, 2, 2)
                # plt.imshow(mask_delta)
                # plt.subplot(2, 2, 3)
                # plt.imshow(images[n].transpose(1,2,0))
                # plt.subplot(2, 2, 4)
                # plt.imshow(masks[n, 0, :, :])

                labels[n] = int(species.split('.')[0])
                masks_left = masks[n,0].copy()
                masks_left[:,-1] = .01
                left = masks_left.argmax(axis=1).min()
                masks_top = masks[n,0].copy()
                masks_top[-1,:] = .01
                top = masks_top.argmax(axis=0).min()
                masks_right = masks[n,0].copy()
                masks_right[:,0] = .01
                right = masks.shape[3] - masks_right[:, ::-1].argmax(axis=1).min() - 1
                masks_bottom = masks[n,0].copy()
                masks_bottom[0,:] = .01
                bottom = masks.shape[2] - masks_bottom[::-1, :].argmax(axis=0).min() - 1
                delta = max(right - left + 1, bottom - top + 1)
                rect = fit_square(left, top, delta, modesize, modesize, allow_shrink=True)
                bboxes.append({'left': int(rect.get_x()), 'top': int(rect.get_y()),
                               'delta': int(rect.get_width())})

            if testset is not None and image_name in testset:
                idx = testset.index(image_name)
                test_images[idx] = images[n]
                test_masks[idx] = masks[n]
                test_labels[idx] = labels[n]
                test_bboxes.append(bboxes[-1])
                print('Added test image {}: {}'.format(idx, image_name))
            n += 1

    print('done')
    testset = None if testset is None else \
        {'images': test_images, 'masks': test_masks, 'labels': test_labels, 'bboxes': test_bboxes}
    return images, masks, labels, bboxes, testset


def create_dataset_caltech_ucsd(path, batch_size, imsize=128, type='pickle', mode='cropped', size=None, testset=None):
    images, masks, labels, bboxes, testset = load_caltech_data(path, imsize, type, mode, size=size, testset=testset)

    caltech_data = np.concatenate([images, masks], axis=1).astype(np.float32)
    dataset = ArrayDataset(caltech_data, labels=labels.astype(np.int32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if testset is not None:
        testset['images'] = torch.from_numpy(np.concatenate([testset['images'], testset['masks']], axis=1).astype(np.float32))
        testset['labels'] = torch.from_numpy(testset['labels'].astype(np.int32))
        testset.pop('masks')
    return dataloader, torch.from_numpy(caltech_data), {'labels': torch.from_numpy(labels.astype(np.int64)), 'bboxes': bboxes}, testset


if __name__ == '__main__':
    test = ['Red_winged_Blackbird_0017_583846699', 'Yellow_headed_Blackbird_0009_483173184',
     'Lazuli_Bunting_0010_522399154', 'Painted_Bunting_0006_2862481106',
     'Gray_Catbird_0031_148467783', 'Purple_Finch_0006_2329434675', 'American_Goldfinch_0004_155617438',
     'Blue_Grosbeak_0008_2450854752', 'Green_Kingfisher_0002_228927324', 'Pied_Kingfisher_0002_1020026028']
    images, masks, labels, bboxes, testset = load_caltech_data('C:/Datasets/Caltech-UCSD-Birds-200', mode='range', testset=test, testonly=True)
    images_cropped = crop_data(torch.from_numpy(testset['images'][:,:3]), testset['bboxes'], 80).numpy()
    images_with_mask_cropped = crop_data(torch.from_numpy(np.concatenate((testset['images'], testset['masks']), axis=1)), testset['bboxes'], 80).numpy()
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(testset['images'].shape[0]):
        plt.subplot(3,10,i+1)
        plt.imshow(testset['images'][i].transpose(1,2,0))
        plt.title(testset['labels'][i])
        plt.subplot(3, 10, i + 10 + 1)
        plt.imshow(images_cropped[i].transpose(1, 2, 0))
        plt.subplot(3, 10, i + 20 + 1)
        plt.imshow(images_with_mask_cropped[i].transpose(1, 2, 0))
    print('done')



