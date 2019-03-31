import os
import pickle

from scipy.io import loadmat
import numpy as np
import skimage.io
import skimage.transform
import scipy.misc
import matplotlib.pyplot as plt
import torch.utils.data

from data.data_utils import ArrayDataset
from util.caltech_images import fit_square

def load_caltech_data(path, imsize=128, type='pickle', mode='cropped'):
    assert type == 'pickle' or mode == 'original', 'caltech ucsd dataset in type mat cannot be cropped'
    caltech_images_path = os.path.join(path, 'images')
    caltech_annotations_path = os.path.join(path, 'annotations-' + type)

    N = 0
    for species in os.listdir(caltech_annotations_path):
        N += len(os.listdir(os.path.join(caltech_annotations_path, species)))

    if mode == 'cropped':
        images = np.zeros((N, 3, imsize, imsize))
        masks = np.zeros((N, 1, imsize, imsize))
        labels = np.zeros((N,))

    n = 0
    for species in os.listdir(caltech_annotations_path):
        species_images_path = os.path.join(caltech_images_path, species)
        species_annotations_path = os.path.join(caltech_annotations_path, species)
        for image in os.listdir(species_annotations_path):
            image_name = image.split('.')[0]
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
            if mode=='cropped':
                if type=='mat':
                    left, top, height, width = bbox['left'], bbox['top'], bbox['height'], bbox['width']
                else:
                    left, top, height, width = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                delta = max(height, width)
                rect = fit_square(left - (delta - width)//2,
                                     top - (delta - height)//2,
                                     max(height, width), H, W, allow_shrink=False)
                left, top, height, width = int(rect.get_x()), int(rect.get_y()), int(rect.get_height()), int(rect.get_width())
                image_delta = np.zeros([delta, delta, 3])
                mask_delta = np.zeros([delta, delta])
                image_delta[max(-top,0):min(delta, H-top), max(-left,0):min(delta, W-left), :] = \
                    image[max(0,top):min(top+delta, H), max(0,left):min(left+delta, W), :]
                mask_delta[max(-top,0):min(delta, H-top), max(-left,0):min(delta, W-left)] = \
                    seg[max(0,top):min(top+delta, H), max(0,left):min(left+delta, W)]

                images[n] = scipy.misc.imresize(image_delta, [imsize, imsize, 3]).transpose(2, 0, 1) / 255
                masks[n, 0] = skimage.transform.resize(mask_delta, [imsize, imsize])

                plt.subplot(2, 2, 1)
                plt.imshow(image_delta)
                plt.subplot(2, 2, 2)
                plt.imshow(mask_delta)
                plt.subplot(2, 2, 3)
                plt.imshow(images[n].transpose(1,2,0))
                plt.subplot(2, 2, 4)
                plt.imshow(masks[n, 0, :, :])

            labels[n] = int(species.split('.')[0])
            n += 1
    return images, masks, labels


def create_data_caltech_ucsd(path, batch_size, size=128, type='pickle', mode='cropped'):
    images, masks, labels = load_caltech_data(path, size, type, mode)

    caltech_data = np.concatenate([images, masks], axis=1).astype(np.float32)
    dataset = ArrayDataset(caltech_data, labels=labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, caltech_data, labels


if __name__ == '__main__':
    images, mask, labels = load_caltech_data('C:/Datasets/Caltech-UCSD-Birds-200')


