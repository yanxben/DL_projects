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


def load_caltech_data(path, imsize=128, type='pickle', mode='cropped', min_count=5, size=None, testset=None):
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
        images = np.zeros((N, 3, imsize, imsize))
        masks = np.zeros((N, 1, imsize, imsize))
        labels = np.zeros((N,))

    if testset is not None:
        test_images = np.zeros((len(testset), 3, imsize, imsize))
        test_masks = np.zeros((len(testset), 1, imsize, imsize))
        test_labels = np.zeros((len(testset), ))
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
                #images[n] = skimage.transform.resize(mask_delta, [imsize, imsize], anti_aliasing=True, preserve_range=True)
                masks[n, 0] = skimage.transform.resize(mask_delta, [imsize, imsize], anti_aliasing=True, preserve_range=True)

                # plt.subplot(2, 2, 1)
                # plt.imshow(image_delta)
                # plt.subplot(2, 2, 2)
                # plt.imshow(mask_delta)
                # plt.subplot(2, 2, 3)
                # plt.imshow(images[n].transpose(1,2,0))
                # plt.subplot(2, 2, 4)
                # plt.imshow(masks[n, 0, :, :])

            labels[n] = int(species.split('.')[0])

            if testset is not None:
                if image_name in testset:
                    idx = testset.index(image_name)
                    test_images[idx] = images[n]
                    test_masks[idx] = masks[n]
                    test_labels[idx] = labels[n]
                    print('Added test image {}: {}'.format(idx, image_name))
            n += 1

    print('done')
    testset = {'images': test_images, 'masks': test_masks, 'labels': test_labels} if testset is not None else None
    return images, masks, labels, testset


def create_dataset_caltech_ucsd(path, batch_size, imsize=128, type='pickle', mode='cropped', size=None, testset=None):
    images, masks, labels, testset = load_caltech_data(path, imsize, type, mode, size=size, testset=testset)

    caltech_data = np.concatenate([images, masks], axis=1).astype(np.float32)
    dataset = ArrayDataset(caltech_data, labels=labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if testset is not None:
        testset['images'] = torch.from_numpy(np.concatenate([testset['images'], testset['masks']], axis=1).astype(np.float32))
        testset['labels'] = torch.from_numpy(testset['labels'])
        testset.pop('masks')
    return dataloader, torch.from_numpy(caltech_data), torch.from_numpy(labels), testset


if __name__ == '__main__':
    test = ['Red_winged_Blackbird_0017_583846699', 'Yellow_headed_Blackbird_0009_483173184',
     'Lazuli_Bunting_0010_522399154', 'Painted_Bunting_0006_2862481106',
     'Gray_Catbird_0031_148467783', 'Purple_Finch_0006_2329434675', 'American_Goldfinch_0004_155617438',
     'Blue_Grosbeak_0008_2450854752', 'Green_Kingfisher_0002_228927324', 'Pied_Kingfisher_0002_1020026028']
    images, mask, labels, testset = load_caltech_data('C:/Datasets/Caltech-UCSD-Birds-200', testset=test)

    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(testset['images'].shape[0]):
        plt.subplot(2,6,i+1)
        plt.imshow(testset['images'][i].transpose(1,2,0))
        plt.title(testset['labels'][i])



