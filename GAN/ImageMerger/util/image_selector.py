import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa, polygon
from skimage.io import imsave
import json

def image_selector(images, start):
    # def onclick(event):
    #     global coords, i
    #     ix, iy = event.xdata, event.ydata
    #     print('image {} -- x = {}, y = {}'.format(i, ix, iy), end='\r')
    #
    #     coords.append([ix, iy])
    #
    #     fig = plt.gcf()
    #     plt.scatter(ix, iy, marker='.')
    #
    #     if len(coords) > 1:
    #         plt.plot((ix, coords[-2][0]), (iy, coords[-2][1]), 'b')
    #         if l1_dist([ix, iy], coords[0]) < 3:
    #             coords.pop()
    #
    #             create_mask(images.shape[1:3], coords, os.path.join(path, 'image_{}.png'.format(i)))
    #
    #             i += 1
    #             coords = []
    #
    #             if i >= images.shape[0]:
    #                 fig.canvas.mpl_disconnect(cid)
    #             else:
    #                 plt.cla()
    #                 plt.imshow(images[i])
    #                 plt.title('image {}'.format(i))
    #                 #plt.pause(0.001)
    #
    #     plt.show(block=False)
    #     #plt.pause(0.001)

    fig = plt.figure()
    indices = []
    for i in range(start, images.shape[0]):
        # Show image
        plt.imshow(images[i])
        plt.title('image {}'.format(i))
        plt.show(block=False)
        plt.pause(0.001)

        # Collect input
        try:
            choice = int(input('select image {}?:'.format(i)))
        except:
            return indices

        if choice > 0:
            indices.append(i)
        if choice < 0:
            return indices

        plt.cla()

    return indices


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--stl_path', dest='stl_path', required=False, default='C:/Datasets',
                        help='path to STL10 dataset')
    parser.add_argument('--stl_type', dest='stl_type', required=False, default='train',
                        help='path to STL10 dataset')
    parser.add_argument('--save_path', dest='save_path', required=False, default='C:/Datasets/masks/stl10_birds',
                        help='path to folder where to save indices')
    parser.add_argument('--start', type=int, dest='start', required=False, default=0,
                        help='index of starting image')
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    import torchvision


    stl10_dataset = torchvision.datasets.STL10(args.stl_path, split=args.stl_type, download=False)
    print(stl10_dataset.data.shape[0])
    if args.stl_type == 'unlabeled':
        stl10_data = stl10_dataset.data
    else:
        stl10_data = stl10_dataset.data[[label == stl10_dataset.classes.index('bird') for label in stl10_dataset.labels]]

    print(stl10_data.shape[0])

    indices = image_selector(stl10_data.transpose((0, 2, 3, 1)), args.start)
    print(len(indices))

    file = os.path.join(args.save_path, args.stl_type + '.json')
    if os.path.isfile(file):
        with open(file, 'r') as f:
            indices += json.load(f)
            f.close()

    indices = list(set(indices))
    print(len(indices))

    with open(file, 'w') as f:
        json.dump(indices, f, ensure_ascii=False)

    print('done')

