import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa, polygon
from skimage.io import imsave

coords = []
done = False
i = 0

def mask_image(images, path):
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print('x = {}, y = {}'.format(ix, iy))

        global coords, done
        coords.append([ix, iy])

        plt.scatter(ix, iy, marker='.')

        if len(coords) > 1:
            plt.plot((ix, coords[-2][0]), (iy, coords[-2][1]), 'b')
            if l1_dist([ix, iy], coords[0]) < 1:
                coords.pop()
                done = True

        plt.pause(0.001)

    fig = plt.figure()

    for i in range(images.shape[0]):
        # Show image
        plt.imshow(images[i])
        plt.pause(0.001)
        # Collect coordinates
        global coords, done
        coords = []
        done = False

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        while not done:
            1 + 1

        fig.canvas.mpl_disconnect(cid)
        plt.cla()

        if len(coords) >= 3:
            create_mask(images.shape[1:], coords, os.path.join(path, 'image_{}.png'.format(i)))


def mask_image2(images, path):
    def onclick(event):
        global coords, i
        ix, iy = event.xdata, event.ydata
        print('image {} -- x = {}, y = {}'.format(i, ix, iy), end='\r')

        coords.append([ix, iy])

        fig = plt.gcf()
        plt.scatter(ix, iy, marker='.')

        if len(coords) > 1:
            plt.plot((ix, coords[-2][0]), (iy, coords[-2][1]), 'b')
            if l1_dist([ix, iy], coords[0]) < 3:
                coords.pop()

                create_mask(images.shape[1:3], coords, os.path.join(path, 'image_{}.png'.format(i)))

                i += 1
                coords = []

                if i >= images.shape[0]:
                    fig.canvas.mpl_disconnect(cid)
                else:
                    plt.cla()
                    plt.imshow(images[i])
                    plt.title('image {}'.format(i))
                    #plt.pause(0.001)

        plt.show(block=False)
        #plt.pause(0.001)

    fig = plt.figure()

    # Show image
    global i
    plt.imshow(images[i])
    plt.show(block=False)
    #plt.pause(0.001)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)


def create_mask(shape, coords, path):
    mask = np.zeros(shape, dtype=np.uint8)

    for i in range(len(coords)):
        #c1 = coords[i]
        #c2 = coords[(i + 1) % len(coords)]
        #rr, cc, _ = line_aa(int(c1[1]), int(c1[0]), int(c2[1]), int(c2[0]))
        rr, cc = polygon([round(c[1]) for c in coords], [round(r[0]) for r in coords], shape)
        mask[rr, cc] = 255

    plt.imshow(mask)
    plt.show(block=False)
    #plt.pause(0.1)
    imsave(path, mask)


def l1_dist(c1, c2):
    return sum([abs(c1[j] - c2[j]) for j in range(len(c1))])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--stlpath', dest='stlpath', required=False, default='C:/Datasets',
                        help='path to STL10 dataset')
    parser.add_argument('--maskpath', dest='maskpath', required=False, default='C:/Datasets/masks/stl10_birds',
                        help='path to folder where to save masks')
    parser.add_argument('--i0', type=int, dest='i0', required=False, default=0,
                        help='index of starting image')
    parser.set_defaults(feature=True)
    args = parser.parse_args()

    import torchvision

    i = args.i0

    stl10_train = torchvision.datasets.STL10(args.stlpath, split='train', download=False)
    stl10_bird = stl10_train.data[[label == 1 for label in stl10_train.labels]]
    mask_image2(stl10_bird.transpose((0, 2, 3, 1)), path=args.maskpath)
    plt.show()