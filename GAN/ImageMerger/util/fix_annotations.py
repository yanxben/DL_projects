import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import line_aa, polygon
from scipy.signal import convolve2d
from scipy.io import loadmat
from skimage.io import imsave, imread

coords = []
idx = None
n_idx = 0
i_idx = 0


def fit_rectangle(left, top, delta, H, W, color):
    left = max(0, left)
    top = max(0, top)
    if delta > H or delta > W:
        delta = min(H,W)

    if left + delta >= W:
        left = W - 1 - delta
    if top + delta >= H:
        top = H - 1 - delta
    return patches.Rectangle((left, top), delta, delta, edgecolor=color, fill=False)


def image_selector(image, annotation, species, image_path, annotation_path):
    global i, idx, n_idx, i_idx
    H, W, C = image.shape

    # def onclick(event):
    #     global coords
    #     ix, iy = event.xdata, event.ydata
    #     print('image {} -- x = {}, y = {}'.format(idx[0], ix, iy), end='\r')
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
    #             create_mask(image.shape[1:3], coords, os.path.join(path, type, 'image_{}.png'.format(idx[0])))
    #
    #             coords = []
    #
    #             if len(idx) == 0:
    #                 fig.canvas.mpl_disconnect(cid)
    #             else:
    #                 idx.pop(0)
    #                 i_idx += 1
    #                 plt.cla()
    #                 plt.imshow(images[idx[0]])
    #                 plt.title('image {} {}/{}'.format(idx[0], i_idx, n_idx))
    #
    #     plt.show(block=False)

    seg = annotation['seg'].copy()
    bbox = annotation['bbox']

    fig = plt.figure(1, figsize=(1440 / 100, 900 / 100), dpi=100)
    plt.clf()
    ax = plt.gca()

    # Show image
    seg = convolve2d(seg, np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]]), mode='same') > 0
    seg_mat = np.repeat(np.expand_dims(seg, 2), 3, axis=2)
    image_temp = image.copy()
    image_temp = np.where(seg_mat, 255, image_temp)
    plt.imshow(image_temp)
    plt.title('image {}, species {}'.format(i, species))

    # Draw bounding box
    rect = patches.Rectangle((bbox['left'], bbox['top']), bbox['right'] - bbox['left'], bbox['bottom'] - bbox['top'],
                             edgecolor='y', fill=False)
    ax.add_patch(rect)

    # Draw square bounding box
    delta = max(bbox['right'] - bbox['left'],bbox['bottom'] - bbox['top'])
    rect1 = fit_rectangle(bbox['left'] - (delta - bbox['right'] + bbox['left'])//2,
                          bbox['top'] - (delta - bbox['bottom'] + bbox['top'])//2,
                          delta, H, W, color='b')
    ax.add_patch(rect1)

    # Draw increased bounding box
    rect2 = fit_rectangle(bbox['left'] - delta // 5,
                          bbox['top'] - delta // 5,
                          delta + 2 * delta // 5, H, W, 'r')
    ax.add_patch(rect2)

    plt.show(block=False)
    plt.pause(0.001)
    # Collect input
    try:
        choice = int(input('select image {}?:'.format(i)))
    except:
        return 1

    if choice == -1:  # Discard species
        return -1
    elif choice == 0:  # Discard image
        return None
    elif choice == 1:  # Save image as is
        left, top, height, width = 0, 0, H, W
        annotation['seg'] = annotation['seg']
        annotation['bbox'] = [left, top, height, width]
        #return image, annotation
    elif choice == 2:  # Save image with custom segmentation
        return None
    elif choice == 3:  # Save image with original bounding box
        left = int(rect1.get_x())
        top = int(rect1.get_y())
        height = int(rect1.get_height())
        width = int(rect1.get_width())
        image = image[top:top+height, left:left+width]
        annotation['seg'] = annotation['seg'][top:top+height, left:left+width]
        annotation['bbox'] = [left, top, height, width]
        #return image, annotation
    elif choice == 4:  # Save image with increased bounding box
        left = int(rect2.get_x())
        top = int(rect2.get_y())
        height = int(rect2.get_height())
        width = int(rect2.get_width())
        image = image[top:top+height, left:left+width]
        annotation['seg'] = annotation['seg'][top:top+height, left:left+width]
        annotation['bbox'] = [left, top, height, width]
        #return image, annotation
    elif choice == 5:  # Save image custom bbox
        return None
    elif choice == 6:  # Save image custom segmentation and bbox
        return None
    else:
        return 1

    # Save image and annotation
    #imsave(image_path, image)
    with open(annotation_path, 'wb') as f:
        pickle.dump(annotation, f)
    return None

    # with open(idx_path, 'r') as f:
    #     idx = json.load(f)
    #     n_idx = len(idx)
    #
    # while i > idx[0]:
    #     idx.pop(0)
    #     i_idx += 1
    #
    # plt.imshow(image)
    # plt.title('image {} {}/{}'.format(idx[0], i_idx, n_idx))
    # plt.show(block=False)
    #
    # cid = fig.canvas.mpl_connect('button_press_event', onclick)


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


def dataset_iterator(caltech_path, caltech_new_dir):
    global i, i0
    #caltech_images_path = os.path.join(caltech_path, 'images')
    #caltech_new_images_path = os.path.join(caltech_new_path, 'images')
    caltech_annotation_path = os.path.join(caltech_path, 'annotations-mat')
    caltech_new_annotation_path = os.path.join(caltech_path, caltech_new_dir)

    #if not os.path.isdir(caltech_new_path):
    #    os.mkdir(caltech_new_path)
    #if not os.path.isdir(caltech_new_images_path):
    #    os.mkdir(caltech_new_images_path)
    if not os.path.isdir(caltech_new_annotation_path):
        os.mkdir(caltech_new_annotation_path)

    for species in os.listdir(caltech_new_annotation_path):
        print(species)
        #species_images_path = os.path.join(caltech_images_path, species)
        species_annotation_path = os.path.join(caltech_annotation_path, species)
        #species_new_images_path = os.path.join(caltech_new_images_path, species)
        species_new_annotation_path = os.path.join(caltech_new_annotation_path, species)
        if species[:2] == '._':  # Clean images directory:
            1+1#os.remove(species_images_path)
        else:
            #if not os.path.isdir(species_new_images_path):
            #    os.mkdir(species_new_images_path)
            if not os.path.isdir(species_new_annotation_path):
                os.mkdir(species_new_annotation_path)
            for image_file in os.listdir(species_new_annotation_path):
                #image_path = os.path.join(species_images_path, image_name)
                if image_file[:2] == '._':  # Clean images directory:
                    1+1#os.remove(image_path)
                else:
                    if i < i0:
                        i += 1
                    else:
                        if True:  # flap == -1 skips species
                            # Load image and annotation
                            #image = imread(image_path)
                            annotation_path = os.path.join(species_annotation_path, image_file.split('.')[0] + '.mat')
                            annotation = loadmat(annotation_path, squeeze_me=True)

                            # Save image and annotation
                            #new_image_path = os.path.join(species_new_images_path, image_name)
                            new_annotation_path = os.path.join(species_new_annotation_path, image_file.split('.')[0] + '.pickle')
                            #flag = image_selector(image, annotation, species, None, new_annotation_path)
                            import pickle
                            bbox = annotation['bbox']
                            seg = annotation['seg']
                            if seg.max() == 0:
                                print(image_file)
                            left, top = bbox['left'], bbox['top']
                            height, width = bbox['bottom'] - bbox['top'], bbox['right'] - bbox['left']
                            annotation['bbox'] = [left, top, height, width]

                            with open(new_annotation_path, 'wb') as f:
                                pickle.dump(annotation, f)

                        #if flag is not None and flag != -1:
                            #print('Bad input at image {} species {}'.format(i, species))
                            #return

                        i0 += 1
                        i += 1


if __name__ == '__main__':
    global i, i0
    i0 = 0
    i = 0

    caltech_path = 'C:/Datasets/Caltech-UCSD-Birds-200'
    caltech_new_dir = 'annotations-pickle'
    dataset_iterator(caltech_path, caltech_new_dir)
    print('done')
