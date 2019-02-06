import os
import time
import shutil
#import random

#import imageio
#import torch
import numpy as np
import pandas as pd
#from skimage import img_as_float, img_as_ubyte

#from fetch_data import _get_negative_imagepaths, _image_to_array, fetch_negative_mining
#from image_utils import channels_last2first, channels_first2last
#from model import Classifier12FCN
#from train_utils import load_model
#from skimage.transform import rescale

from utils_game import Game, check_final_step

data_dir = os.path.join(os.getcwd(), 'data')

BOARD_W, BOARD_H = 7, 6


def create_images(outdir, DATA_SIZE=100000):

    print("Creating data...")
    game = Game()
    data = np.ndarray([DATA_SIZE, BOARD_H, BOARD_W, 3])
    win_labels = np.ndarray([DATA_SIZE, BOARD_W])
    lose_labels = np.ndarray([DATA_SIZE, BOARD_W])
    labels = np.ndarray([DATA_SIZE, BOARD_W])
    for n in range(DATA_SIZE):
        if n % 100 == 0:
            print('image {}' .format(n))
        win_actions, lose_actions = [0] * BOARD_W, [0] * BOARD_W
        while not (any(win_actions) or any(lose_actions)):
            state, _, _ = game.rand_state(final=True)
            win_actions, lose_actions = check_final_step(state)

        if any(win_actions) or any(lose_actions):
            data[n, :, :, :] = state
            win_labels[n, :] = win_actions
            lose_labels[n, :] = lose_actions
            if any(win_actions):
                labels[n, :] = win_actions
            else:
                labels[n, :] = lose_actions

    data = data.reshape([DATA_SIZE, -1])

    print("Saving to file...")
    data_df = pd.DataFrame(data)
    data_df.to_csv(os.path.join(outdir, 'images_df.csv'))
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(os.path.join(outdir, 'labels_df.csv'))
    win_labels_df = pd.DataFrame(win_labels)
    win_labels_df.to_csv(os.path.join(outdir, 'win_labels_df.csv'))
    lose_labels_df = pd.DataFrame(lose_labels)
    lose_labels_df.to_csv(os.path.join(outdir, 'lose_labels_df.csv'))
    #np.savetxt(os.path.join(outdir, 'images.csv'), data, delimiter=',')
    #np.savetxt(os.path.join(outdir, 'labels.csv'), labels, delimiter=',')
    #np.savetxt(os.path.join(outdir, 'win_labels.csv'), win_labels, delimiter=',')
    #np.savetxt(os.path.join(outdir, 'lose_labels.csv'), lose_labels, delimiter=',')


if __name__ == '__main__':
    data_path = os.path.join(data_dir, "supervised_images")
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)

    time.sleep(2)
    os.mkdir(data_path)

    print(os.getcwd())
    print(data_path)
    create_images(data_path)
