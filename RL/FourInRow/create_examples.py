import os
import time
import shutil

import numpy as np

from utils_game import Game, _check_final_step
from utils_data import save_end_game_data
BOARD_W, BOARD_H = 7, 6


def create_images(outdir=None, DATA_SIZE=100000):

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
            state, _, _ = game.rand_state(max_stage=0, final=True)
            win_actions, lose_actions = _check_final_step(state)

        if any(win_actions) or any(lose_actions):
            data[n, :, :, :] = state
            win_labels[n, :] = win_actions
            lose_labels[n, :] = lose_actions
            if any(win_actions):
                labels[n, :] = win_actions
            else:
                labels[n, :] = lose_actions

    data = data.reshape([DATA_SIZE, -1])

    if outdir is not None:
        print("Saving to file...")
        save_end_game_data(data, labels, win_labels, lose_labels, outdir)

    return data, labels, win_labels, lose_labels


if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'data')
    data_path = os.path.join(data_dir, "supervised_images")
    if os.path.isdir(data_path):
        shutil.rmtree(data_path)

    time.sleep(2)
    os.mkdir(data_path)

    print(os.getcwd())
    print(data_path)
    create_images(data_path)
