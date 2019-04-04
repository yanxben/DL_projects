import os
import pandas as pd

BOARD_W = 7
BOARD_H = 6

def load_end_game_data(size, path='./data/supervised_images/'):
    print('-data-')
    data_df = pd.read_csv(os.path.join(path, 'images_df.csv'))
    data_txt = data_df.values
    data = data_txt[:size, 1:].reshape([size, BOARD_H, BOARD_W, 3]).transpose([0, 3, 1, 2])
    print('-labels-')
    labels_df = pd.read_csv(os.path.join(path, 'labels_df.csv'))
    labels_txt = labels_df.values
    labels = labels_txt[:size, 1:]
    print('-win_labels-')
    win_labels_df = pd.read_csv(os.path.join(path, 'win_labels_df.csv'))
    win_labels_txt = win_labels_df.values
    win_labels = win_labels_txt[:size, 1:]
    print('-lose_labels-')
    lose_labels_df = pd.read_csv(os.path.join(path, 'lose_labels_df.csv'))
    lose_labels_txt = lose_labels_df.values
    lose_labels = lose_labels_txt[:size, 1:]

    return data, labels, win_labels, lose_labels


def save_end_game_data(data, labels, win_labels, lose_labels, path='./data/supervised_images/'):
    print('-data-')
    pd.DataFrame(data.reshape([data.shape[0], BOARD_H * BOARD_W * 3])).to_csv(os.path.join(path, 'images_df.csv'))
    print('-labels-')
    pd.DataFrame(labels).to_csv(os.path.join(path, 'labels_df.csv'))
    print('-win_labels-')
    pd.DataFrame(win_labels).to_csv(os.path.join(path, 'win_labels_df.csv'))
    print('-lose_labels-')
    pd.DataFrame(lose_labels).to_csv(os.path.join(path, 'lose_labels_df.csv'))
