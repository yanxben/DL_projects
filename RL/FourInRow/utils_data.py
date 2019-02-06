import pandas as pd

BOARD_W = 7
BOARD_H = 6

def load_end_game_data(size):
    print('-labels-')
    # labels_txt = np.loadtxt(os.path.join(os.getcwd(), 'data\\supervised_images\\labels.csv'), delimiter=',')
    labels_df = pd.read_csv('./data/supervised_images/labels_df.csv')
    labels_txt = labels_df.values
    labels = labels_txt[:size, 1:]
    print('-win_labels-')
    # win_labels_txt = np.loadtxt(os.path.join(os.getcwd(), 'data\\supervised_images\\win_labels.csv'), delimiter=',')
    win_labels_df = pd.read_csv('./data/supervised_images/win_labels_df.csv')
    win_labels_txt = win_labels_df.values
    win_labels = win_labels_txt[:size, 1:]
    print('-lose_labels-')
    # lose_labels_txt = np.loadtxt(os.path.join(os.getcwd(), 'data\\supervised_images\\lose_labels.csv'), delimiter=',')
    lose_labels_df = pd.read_csv('./data/supervised_images/lose_labels_df.csv')
    lose_labels_txt = lose_labels_df.values
    lose_labels = lose_labels_txt[:size, 1:]
    print('-data-')
    # data_txt = np.loadtxt(os.path.join(os.getcwd(), 'data\\supervised_images\\images.csv'), delimiter=',')
    data_df = pd.read_csv('./data/supervised_images/images_df.csv')
    data_txt = data_df.values
    data = data_txt[:size, 1:].reshape([size, BOARD_H, BOARD_W, 3]).transpose([0, 3, 1, 2])

    return data, labels, win_labels, lose_labels