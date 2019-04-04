import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from model import DQN_CNN, DQN_CNN_WIDE, DQN_LINEAR, DQN_SKIP, DQN_SKIP_WIDE
from utils_data import load_end_game_data
from create_examples import create_images

CREATE_DATA = False
data_save_dir = None
DATA_SIZE = 100000
TRAIN_SIZE = int(0.9 * DATA_SIZE)
BOARD_W, BOARD_H = 7, 6
lr = 1e-4
batch_size = 50
epochs = 10


USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print('GPU')
    dtype = torch.cuda.FloatTensor
else:
    print('CPU')
    dtype = torch.FloatTensor


def main():

    criterion = nn.MSELoss()

    # Get dataset
    if CREATE_DATA:
        print("Creating data...")
        data, labels, win_labels, lose_labels = create_images(data_save_dir, DATA_SIZE=DATA_SIZE)
    else:
        print("Loading data...")
        # Temporary swap until recollect data
        data, labels, win_labels, lose_labels = load_end_game_data(DATA_SIZE)

    plt.figure()
    for i in range(5):
        for j in range(5):
            plt.subplot(5, 5, i*5 + j + 1)
            plt.imshow(data[i*5 + j].transpose([1,2,0]))

    # Split train-test
    train_data = data[:TRAIN_SIZE]
    val_data = data[TRAIN_SIZE:]

    train_labels = labels[:TRAIN_SIZE]
    val_labels = labels[TRAIN_SIZE:]

    # Train
    model_names = ['DQN_LINEAR', 'DQN_CNN', 'DQN_CNN_WIDE', 'DQN_SKIP', 'DQN_SKIP_WIDE']
    models = [DQN_LINEAR, DQN_CNN, DQN_CNN_WIDE, DQN_SKIP, DQN_SKIP_WIDE]
    train_sizes = list(range(TRAIN_SIZE//9, TRAIN_SIZE+1, TRAIN_SIZE//9))
    learning_curve = [[] for _ in model_names]
    for i, model_func in enumerate(models):
        for train_size in train_sizes:
            # Define model
            print("Creating model {} train size: {}".format(model_names[i], train_size))
            model = model_func().cuda()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            # Train model
            learning_curve[i].append(train(model, optimizer, criterion, train_data[:train_size], train_labels[:train_size], val_data, val_labels))
            # Delete model
            del model

    plot_learning_curve(learning_curve, train_sizes, model_names)


def train(model, optimizer, criterion, train_data, train_labels, val_data, val_labels):
    batch_epoch = 0
    learning_curve = []

    train_size = train_data.shape[0]
    for epoch in range(epochs * TRAIN_SIZE // train_size):
        running_error = 0
        batch_num = 0
        idx = np.array(list(range(train_size)))
        np.random.shuffle(idx)

        for batch_idx in np.split(idx, train_size//batch_size):
            # Get batch data
            states = Variable(torch.Tensor(train_data[batch_idx])).type(dtype)
            labels = Variable(torch.Tensor(train_labels[batch_idx])).type(dtype)

            # Evaluate
            actions = model(states[:, :2, :, :])

            error = criterion(actions, labels)
            batch_loss = error.item()
            batch_error = evaluate(actions, labels)
            running_error += error.item()
            batch_epoch += 1
            batch_num += 1

            # Do step at the end to keep that it will be done after validation
            optimizer.zero_grad()
            error.backward()  # Compute the gradient for each variable
            optimizer.step()  # Update the weights according to the computed gradient

            # Plot average train loss across 10 batches and validation loss
            if not batch_epoch % 100:

                # Calculate validation loss
                with torch.no_grad():
                    val_data_tensor = torch.Tensor(val_data).type(dtype)
                    val_actions = model(val_data_tensor[:,:2,:,:])
                    val_error = evaluate(val_actions, val_labels, True)

                    # Average train loss over 100 batches for stability
                    learning_curve.append((batch_epoch, running_error / 100, val_error))
                    running_error = 0

                # Plot every 100 batches
                print(
                    'Epoch: {:2d} Batch: {:3d} Loss: {:.5f} Error: {:.5f} Validation Error: {:.5f}'
                    .format(epoch, batch_epoch, batch_loss, batch_error, val_error)
                )

    return learning_curve


def evaluate(actions, labels):
    action_argmax = np.argmax(actions.cpu().detach().numpy(), axis=1)
    labels_argmax = np.array([ [labels[i, action_argmax[i]]] for i in range(labels.shape[0]) ])
    error_rate = np.mean(1 - labels_argmax)
    return error_rate


def plot_learning_curve(learning_curve, data_sizes, model_names):
    num_models = len(learning_curve)
    num_bars = len(learning_curve[0])

    fig, ax = plt.subplots()

    width = 1 / (num_models + 1)
    ind = np.arange(num_bars)
    off = 0.5 - width

    import matplotlib
    from matplotlib import cm
    cmap = matplotlib.cm.get_cmap('hsv')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=num_models)
    for i in range(num_models):
        color = cmap(norm(i))
        val_err = [learning_curve[i][j][-1][2] for j in range(num_bars)]
        ax.bar(ind - off + width * i, val_err, width, color=color, alpha=0.5, label=model_names[i])
        trn_err = [learning_curve[i][j][-1][1] for j in range(num_bars)]
        ax.scatter(ind - off + width * i, trn_err, marker='o', c=color, edgecolors='k', label=None)

    ax.set_title('train and validation error over train size')
    ax.set_ylabel('error (val - bar, train - dot)')
    ax.set_xticks(ind)
    ax.set_xticklabels(data_sizes)
    ax.legend()
    plt.grid()
    plt.show()
    plt.savefig('supervised.png')


if __name__ == "__main__":
    main()
