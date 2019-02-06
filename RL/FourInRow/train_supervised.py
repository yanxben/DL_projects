import os, sys, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from model import DQN_FCN, DQN_FCN_WIDE, DQN_LINEAR
from utils_game import Game, _check_final_step
from utils_plot import plot_state
from utils_data import load_end_game_data


DATA_SIZE = 100000
TRAIN_SIZE = int(0.7 * DATA_SIZE)
BOARD_W, BOARD_H = 7, 6
lr = 1e-4
batch_size = 50
epochs = 1

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print('GPU')
    dtype = torch.cuda.FloatTensor
else:
    print('CPU')
    dtype = torch.FloatTensor

#dtype = torch.FloatTensor


def main():

    # Define model
    print("Creating model")
    model = DQN_FCN().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Get dataset
    if False:
        print("Creating data...")
        game = Game()
        data = np.ndarray([DATA_SIZE, 3, BOARD_H, BOARD_W])
        win_labels = np.ndarray([DATA_SIZE, BOARD_W])
        lose_labels = np.ndarray([DATA_SIZE, BOARD_W])
        labels = np.ndarray([DATA_SIZE, BOARD_W])
        for n in range(DATA_SIZE):

            win_actions, lose_actions = [0] * BOARD_W, [0] * BOARD_W
            while not (any(win_actions) or any(lose_actions)):
                state, _, _ = game.rand_state()
                win_actions, lose_actions = _check_final_step(state)

            if any(win_actions) or any(lose_actions):
                data[n, :, :, :] = state.transpose([2, 0, 1])
                win_labels[n, :] = win_actions
                lose_labels[n, :] = lose_actions
                if any(win_actions):
                    labels[n, :] = win_actions
                else:
                    labels[n, :] = lose_actions
    else:
        print("Loading data...")
        # Temporary swap untila recollect data
        #data, labels, win_labels, lose_labels = load_end_game_data(DATA_SIZE)
        data, labels, lose_labels, win_labels = load_end_game_data(DATA_SIZE)


    plt.figure()
    for i in range(5):
        for j in range(5):
            plt.subplot(5, 5, i*5 + j + 1)
            plt.imshow(data[i*5 + j, :, :, :].transpose([1,2,0]))

    # Split train-test
    train_data = data[:TRAIN_SIZE, :, :, :]
    val_data = data[TRAIN_SIZE:, :, :, :]
    train_win_labels = win_labels[:TRAIN_SIZE, :]
    train_lose_labels = lose_labels[:TRAIN_SIZE, :]
    val_win_labels = win_labels[TRAIN_SIZE:, :]
    val_lose_labels = lose_labels[TRAIN_SIZE:, :]

    train_labels = labels[:TRAIN_SIZE, :]
    val_labels = labels[TRAIN_SIZE:, :]

    # Train
    learning_curve = train(model, optimizer, criterion, train_data, train_labels, val_data, val_labels)
    #plot_learning_curve(learning_curve)

    plt.figure()
    with torch.no_grad():
        images = torch.Tensor(val_data[:25, :, :, :]).type(dtype)
        actions = model(images[:,:2,:,:])
        actions = np.argmax(actions.cpu().detach().numpy(), axis=1)

        for i in range(5):
            for j in range(5):
                plt.subplot(5, 5, i * 5 + j + 1)
                plot_state(val_data[i*5 + j, :, :, :].transpose([1,2,0]), win_actions=val_win_labels[i*5+j, :], lose_actions=val_lose_labels[i*5+j, :], action=actions[i*5+j])

    os.getcwd()
    plt.show()


def train(model, optimizer, criterion, train_data, train_labels, val_data, val_labels):
    batch_epoch = 0
    learning_curve = []
    validation_stats = []

    for epoch in range(epochs):
        running_error = 0
        batch_num = 0
        idx = np.array(list(range(TRAIN_SIZE)))
        np.random.shuffle(idx)

        for batch_idx in np.split(idx, TRAIN_SIZE/batch_size):
            # Get batch data
            states = Variable(torch.Tensor(train_data[batch_idx, :, :, :])).type(dtype)
            #win = train_win_labels[batch_idx, :]
            #lose = train_lose_labels[batch_idx, :]
            labels = Variable(torch.Tensor(train_labels[batch_idx, :])).type(dtype)

            # Evaluate
            actions = model(states[:,:2,:,:])

            error = criterion(actions, labels)
            batch_error = evaluate(actions, labels)
            running_error += error.item()
            batch_epoch += 1
            batch_num += 1

            # Do step at the end to keep that it will be done after validation
            optimizer.zero_grad()
            #for param in model.parameters():
            #    print(param.grad.data)
            error.backward()  # Compute the gradient for each variable
            #for param in model.parameters():
            #    print(param.grad.data)
            optimizer.step()  # Update the weights according to the computed gradient

            # Plot average train loss across 10 batches and validation loss
            if not batch_num % 10:

                # Calculate validation loss
                val_error = -1
                if not batch_num % 100:
                    with torch.no_grad():
                        val_data_tensor = torch.Tensor(val_data).type(dtype)
                        val_labels_tensor = torch.Tensor(val_labels).type(dtype)
                        val_actions = model(val_data_tensor[:,:2,:,:])
                        val_error = evaluate(val_actions, val_labels, True)
                        validation_stats.append((batch_epoch, val_error))

                if val_error >= 0:
                    #output_file.write('Epoch: {:2d} Batch: {:3d} Loss: {:.5f} Error: {:.5f} Validation Error: {:.5f}\n'
                    #                  .format(t, batch_num, loss.item(), batch_error, val_error))
                    print(
                        '{} Epoch: {:2d} Batch: {:3d} Loss: {:.5f} Error: {:.5f} Validation Error: {:.5f}'
                        .format(0, epoch, batch_num, error.item(), batch_error, val_error)
                    )
                else:
                    #output_file.write('Epoch: {} Batch: {} Loss: {}\n'
                    #                  .format(t, batch_idx, loss.item()))
                    print(
                        '{} Epoch: {:2d} Batch: {:3d} Loss: {:.5f} Error: {:.5f}'
                        .format(0, epoch, batch_num, error.item(), batch_error)
                    )

                # Plot every 10 batches
                # Average train loss over 10 batches for stability
                learning_curve.append((batch_epoch, running_error / 10))

                running_error = 0



    return learning_curve, validation_stats


def evaluate(actions, labels, dist=False):
    action_argmax = np.argmax(actions.cpu().detach().numpy(), axis=1)
    labels_argmax = np.array([ [labels[i, action_argmax[i]]] for i in range(labels.shape[0]) ])
    if dist:
        print(np.sum(labels, axis=0))
        action_dist = [sum(action_argmax == i) for i in range(BOARD_W)]
        print(action_dist)
    error_rate = np.mean(1 - labels_argmax)
    return error_rate


if __name__ == "__main__":
    main()
