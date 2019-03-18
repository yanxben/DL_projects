import os, random

import numpy as np
import torch
import torch.optim as optim

from model import DQN_CNN, DQN_CNN_WIDE, DQN_CNN_WIDE_PREDICTION, DQN_LINEAR, DQN_SKIP, DQN_SKIP_WIDE
from dqn_train import DQNLearning
from dqn_train import OptimizerSpec
from utils_game import Game
from utils_data import load_end_game_data
from utils_schedule import LinearSchedule
from utils_plot import plot_stats

REPLAY_BUFFER_SIZE = 100000
LEARNING_STARTS = 100000
LEARNING_ENDS = 10000000
GAMMA = 0.99
LEARNING_FREQ = 5
TARGET_UPDATE_FREQ = 5
LOG_FREQ = 4000
BATCH_SIZE = 64
LR_a = 1
LR_b = 5
LEARNING_RATE = LR_a * (10 ** -LR_b)
ALPHA = 0.95
EPS = 0.1
EPS_END = 5000000
SYMMETRY = True
ERROR_CLIP = True
GRAD_CLIP = True
PREDICTION = True
ACTION_MASK = True

# Construct prefix
PREFIX = '{}_{}_{}_lr_{}e{}'.format(LEARNING_ENDS // 1000000, EPS_END // 1000000, EPS, LR_a, LR_b)
if SYMMETRY:
    PREFIX += '_symmetry'
if ERROR_CLIP:
    PREFIX += '_ec'
if GRAD_CLIP:
    PREFIX += '_gc'
if PREDICTION:
    PREFIX += '_pr'
if ACTION_MASK:
    PREFIX += '_am'
PREFIX = PREFIX.replace('.', '')

# Look for GPU
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def train_model(game, validation_data=None, validation_labels=None):
    env = game()

    # Define optimizer for the model
    optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE),
    )

    # Schedule exploration parameter
    exploration = LinearSchedule(EPS_END, EPS)

    # Construct an epsilon greedy policy with given exploration schedule
    def epsilon_greedy_policy(model, obs, t, action_mask=None):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
            action = model(obs)
            return action.data.max(dim=1)[1].cpu().numpy(), True
        else:
            if action_mask is None:
                return np.random.choice(np.arange(env.BOARD_W)), False
            else:
                return random.choice(np.arange(env.BOARD_W)[action_mask == 1]), False

    print(" \n \
    PREFIX {} \n\
    REPLAY_BUFFER_SIZE {} \n\
    LEARNING_STARTS {} \n\
    LEARNING_ENDS {} \n\
    GAMMA {} \n\
    LEARNING_FREQ {} \n\
    TARGET_UPDATE_FREQ {} \n\
    LOG_FREQ {} \n\
    BATCH_SIZE {} \n\
    LR_a {} \n\
    LR_b {} \n\
    LEARNING_RATE {} \n\
    ALPHA {} \n\
    EPS {} \n\
    EPS_END {} \n\
    SYMMETRY {} \n\
    ERROR_CLIP {} \n\
    GRAD_CLIP {} \n\
    PREDICTION {} \n\
    ACTION_MASK {}".format(PREFIX, REPLAY_BUFFER_SIZE, LEARNING_STARTS, LEARNING_ENDS, GAMMA, LEARNING_FREQ,
                           TARGET_UPDATE_FREQ, LOG_FREQ, BATCH_SIZE, LR_a, LR_b, LEARNING_RATE, ALPHA, EPS, EPS_END,
                           SYMMETRY, ERROR_CLIP, GRAD_CLIP, PREDICTION, ACTION_MASK))

    save_path = './checkpoints/model_{}/'.format(PREFIX)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    Q, statistics = DQNLearning(
        game=game,
        q_func=DQN_CNN_WIDE_PREDICTION if PREDICTION else DQN_CNN_WIDE,
        optimizer_spec=optimizer_spec,
        policy_func=epsilon_greedy_policy,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_ends=LEARNING_ENDS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=1,
        target_update_freq=TARGET_UPDATE_FREQ,
        log_freq=LOG_FREQ,
        validation_data=validation_data,
        validation_labels=validation_labels,
        save_path=save_path,
        symmetry=SYMMETRY,
        error_clip=ERROR_CLIP,
        grad_clip=GRAD_CLIP,
        prediction=PREDICTION,
        action_mask=ACTION_MASK
    )

    # Plot and save stats
    plot_path = './plots/'
    plot_stats(statistics, path=plot_path, prefix=PREFIX+'_')


if __name__ == '__main__':
    # Load validation set
    DATA_SIZE = 1000
    # Temporal swap until recollection
    #data, labels, win_labels, lose_labels = load_end_game_data(1000)
    data, labels, lose_labels, win_labels = load_end_game_data(1000)
    for n in range(labels.shape[0]):
        if any(win_labels[n,:]):
            labels[n, :] = win_labels[n,:]
        else:
            labels[n, :] = lose_labels[n,:]
    # Split train-test
    #val_data = data[TRAIN_SIZE:, :, :, :]
    #train_win_labels = win_labels[:TRAIN_SIZE, :]
    #train_lose_labels = lose_labels[:TRAIN_SIZE, :]
    #val_win_labels = win_labels[TRAIN_SIZE:, :]
    #val_lose_labels = lose_labels[TRAIN_SIZE:, :]

    #train_labels = labels[:TRAIN_SIZE, :]
    #val_labels = labels[DATA_SIZE:, :]

    # Run training
    game = Game

    train_model(game, data, labels)
    print('DONE')

