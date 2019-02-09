import os, random

import torch
import torch.optim as optim

from model import DQN_FCN, DQN_FCN_WIDE, DQN_LINEAR, DQN_SKIP
from dqn_train import DQNLearning
from dqn_train import OptimizerSpec
from utils_game import Game
from utils_data import load_end_game_data
from utils_schedule import LinearSchedule
from utils_plot import plot_stats, plot_state
from play import play_game

PREFIX = '5_01_symmetry'
BATCH_SIZE = 64
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 100000
LEARNING_STARTS = 100000
LEARNING_ENDS = 10000000
LEARNING_FREQ = 5
TARGET_UPDATE_FREQ = 5
LOG_FREQ = 1000
LEARNING_RATE = 1e-5
ALPHA = 0.95
EPS = 0.1

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def train_model(env, validation_data=None, validation_labels=None):

    # Define optimizer for the model
    optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE),
    )

    # Schedule exploration parameter
    exploration = LinearSchedule(5000000, EPS)

    # Construct an epsilon greedy policy with given exploration schedule
    def epsilon_greedy_policy(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
            action = model(obs)
            return action.data.max(dim=1)[1].cpu().numpy(), True
        else:
            return random.randint(0, env.BOARD_W-1), False

    # Set stopping criterion
    def stopping_criterion(t):
        return t > LEARNING_ENDS

    print(" \n \
        PREFIX {} \n \
        BATCH_SIZE {} \n \
        GAMMA {} \n \
        REPLAY_BUFFER_SIZE {} \n \
        LEARNING_STARTS {} \n \
        LEARNING_FREQ {} \n \
        TARGET_UPDATE_FREQ {} \n \
        LOG_FREQ {} \n \
        LEARNING_RATE {} \n \
        ALPHA {} \n \
        EPS {}".format(PREFIX, BATCH_SIZE, GAMMA, REPLAY_BUFFER_SIZE, LEARNING_STARTS, LEARNING_FREQ, TARGET_UPDATE_FREQ, LOG_FREQ, LEARNING_RATE, ALPHA, EPS))

    save_path = './model_{}/'.format(PREFIX)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    Q, Statistic = DQNLearning(
        env=env,
        q_func=DQN_FCN_WIDE,
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
        symmetry=True
    )

    # Plot and save stats
    plot_stats(Statistic, path=save_path, prefix=PREFIX+'_')

    # Play Game
    play_game(env, Q)



if __name__ == '__main__':
    # Get Atari games.
    # Change the index to select a different game.
    #task = benchmark.tasks[3]

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
    #seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = Game()

    train_model(env, data, labels)
    print('DONE')

