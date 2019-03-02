import os, random

import torch
import torch.optim as optim

from model import DQN_FCN, DQN_FCN_WIDE, DQN_LINEAR, DQN_SKIP
from dqn_train import DQNLearning3
from dqn_train import OptimizerSpec
from utils_game import Game
from utils_data import load_end_game_data
from utils_schedule import LinearSchedule, ConstSchedule
from utils_plot import plot_stats, plot_state
from play import play_game

REPLAY_BUFFER_SIZE = 10000
LEARNING_STARTS = 10000
LEARNING_ENDS = 10000000
GAMMA = 0.99
LEARNING_FREQ = 5
TARGET_UPDATE_FREQ = 5
LOG_FREQ = 4000
BATCH_SIZE = 64
LR_a = 5
LR_b = 6
LEARNING_RATE = LR_a * (10 ** -LR_b)
ALPHA = 0.95
EPS = 0.1
EPS_END = 5000000
SYMMETRY = True
ERROR_CLIP = False
GRAD_CLIP = False

# Construct prefix
PREFIX = 'test_{}_{}_lr_{}e{}'.format(int(EPS_END / 1000000), EPS, LR_a, LR_b)
if SYMMETRY:
    PREFIX += '_symmetry_good'
if ERROR_CLIP:
    PREFIX += '_ec'
if GRAD_CLIP:
    PREFIX += '_gc'

PREFIX = PREFIX.replace('.', '')

PREFIX = ['5_01_lr_5e6_symmetry_good_ec_gc',
          '5_01_lr_5e6_symmetry_good',
          '5_01_lr_5e6_symmetry_good_gc']

PREFIX = ['very_wide_5_01_lr_5e6_symmetry_good',
          'very_wide_5_01_lr_5e6_symmetry_good',
          'very_wide_5_01_lr_5e6_symmetry_good']

MODELNAME = 'model_min_error_rate'

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
    exploration = ConstSchedule(EPS)

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

    # Save configurations to txt file
    # TODO

    num_models = len(PREFIX)
    load_paths = ['./model_{}/{}.pth.tar'.format(p, MODELNAME) for p in PREFIX]
    save_paths = ['./model_{}_X/'.format(p) for p in PREFIX]

    for i in range(num_models):
        if not os.path.isdir(save_paths[i]):
            os.mkdir(save_paths[i])

    Q, statistics = DQNLearning3(
        game=game,
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
        load_paths=load_paths,
        save_paths=save_paths,
        symmetry=SYMMETRY,
        error_clip=ERROR_CLIP,
        grad_clip=GRAD_CLIP
    )

    # Plot and save stats
    for n in range(num_models):
        plot_stats(statistics[n], path=save_paths[n], prefix=PREFIX+'_X_')


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
        if any(win_labels[n, :]):
            labels[n, :] = win_labels[n, :]
        else:
            labels[n, :] = lose_labels[n, :]


    # Run training
    #seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    game = Game

    train_model(game, data, labels)
    print('DONE')

