import os, pickle

import numpy as np
import matplotlib.pyplot as plt

import torch

from utils_game import Game
from utils_plot import plot_state
from utils_save import load_model

from model import DQN_CNN_WIDE, DQN_CNN_WIDE_PREDICTION

USE_CUDA = False
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def play_game(env, Q1, Q2, plt_flag=False):

    # Play game
    if plt_flag:
        plt.ion()
        plt.figure(figsize=(20, 10))
        stats1 = np.zeros([2, 7])
        stats1_symm = np.zeros([2, 7])
        stats2 = np.zeros([2, 7])
        stats2_symm = np.zeros([2, 7])

    action=None
    obs, _ = env.reset()
    done = False
    win = False
    lose = False
    draw = False
    while not done:
        if plt_flag:
            plt.subplot(2, 1, env.player + 1)
            input_to_dqn = torch.from_numpy(obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)
            stats1[0, :] = Q1(input_to_dqn).data.cpu().numpy()
            stats2[0, :] = Q2(input_to_dqn).data.cpu().numpy()
            opponent_obs = env.swap_state(player=1)[:, :, :2]
            opponent_input_to_dqn = torch.from_numpy(opponent_obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)
            stats1[1, :] = Q1(opponent_input_to_dqn).data.cpu().numpy()
            stats2[1, :] = Q2(opponent_input_to_dqn).data.cpu().numpy()

            input_to_dqn = np.flip(obs.transpose(2, 0, 1), axis=2).copy()
            input_to_dqn = torch.from_numpy(input_to_dqn).type(dtype).unsqueeze(0)
            stats1_symm[0, :] = np.flip(Q1(input_to_dqn).data.cpu().numpy(), axis=1)
            stats2_symm[0, :] = np.flip(Q2(input_to_dqn).data.cpu().numpy(), axis=1)
            opponent_obs = env.swap_state(player=1)[:, :, :2]
            opponent_input_to_dqn = np.flip(opponent_obs.transpose(2, 0, 1), axis=2).copy()
            opponent_input_to_dqn = torch.from_numpy(opponent_input_to_dqn).type(dtype).unsqueeze(0)
            stats1_symm[1, :] = np.flip(Q1(opponent_input_to_dqn).data.cpu().numpy(), axis=1)
            stats2_symm[1, :] = np.flip(Q2(opponent_input_to_dqn).data.cpu().numpy(), axis=1)
            plot_state(env.state, 'Game Turn {}'.format(env.turn), action=action)

            for col in range(len(stats1[0])):
                plt.text(col - 0.2, 5 - 0.2, '{:.2f}'.format(stats1[0][col]), fontsize=12, color='w')
                plt.text(col - 0.2, 5 + 0.2, '{:.2f}'.format(stats1[1][col]), fontsize=12, color='w')
                plt.text(col - 0.2, 3 - 0.2, '{:.2f}'.format(stats2[0][col]), fontsize=12, color='w')
                plt.text(col - 0.2, 3 + 0.2, '{:.2f}'.format(stats2[1][col]), fontsize=12, color='w')

                plt.text(col - 0.2, 4 - 0.2, '{:.2f}'.format(stats1_symm[0][col]), fontsize=12, color='w')
                plt.text(col - 0.2, 4 + 0.2, '{:.2f}'.format(stats1_symm[1][col]), fontsize=12, color='w')
                plt.text(col - 0.2, 2 - 0.2, '{:.2f}'.format(stats2_symm[0][col]), fontsize=12, color='w')
                plt.text(col - 0.2, 2 + 0.2, '{:.2f}'.format(stats2_symm[1][col]), fontsize=12, color='w')

            plt.show()
            plt.pause(0.1)

        if env.player == 0:
            # Get player action
            input_to_dqn = torch.from_numpy(obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)
            with torch.no_grad():
                action = Q1(input_to_dqn).data.max(dim=1)[1].cpu().numpy()
            obs, reward, done, _ = env.step(action)
            if done:
                if reward > 0:
                    win = True
                elif reward < 0:
                    lose = True
                else:
                    draw = True
        else:
            # Get opponent action
            opponent_obs = env.swap_state(player=1)[:, :, :2]
            opponent_input_to_dqn = torch.from_numpy(opponent_obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)
            with torch.no_grad():
                action = Q2(opponent_input_to_dqn).data.max(dim=1)[1].cpu().numpy()
            obs, reward, done, _ = env.step(action)
            if done:
                if reward > 0:
                    lose = True
                elif reward < 0:
                    win = True
                else:
                    draw = True

    if plt_flag:
        plt.subplot(2, 1, 1)
        plot_state(env.state, 'Game Turn {}'.format(env.turn), action=action)
        plt.show()
        plt.pause(0.01)
        print(action)
        print(reward)

    return win - lose

if __name__ == "__main__":


    #checkpoint_path = './model_5_01_lr_5e6_symmetry_good_gc/model_max_wins_6_mask.pth.tar'
    checkpoint_path = []
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_ec_gc/{}.pth.tar'.format('model_min_error_rate'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_ec_gc/{}.pth.tar'.format('model_max_wins_5'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_symmetry_ec_gc/{}.pth.tar'.format('model_min_error_rate'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_symmetry_ec_gc/{}.pth.tar'.format('model_max_wins_6'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_5e6_symmetry/{}.pth.tar'.format('model_min_error_rate'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_5e6_symmetry/{}.pth.tar'.format('model_max_wins_6'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_5e6_symmetry_ec_gc/{}.pth.tar'.format('model_min_error_rate'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_5e6_symmetry_ec_gc/{}.pth.tar'.format('model_max_wins_6'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_5e6_symmetry_gc/{}.pth.tar'.format('model_min_error_rate'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_5e6_symmetry_gc/{}.pth.tar'.format('model_max_wins_6'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_symmetry_ec_gc_pr/{}.pth.tar'.format('model_min_error_rate'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_symmetry_ec_gc_pr/{}.pth.tar'.format('model_max_wins_6'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_symmetry_gc_pr/{}.pth.tar'.format('model_min_error_rate'))
    checkpoint_path.append('./checkpoints/model_5_01_lr_1e5_symmetry_gc_pr/{}.pth.tar'.format('model_max_wins_6'))

    num_models = len(checkpoint_path)
    modeltype = ['min_error', 'best_minimax']
    models = []
    params = []
    for i in range(len(checkpoint_path)):
        models.append(DQN_CNN_WIDE_PREDICTION() if i in [10,11,12,13] else DQN_CNN_WIDE())
        params.append(load_model(models[i], checkpoint_path[i]))
        print(params[i])

    env = Game()

    scores = np.zeros([num_models, num_models+1])
    for i in range(num_models):
        for j in range(i, num_models):
            scores[i, j] = play_game(env, models[i], models[j]) - play_game(env, models[j], models[i])
            scores[j, i] = -scores[i, j]
    scores[:,num_models] = np.sum(scores[:,:num_models], axis=1) / (num_models-1)

    plt.figure()
    mat = plt.imshow(scores)
    plt.xticks(range(num_models))
    plt.yticks(range(num_models), ['{}: {} ({})'.format(i, p.split('/')[2], modeltype[i % 2]) for i, p in enumerate(checkpoint_path)])
    plt.colorbar(mat, ticks=[-2, -1, 0, 1, 2])
    for i in range(num_models):
        plt.text(num_models - 0.3, i, '{:.2f}'.format(scores[i,num_models]), fontsize=12, color='w')
    plt.title('Play Results')
    plt.show()

    plt.figure()
    error = np.zeros([num_models])
    for i in range(num_models):
        if i < 2:
            filename = os.path.join(os.path.split(checkpoint_path[i])[0], 'Statistics.pkl')
            with open(filename, 'rb') as f:
                stats = pickle.load(f)
            error[i] = min([e for e in stats['ERROR_RATE']])
        else:
            error[i] = params[i]['stats']['ERROR_RATE']
    plt.barh(list(reversed(range(num_models))), error)
    plt.xlabel('validation error')
    plt.yticks(list(reversed(range(num_models))), ['{}: {} ({})'.format(i, p.split('/')[2], modeltype[i % 2]) for i, p in enumerate(checkpoint_path)])
    plt.title('Validation Error')
    plt.grid()
    plt.show()