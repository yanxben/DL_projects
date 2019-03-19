import random, time

import numpy as np
import matplotlib.pyplot as plt

import torch

from utils_plot import plot_state

USE_CUDA = False
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


def play_game(env, Q):

    # Play game
    plt.ion()
    plt.figure(figsize=(20, 10))
    flag = 1
    stats = np.zeros([2, 7])
    stats_symm = np.zeros([2, 7])
    action=None
    while flag:
        obs, _ = env.reset(random.randint(0, 1))
        done = False
        win = False
        lose = False
        draw = False
        while not done:
            plt.subplot(2, 1, env.player + 1)
            input_to_dqn = torch.from_numpy(obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)
            stats[0, :] = Q(input_to_dqn).data.cpu().numpy()
            opponent_obs = env.swap_state(player=1)[:, :, :2]
            opponent_input_to_dqn = torch.from_numpy(opponent_obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)
            stats[1, :] = Q(opponent_input_to_dqn).data.cpu().numpy()

            input_to_dqn = np.flip(obs.transpose(2, 0, 1), axis=2).copy()
            input_to_dqn = torch.from_numpy(input_to_dqn).type(dtype).unsqueeze(0)
            stats_symm[0, :] = np.flip(Q(input_to_dqn).data.cpu().numpy(), axis=1)
            opponent_obs = env.swap_state(player=1)[:, :, :2]
            opponent_input_to_dqn = np.flip(opponent_obs.transpose(2, 0, 1), axis=2).copy()
            opponent_input_to_dqn = torch.from_numpy(opponent_input_to_dqn).type(dtype).unsqueeze(0)
            stats_symm[1, :] = np.flip(Q(opponent_input_to_dqn).data.cpu().numpy(), axis=1)
            plot_state(env.state, 'Game Turn {}'.format(env.turn), action=action, stats=stats, stats_symm=stats_symm)
            plt.show()
            plt.pause(0.01)

            if env.player == 0:
                # Get player action
                action = int(input("What is your move? (Choose from 0 to {})".format(env.BOARD_W - 1)))
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

                tic = time.perf_counter()
                with torch.no_grad():
                    action = Q(opponent_input_to_dqn).data.max(dim=1)[1].cpu().numpy()
                print('time: {}'.format(time.perf_counter() - tic))

                obs, reward, done, _ = env.step(action)
                if done:
                    if reward > 0:
                        lose = True
                    elif reward < 0:
                        win = True
                    else:
                        draw = True

        plt.subplot(2, 1, 1)
        plot_state(env.state, 'Game Turn {}'.format(env.turn), action=action)
        plt.show()
        plt.pause(0.01)
        print(action)
        print(reward)
        if win:
            print('YOU WIN')
        if lose:
            print('YOU LOSE!')
        if draw:
            print('DRAW!')
        flag = int(input("Play again? (Choose from 0,1)"))


if __name__ == "__main__":
    from model import DQN_CNN_WIDE_PREDICTION
    from utils_save import load_model
    from utils_game import Game

    env = Game()

    Q = DQN_CNN_WIDE_PREDICTION()
    model_path = './best_model/'
    model = 'best_model'
    checkpoint_path = model_path + model + '.pth.tar'
    params = load_model(Q, checkpoint_path)
    print(params)

    def policy(obs):
        return Q(obs)

    play_game(env, policy)
