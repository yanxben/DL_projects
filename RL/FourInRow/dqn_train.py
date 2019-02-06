"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys, time
import pickle
import numpy as np
from collections import namedtuple, Counter
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer

from utils_game import Game
from utils_save import save_model
from utils_plot import plot_obs

from algorithms import minimax, alphabeta
from train_supervised import evaluate

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    'TURNS_RATE': [],
    'ERROR_RATE': [],
    'MINIMAX_1': [],
    'MINIMAX_2': [],
    'MINIMAX_4': []
}


def DQNLearning(
    env,
    q_func,
    optimizer_spec,
    policy_func,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=100000,
    learning_ends=10000000,
    learning_freq=5,
    frame_history_len=1,
    target_update_freq=5,
    log_freq=10,
    validation_data=None,
    validation_labels=None,
    save_path='./checkpoints/'
):
    """Run Deep Q-learning algorithm.

    Parameters
    ----------
    env: environment
        Environment to train on.
    q_func: function
        Model to use for computing the q function.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (t) -> bool
        Criterion for stopping the train routine.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network.
    validation_data: numpy array
        Validation data for validation between epochs.
    validation_labels: numpy array
        Validation labels for validaition between epochs.

    """

    ###############
    # BUILD MODEL #
    ###############
    # Initialize target q function and q function, i.e. build the model.
    Q = q_func().type(dtype)
    target_Q = q_func().type(dtype)
    target_Q.load_state_dict(Q.state_dict())
    target_Q.eval()

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    # Initialize parameters
    t0 = 0
    t0_time = time.perf_counter()
    epoch = 0
    plays = 0
    num_param_updates = 0
    store_step = False
    state_dict = [None] * target_update_freq
    reward_history = []
    turn_history = []

    # Checkpoint parameters
    min_error_rate = 1.
    most_wins = 0
    min_error_rate_wins = 1.
    most_wins_minimax4 = 0
    most_wins_with_minimax4 = 0
    min_error_rate_minimax4 = 1.

    # Initialize environment
    last_obs, _ = env.reset()
    done = False
    reward = 0
    players = env.players

    validation_data = torch.Tensor(validation_data).type(dtype).cuda()

    # Start train routine
    for t in count():
        # Check stopping criterion
        if t > learning_ends:
            break

        # Perform env step according to player
        if players == 1 or env.player == 0:
            # Store last obs
            index_obs_stored = replay_buffer.store_frame(last_obs)

            # Get input to dqc
            input_to_dqn = replay_buffer.encode_recent_observation()

            # Choose action according to play policy
            action, flag = policy_func(Q, input_to_dqn, t)

            # Perform env step according to action
            obs, reward, done, _ = env.step(action)

            # Raise flag for store step
            store_step = True

        #    if plays % 1000 == 0:
        #        plot_obs(obs, title='{} {}' .format('Random' if flag else 'Greedy', done))
        # Perform env step according to opponents
        if players > 1:
            while env.player != 0:
                if done:
                    break
                # Swap observed perspective
                opponent_obs = env.swap_state(player=env.player)[:, :, :2]

                # Align obs shape to model input
                opponent_input_to_dqn = torch.from_numpy(opponent_obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)

                # Choose action according to model
                with torch.no_grad():
                    opponent_action = Q(opponent_input_to_dqn).data.max(dim=1)[1].cpu().numpy()

                # perform env step according to action
                new_obs, new_reward, new_done, _ = env.step(opponent_action, player=env.player)

                # Aggregate information
                #reward -= new_reward
                reward -= new_reward if new_reward >= 0 else -0.01  # Do not learn from opponent stupidity
                done = new_done
                obs = new_obs

        #if plays % 1000 == 0:
        #    plot_obs(obs, 'Greedy {}' .format(done))

        # Store data needed for the curr step
        if store_step:
            replay_buffer.store_effect(index_obs_stored, action, reward, done)
        store_step = False

        # If done, start new game
        if done:
            # Store data history
            if len(reward_history) >= 1000:
                reward_history[plays % 1000] = reward
                turn_history[plays % 1000] = env.turn
            else:
                reward_history.append(reward)
                turn_history.append(env.turn)

            # Restart game
            last_obs, _ = env.reset(random.choice(range(2)))
            done = False
            reward = 0
            plays += 1
        else:
            last_obs = obs

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        # Perform experience replay and train the network
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # Collect experience batch
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
            act_batch = Variable(torch.from_numpy(act_batch)).type(torch.LongTensor).view(-1,1)
            rew_batch = Variable(torch.from_numpy(rew_batch).type(dtype))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype))
            done_mask = Variable(torch.from_numpy(done_mask).type(dtype))

            if USE_CUDA:
                act_batch = act_batch.cuda()

            # Calculate Q values for chosen action
            current_Q_values = Q(obs_batch).gather(1, act_batch).squeeze(1)

            # Calculate best action for next state
            with torch.no_grad():
                next_act_batch = Q(next_obs_batch).max(dim=1)[1].detach().view(-1,1)

            # Extimate next step Q values
            with torch.no_grad():
                next_Q_max_unmasked = target_Q(next_obs_batch).gather(1, next_act_batch).detach().squeeze(1)

            # mask final steps
            next_Q_max_masked = next_Q_max_unmasked * (1 - done_mask)

            # Estimate Q values according to Bellman equation
            target_Q_values = rew_batch + (gamma * next_Q_max_masked)

            # Compute difference between current estimation and next step estimation
            d_error = (-1.0 * (target_Q_values - current_Q_values)).clamp(-1, 1)  # Clip error for stability

            # Update Q network
            optimizer.zero_grad()
            current_Q_values.backward(d_error.data)
            torch.nn.utils.clip_grad_value_(Q.parameters(), 1)  # Clip gradients for stability
            optimizer.step()  # Does the update

            # Update target Q network and validate benchmarks
            num_param_updates += 1
            state_dict.append(Q.state_dict())
            target_state_dict = state_dict.pop(0)
            if num_param_updates > len(state_dict):
                epoch += 1
                # Update target_Q network
                target_Q.load_state_dict(target_state_dict)

                # Evaluate model on benchmarks
                if epoch % log_freq == 0:
                    # Report reward and turn history
                    c = Counter(reward_history)
                    print('EPOCH {} T {} PLAYS {}'.format(epoch, t, plays))
                    print('WIN {} # LOSE {} # MYBAD {} # HISBAD {} # TIE {}'
                          .format(c[1] / len(reward_history),
                                  c[-1] / len(reward_history),
                                  c[-1.01] / len(reward_history),
                                  c[0.01] / len(reward_history),
                                  c[0] / len(reward_history)), end='')
                    Statistic['TURNS_RATE'].append(np.mean(turn_history))
                    print(' ##### AVG LENGTH {}' .format(Statistic['TURNS_RATE'][-1]))


                    # Evaluate model on critical actions (validation set)
                    if validation_data is not None:
                        with torch.no_grad():
                            val_actions = Q(validation_data[:, :2, :, :])
                        Statistic['ERROR_RATE'].append(evaluate(val_actions, validation_labels))
                        print('ERROR RATE {}'.format(Statistic['ERROR_RATE'][-1]))

                    # Evaluate model on minimax opponent
                    side_game = Game()
                    turns = [0] * 2
                    scores = [0] * 2

                    for i, depth in enumerate([1, 2, 4]):
                        score = [0] * 3
                        bad_moves = 0
                        print('MINIMAX # DEPTH {} # ' .format(depth), end='')
                        for p in range(2):
                            new_score, bad_move, turn = playGame(side_game, Q, depth, p % 2)
                            score[0] += new_score > 0
                            score[1] += new_score == 0
                            score[2] += new_score < 0
                            bad_moves += bad_move
                            turns[p] = turn
                            scores[p] = new_score

                        stats = score + [bad_moves, np.mean(turns)]
                        Statistic['MINIMAX_{}'.format(depth)].append(stats)
                        print('SCORE {}:{}:{} # BAD_MOVES {} $$$ {}:{} {}:{}' .format(score[0], score[1], score[2], bad_moves, turns[0], scores[0], turns[1], scores[1]))

                    # Save model params
                    if validation_data is not None and Statistic['ERROR_RATE'][-1] < min_error_rate:
                        model_save_filename = save_path + 'model_min_error_rate'
                        model_save_path = model_save_filename + '.pth.tar'
                        save_model(Q, model_save_path)
                        min_error_rate = Statistic['ERROR_RATE'][-1]

                    wins = Statistic['MINIMAX_1'][-1][0] + Statistic['MINIMAX_2'][-1][0] + Statistic['MINIMAX_4'][-1][0] # + Statistic['MINIMAX_6'][-1][0]
                    if wins > most_wins or \
                            wins == most_wins and Statistic['ERROR_RATE'][-1] < min_error_rate_wins:
                        model_save_filename = save_path + 'model_max_wins_{}'.format(wins)
                        model_save_path = model_save_filename + '.pth.tar'
                        save_model(Q, model_save_path)
                        most_wins = wins
                        min_error_rate_wins = Statistic['ERROR_RATE'][-1]

                    if Statistic['MINIMAX_4'][-1][0] > most_wins_minimax4 or \
                            Statistic['MINIMAX_4'][-1][0] == most_wins_minimax4 and wins > most_wins_with_minimax4 or \
                            Statistic['MINIMAX_4'][-1][0] == most_wins_minimax4 and wins == most_wins_with_minimax4 and Statistic['ERROR_RATE'][-1] < min_error_rate_minimax4:
                        model_save_filename = save_path + 'model_minimax4'
                        model_save_path = model_save_filename + '.pth.tar'
                        save_model(Q, model_save_path)
                        most_wins_minimax4 = Statistic['MINIMAX_4'][-1][0]
                        most_wins_with_minimax4 = wins
                        min_error_rate_minimax4 = Statistic['ERROR_RATE'][-1]

                    print('---------------------------------------')
                    tm = time.localtime()
                    dt = (time.perf_counter() - t0_time) / 60
                    eta = round(dt * (learning_ends - t) / (t - t0))
                    eta_min = eta % 60
                    eta_hour = (eta // 60) % 24
                    eta_day = (eta // 60) // 24
                    print('TIME: {:02d}:{:02d}' .format(tm.tm_hour, tm.tm_min), end='')
                    print(' ### ETA: {}::{}:{}' .format(eta_day, eta_hour, eta_min))
                    t0_time = time.perf_counter()
                    t0 = t

    # Save last model
    model_save_filename = save_path + 'model_last'
    model_save_path = model_save_filename + '.pth.tar'
    save_model(Q, model_save_path)

    # Dump statistics to pickle
    with open(save_path + 'statistics.pkl', 'wb') as f:
        pickle.dump(Statistic, f)

    # Return trained model and stats
    return Q, Statistic


def playGame(side_game, model, depth, player=None):
    if player==None:
        player = random.randrange(2)
    obs, _ = side_game.reset(player)
    done = False

    bad_move = 0

    while not done:
        if side_game.player == 0:
            input_to_dqn = torch.from_numpy(obs.transpose(2,0,1)).type(dtype).unsqueeze(0)
            action = model(input_to_dqn).data.max(dim=1)[1].cpu().numpy()

            obs, reward, done, _ = side_game.step(action)

            if reward < 0:
                bad_move = 1
            if done:
                return reward, bad_move, side_game.turn
        else:
            state = side_game.swap_state(player=1)
            action, _, _ = alphabeta(state, depth, float('inf'))

            obs, reward, done, _ = side_game.step(action, player=1)

            if done:
                return -reward, 0, side_game.turn
