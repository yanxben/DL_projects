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

from utils_replay_buffer import ReplayBuffer

from utils_game import Game, _valid_action
from utils_save import save_model, load_model

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


def DQNLearning(
    game,
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
    save_path='./checkpoints/',
    symmetry=False,
    error_clip=False,
    grad_clip=False,
    prediction=False
):
    """Run Deep Q-learning algorithm.

    Parameters
    ----------
    game: environment
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
    state_dict = [None] * target_update_freq
    reward_history = []
    turn_history = []

    # Checkpoint parameters
    statistics = {
        'TURNS_RATE': [],
        'ERROR_RATE': [],
        'MINIMAX_1': [],
        'MINIMAX_2': [],
        'MINIMAX_4': [],
        'MINIMAX_1_MASK': [],
        'MINIMAX_2_MASK': [],
        'MINIMAX_4_MASK': [],
    }
    min_error_rate = 1.
    most_wins = 0
    min_error_rate_wins = 1.
    most_wins_minimax4 = 0
    most_wins_with_minimax4 = 0
    min_error_rate_minimax4 = 1.
    most_wins_mask = 0
    min_error_rate_wins_mask = 1.
    most_wins_minimax4_mask = 0
    most_wins_with_minimax4_mask = 0
    min_error_rate_minimax4_mask = 1.
    stats_values = [min_error_rate, most_wins, min_error_rate_wins,
                    most_wins_minimax4, most_wins_with_minimax4, min_error_rate_minimax4,
                    most_wins_mask, min_error_rate_wins_mask,
                    most_wins_minimax4_mask, most_wins_with_minimax4_mask, min_error_rate_minimax4_mask]

    # Initialize environment
    env = game()
    last_obs, _ = env.reset()

    validation_data = torch.Tensor(validation_data).type(dtype).cuda()

    # Start train routine
    for t in count():
        # Check stopping criterion
        if t > learning_ends:
            break

        # Perform env step according to player
        obs, done, reward = explore_step(env, last_obs, Q, Q, policy_func, replay_buffer, t)

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

            train_step(Q, target_Q, optimizer, replay_buffer, batch_size, gamma, symmetry, error_clip, grad_clip)

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
                    statistics['TURNS_RATE'].append(np.mean(turn_history))
                    print(' ##### AVG LENGTH {}' .format(statistics['TURNS_RATE'][-1]))

                    stats_values, t0, t0_time = evaluation_step(game,
                        Q, optimizer, validation_data, validation_labels,
                        save_path, statistics,
                        *stats_values, t, learning_ends, epoch, t0, t0_time
                    )
                    print('---------------------------------------')

    # Save last model
    model_save_filename = save_path + 'model_last'
    model_save_path = model_save_filename + '.pth.tar'
    save_model(Q, model_save_path, optimizer, params={'t': learning_ends, 'epoch': epoch,
                                                      'stats': {key: statistics[key][-1] for key in
                                                                list(statistics.keys())}})

    # Dump statistics to pickle
    with open(save_path + 'statistics.pkl', 'wb') as f:
        pickle.dump(statistics, f)

    # Return trained model and stats
    return Q, statistics


def DQNLearning3(
    game,
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
    load_paths=None,
    save_paths=['./checkpoints1/', './checkpoints2/', './checkpoints3/'],
    symmetry=False,
    error_clip=False,
    grad_clip=False
):
    """Run Deep Q-learning algorithm.

    Parameters
    ----------
    game: environment
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

    assert load_paths is not None and len(load_paths) == len(save_paths)

    ###############
    # BUILD MODEL #
    ###############
    num_models = len(load_paths)
    Q = []
    target_Q = []
    optimizer = []
    replay_buffer = []
    for n in range(num_models):
        # Initialize Q network, i.e. build the model.
        Q.append(q_func().type(dtype))
        # Construct Q network optimizer function
        optimizer.append(optimizer_spec.constructor(Q[n].parameters(), **optimizer_spec.kwargs))
        # Load trained model
        print(load_model(Q[n], load_paths[n], optimizer[n]))
        # Initialize target Q network
        target_Q.append(q_func().type(dtype))
        target_Q[n].load_state_dict(Q[n].state_dict())
        target_Q[n].eval()
        # Construct the replay buffer
        replay_buffer.append(ReplayBuffer(replay_buffer_size, frame_history_len))

    ###############
    # RUN ENV     #
    ###############
    # Initialize parameters
    t0 = [0 for _ in range(num_models)]
    t0_time = [time.perf_counter() for _ in range(num_models)]
    epoch = 0
    #model_playing = 0
    model_countering = [random.randrange(num_models) for _ in range(num_models)]
    plays = [0] * num_models
    num_param_updates = 0
    state_dict = [[None] * target_update_freq for _ in range(num_models)]
    reward_history = [[] for _ in range(num_models)]
    turn_history = [[] for _ in range(num_models)]

    # Checkpoint parameters
    statistics = [{
        'TURNS_RATE': [],
        'ERROR_RATE': [],
        'MINIMAX_1': [],
        'MINIMAX_2': [],
        'MINIMAX_4': [],
        'MINIMAX_1_MASK': [],
        'MINIMAX_2_MASK': [],
        'MINIMAX_4_MASK': [],
    } for _ in range(num_models)]

    min_error_rate = 1.
    most_wins = 0
    min_error_rate_wins = 1.
    most_wins_minimax4 = 0
    most_wins_with_minimax4 = 0
    min_error_rate_minimax4 = 1.
    most_wins_mask = 0
    min_error_rate_wins_mask = 1.
    most_wins_minimax4_mask = 0
    most_wins_with_minimax4_mask = 0
    min_error_rate_minimax4_mask = 1.
    stats_values = [[min_error_rate, most_wins, min_error_rate_wins,
                    most_wins_minimax4, most_wins_with_minimax4, min_error_rate_minimax4,
                    most_wins_mask, min_error_rate_wins_mask,
                    most_wins_minimax4_mask, most_wins_with_minimax4_mask, min_error_rate_minimax4_mask]
                    for _ in range(num_models)]

    # Initialize environment
    envs = [game() for _ in range(num_models)]
    last_obs = [None for _ in range(num_models)]
    for n in range(num_models):
        last_obs[n], _ = envs[n].reset()

    validation_data = torch.Tensor(validation_data).type(dtype).cuda()

    # Start train routine
    for t in count():
        # Check stopping criterion
        if t > learning_ends:
            break

        for n in range(num_models):
            # Perform env step according to player
            obs, done, reward = explore_step(envs[n], last_obs[n], Q[n], Q[model_countering[n]], policy_func, replay_buffer[n], t)

            # If done, start new game
            if done:
                # Store data history
                if len(reward_history[n]) >= 1000:
                    reward_history[n][plays[n] % 1000] = reward
                    turn_history[n][plays[n] % 1000] = envs[n].turn
                else:
                    reward_history[n].append(reward)
                    turn_history[n].append(envs[n].turn)

                # Restart game
                last_obs[n], _ = envs[n].reset(random.choice(range(2)))
                plays[n] += 1
                #model_playing = (model_playing + 1) % num_models
                model_countering[n] = random.randrange(num_models)
            else:
                last_obs[n] = obs

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        # Perform experience replay and train the network
        if (t > learning_starts and
                t % learning_freq == 0 and
                all([replay_buffer[n].can_sample(batch_size) for n in range(num_models)])):

            for n in range(num_models):
                train_step(Q[n], target_Q[n], optimizer[n], replay_buffer[n], batch_size, gamma, symmetry, error_clip, grad_clip)

            # Update target Q network and validate benchmarks
            num_param_updates += 1
            for n in range(num_models):
                state_dict[n].append(Q[n].state_dict())
                target_state_dict = state_dict[n].pop(0)
                if num_param_updates > len(state_dict[n]):
                    if n == 0:
                        epoch += 1
                    # Update target_Q network
                    target_Q[n].load_state_dict(target_state_dict)

                    # Evaluate model on benchmarks
                    if epoch % log_freq == 0:
                        print('---------------------------------------')
                        print(' *** MODEL {} ***' .format(n))
                        # Report reward and turn history
                        c = Counter(reward_history[n])
                        print('EPOCH {} T {} PLAYS {}'.format(epoch, t, plays[n]))
                        print('WIN {} # LOSE {} # MYBAD {} # HISBAD {} # TIE {}'
                              .format(c[1] / len(reward_history[n]),
                                      c[-1] / len(reward_history[n]),
                                      c[-1.01] / len(reward_history[n]),
                                      c[0.01] / len(reward_history[n]),
                                      c[0] / len(reward_history[n])), end='')
                        statistics[n]['TURNS_RATE'].append(np.mean(turn_history[n]))
                        print(' ##### AVG LENGTH {}' .format(statistics[n]['TURNS_RATE'][-1]))

                        stats_values[n], t0[n], t0_time[n] = evaluation_step(game,
                            Q[n], optimizer[n], validation_data, validation_labels,
                            save_paths[n], statistics[n],
                            *stats_values[n], t, learning_ends, epoch, t0[n], t0_time[n]
                        )

    print('---------------------------------------')
    # Save last model
    for n in range(num_models):
        model_save_filename = save_paths[n] + 'model_last'
        model_save_path = model_save_filename + '.pth.tar'
        save_model(Q[n], model_save_path, optimizer[n], params={'t': learning_ends, 'epoch': epoch,
                                                                'stats': {key: statistics[n][key][-1] for key in
                                                                list(statistics[n].keys())}})

        # Dump statistics to pickle
        with open(save_paths[n] + 'statistics.pkl', 'wb') as f:
            pickle.dump(statistics[n], f)

    # Return trained model and stats
    return Q, statistics


def explore_step(env, last_obs, Q1, Q2, policy_func, replay_buffer, t):
    store_step = False
    reward = 0

    if env.players == 1 or env.player == 0:
        # Store last obs
        index_obs_stored = replay_buffer.store_frame(last_obs)

        # Get input to dqc
        input_to_dqn = replay_buffer.encode_recent_observation()

        # Choose action according to play policy
        action, flag = policy_func(Q1, input_to_dqn, t)

        # Perform env step according to action
        obs, reward, _, _ = env.step(action)

        # Raise flag for store step
        store_step = True

    # Perform env step according to opponents
    if env.players > 1:
        while env.player != 0:
            if env.done:
                break
            # Swap observed perspective
            opponent_obs = env.swap_state(player=env.player)[:, :, :2]

            # Align obs shape to model input
            opponent_input_to_dqn = torch.from_numpy(opponent_obs.transpose(2, 0, 1)).type(dtype).unsqueeze(0)

            # Choose action according to model
            with torch.no_grad():
                opponent_action = Q2(opponent_input_to_dqn).data.max(dim=1)[1].cpu().numpy()

            # perform env step according to action
            new_obs, new_reward, new_done, _ = env.step(opponent_action, player=env.player)

            # Aggregate information
            reward -= new_reward if new_reward >= 0 else -0.01  # Do not learn from opponent stupidity and exploit
            obs = new_obs

    # Store data needed for the curr step
    if store_step:
        replay_buffer.store_effect(index_obs_stored, action, reward, env.done)

    return obs, env.done, reward


def train_step(Q, target_Q, optimizer, replay_buffer, batch_size, gamma, symmetry=False, error_clip=False, grad_clip=False, prediction=False):
    # Collect experience batch
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

    if symmetry:  # This feature hopefully helps the model to generate by forcing symmetric similarity
        symm_mask1 = np.random.randint(0, 2, batch_size)
        symm_mask2 = np.random.randint(0, 2, batch_size)
        obs_batch[symm_mask1 == 1, :, :, :] = obs_batch[symm_mask1 == 1, :, :, ::-1]
        act_batch[symm_mask1 == 1] = 6 - act_batch[symm_mask1 == 1]
        next_obs_batch[symm_mask2 == 1, :, :, :] = next_obs_batch[symm_mask2 == 1, :, :, ::-1]

    obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype))
    act_batch = Variable(torch.from_numpy(act_batch)).type(torch.LongTensor).view(-1, 1)
    rew_batch = Variable(torch.from_numpy(rew_batch).type(dtype))
    next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype))
    done_mask = Variable(torch.from_numpy(done_mask).type(dtype))

    if USE_CUDA:
        act_batch = act_batch.cuda()

    # Calculate Q values for chosen action
    if prediction:
        current_Q_values_all, predict_done_all, predict_reward_all = Q(obs_batch, prediction)
        current_Q_values = current_Q_values_all.gather(1, act_batch).squeeze(1)
        predict_done = predict_done_all.gather(1, act_batch).squeeze(1)
        predict_reward = predict_reward_all.gather(1, act_batch).squeeze(1)

    else:
        current_Q_values = Q(obs_batch).gather(1, act_batch).squeeze(1)

    # Calculate best action for next state (according to Q)
    # Estimate next step Q values (according to target_Q)
    with torch.no_grad():
        next_act_batch = Q(next_obs_batch).max(dim=1)[1].detach().view(-1, 1)
        next_Q_max_unmasked = target_Q(next_obs_batch).gather(1, next_act_batch).detach().squeeze(1)

    # mask final steps
    next_Q_max_masked = next_Q_max_unmasked * (1 - done_mask)

    # Estimate Q values according to Bellman equation
    target_Q_values = rew_batch + (gamma * next_Q_max_masked)

    # Compute difference between current estimation and next step estimation
    loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

    # Compare prediction indicators
    if prediction:
        loss += 0.2 * F.binary_cross_entropy(predict_done, done_mask)
        loss += 0.2 * F.smooth_l1_loss(predict_reward, rew_batch)

    if error_clip:
        loss = loss.clamp(-1, 1)

    # Update Q network
    optimizer.zero_grad()
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_value_(Q.parameters(), 1)  # Clip gradients for stability
    optimizer.step()  # Does the update


def evaluation_step(game, Q, optimizer, validation_data, validation_labels,
                    save_path, statistics,
                    min_error_rate,
                    most_wins, min_error_rate_wins,
                    most_wins_minimax4, most_wins_with_minimax4, min_error_rate_minimax4,
                    most_wins_mask, min_error_rate_wins_mask,
                    most_wins_minimax4_mask, most_wins_with_minimax4_mask, min_error_rate_minimax4_mask,
                    t, learning_ends, epoch, t0, t0_time):
    # Evaluate model on critical actions (validation set)
    if validation_data is not None:
        with torch.no_grad():
            val_actions = Q(validation_data[:, :2, :, :])
        statistics['ERROR_RATE'].append(evaluate(val_actions, validation_labels))
        print('ERROR RATE {}'.format(statistics['ERROR_RATE'][-1]))

    # Evaluate model on minimax opponent
    side_game = game()
    turns = [0] * 2
    scores = [0] * 2
    turns_mask = [0] * 2
    scores_mask = [0] * 2

    for i, depth in enumerate([1, 2, 4]):
        score = [0] * 3
        bad_moves = 0
        score_mask = [0] * 3
        print('MINIMAX # DEPTH {} # '.format(depth), end='')

        for p in range(2):
            new_score, bad_move, turn, new_score_mask, turn_mask = play_game(side_game, Q, depth, p % 2, True)
            score[0] += new_score > 0
            score[1] += new_score == 0
            score[2] += new_score < 0
            bad_moves += bad_move
            turns[p] = turn
            scores[p] = new_score

            score_mask[0] += new_score_mask > 0
            score_mask[1] += new_score_mask == 0
            score_mask[2] += new_score_mask < 0
            turns_mask[p] = turn_mask
            scores_mask[p] = new_score_mask

        stats = score + [bad_moves, np.mean(turns)]
        stats_mask = score_mask + [0, np.mean(turns_mask)]
        statistics['MINIMAX_{}'.format(depth)].append(stats)
        statistics['MINIMAX_{}_MASK'.format(depth)].append(stats_mask)
        print('SCORE {}:{}:{} # BAD_MOVES {} $$$ {}:{} {}:{}'.format(score[0], score[1], score[2],
                                                                     bad_moves, turns[0], scores[0],
                                                                     turns[1], scores[1]))
        print('MINIMAX # DEPTH {} # '.format(depth), end='')
        print('SCORE {}:{}:{} #             $$$ {}:{} {}:{}'.format(score_mask[0], score_mask[1], score_mask[2],
                                                                    turns_mask[0], scores_mask[0],
                                                                    turns_mask[1], scores_mask[1]))

    # Save model params
    if validation_data is not None and statistics['ERROR_RATE'][-1] < min_error_rate:
        model_save_filename = save_path + 'model_min_error_rate'
        model_save_path = model_save_filename + '.pth.tar'
        save_model(Q, model_save_path, optimizer, params={'t': t, 'epoch': epoch,
                                                          'stats': {key: statistics[key][-1] for key in
                                                                    list(statistics.keys())}})
        min_error_rate = statistics['ERROR_RATE'][-1]

    wins = statistics['MINIMAX_1'][-1][0] + statistics['MINIMAX_2'][-1][0] + statistics['MINIMAX_4'][-1][
        0]  # + statistics['MINIMAX_6'][-1][0]
    if wins > most_wins or \
            wins == most_wins and statistics['ERROR_RATE'][-1] < min_error_rate_wins:
        model_save_filename = save_path + 'model_max_wins_{}'.format(wins)
        model_save_path = model_save_filename + '.pth.tar'
        save_model(Q, model_save_path, optimizer, params={'t': t, 'epoch': epoch,
                                                          'stats': {key: statistics[key][-1] for key in
                                                                    list(statistics.keys())}})
        most_wins = wins
        min_error_rate_wins = statistics['ERROR_RATE'][-1]

    if statistics['MINIMAX_4'][-1][0] > most_wins_minimax4 or \
            statistics['MINIMAX_4'][-1][0] == most_wins_minimax4 and wins > most_wins_with_minimax4 or \
            statistics['MINIMAX_4'][-1][0] == most_wins_minimax4 and wins == most_wins_with_minimax4 and \
            statistics['ERROR_RATE'][-1] < min_error_rate_minimax4:
        model_save_filename = save_path + 'model_minimax4'
        model_save_path = model_save_filename + '.pth.tar'
        save_model(Q, model_save_path, optimizer, params={'t': t, 'epoch': epoch,
                                                          'stats': {key: statistics[key][-1] for key in
                                                                    list(statistics.keys())}})
        most_wins_minimax4 = statistics['MINIMAX_4'][-1][0]
        most_wins_with_minimax4 = wins
        min_error_rate_minimax4 = statistics['ERROR_RATE'][-1]

    wins_mask = statistics['MINIMAX_1_MASK'][-1][0] + statistics['MINIMAX_2_MASK'][-1][0] + \
                statistics['MINIMAX_4_MASK'][-1][0]  # + statistics['MINIMAX_6_MASK'][-1][0]
    if wins_mask > most_wins_mask or \
            wins_mask == most_wins_mask and statistics['ERROR_RATE'][-1] < min_error_rate_wins_mask:
        model_save_filename = save_path + 'model_max_wins_{}_mask'.format(wins_mask)
        model_save_path = model_save_filename + '.pth.tar'
        save_model(Q, model_save_path, optimizer, params={'t': t, 'epoch': epoch,
                                                          'stats': {key: statistics[key][-1] for key in
                                                                    list(statistics.keys())}})
        most_wins_mask = wins_mask
        min_error_rate_wins_mask = statistics['ERROR_RATE'][-1]

    if statistics['MINIMAX_4_MASK'][-1][0] > most_wins_minimax4_mask or \
            statistics['MINIMAX_4_MASK'][-1][
                0] == most_wins_minimax4_mask and wins_mask > most_wins_with_minimax4_mask or \
            statistics['MINIMAX_4_MASK'][-1][
                0] == most_wins_minimax4_mask and wins_mask == most_wins_with_minimax4_mask and statistics['ERROR_RATE'][
        -1] < min_error_rate_minimax4_mask:
        model_save_filename = save_path + 'model_minimax4_mask'
        model_save_path = model_save_filename + '.pth.tar'
        save_model(Q, model_save_path, optimizer, params={'t': t, 'epoch': epoch,
                                                          'stats': {key: statistics[key][-1] for key in
                                                                    list(statistics.keys())}})
        most_wins_minimax4_mask = statistics['MINIMAX_4_MASK'][-1][0]
        most_wins_with_minimax4_mask = wins_mask
        min_error_rate_minimax4_mask = statistics['ERROR_RATE'][-1]

    print('BEST: ER {} # WINS {} # WINS_MASK {}'.format(min_error_rate, most_wins, most_wins_mask))

    tm = time.localtime()
    dt = (time.perf_counter() - t0_time) / 60
    eta = round(dt * (learning_ends - t) / (t - t0))
    eta_min = eta % 60
    eta_hour = (eta // 60) % 24
    eta_day = (eta // 60) // 24
    print('TIME: {:02d}:{:02d}'.format(tm.tm_hour, tm.tm_min), end='')
    print(' ### ETA: {}::{}:{}'.format(eta_day, eta_hour, eta_min))
    t0_time = time.perf_counter()
    t0 = t

    return [min_error_rate, \
           most_wins, min_error_rate_wins, \
           most_wins_minimax4, most_wins_with_minimax4, min_error_rate_minimax4, \
           most_wins_mask, min_error_rate_wins_mask, \
           most_wins_minimax4_mask, most_wins_with_minimax4_mask, min_error_rate_minimax4_mask], \
           t0, t0_time


def play_game(side_game, model, depth, player=None, mask_bad=False):
    if player is None:
        player = random.randrange(2)
    obs, _ = side_game.reset(player)
    done = False

    bad_move = 0
    if mask_bad:
        reward_pre_mask = 0
        turn_pre_mask = 0

    while not done:
        if side_game.player == 0:
            input_to_dqn = torch.from_numpy(obs.transpose(2,0,1)).type(dtype).unsqueeze(0)
            action = model(input_to_dqn).data.cpu().numpy()
            if mask_bad:
                valid_mask = _valid_action(side_game.state)
                if not bad_move and not valid_mask[action.argmax(axis=1)]:
                    bad_move=1
                    reward_pre_mask = side_game.bad_move_penalty
                    turn_pre_mask = side_game.turn

                action[0, valid_mask == 0] = np.float('-inf')
            action = action.argmax(axis=1)
            obs, reward, done, _ = side_game.step(action)

            if reward < 0:
                bad_move = 1
            if done:
                if not mask_bad:
                    return reward, bad_move, side_game.turn
                else:
                    reward_pre_mask = reward_pre_mask if bad_move else reward
                    turn_pre_mask = turn_pre_mask if bad_move else side_game.turn
                    return reward_pre_mask, bad_move, turn_pre_mask, reward, side_game.turn
        else:
            state = side_game.swap_state(player=1)
            action, _, _ = alphabeta(state, depth, float('inf'))

            obs, reward, done, _ = side_game.step(action, player=1)

            if done:
                if not mask_bad:
                    return -reward, 0, side_game.turn
                else:
                    reward_pre_mask = reward_pre_mask if bad_move else -reward
                    turn_pre_mask = turn_pre_mask if bad_move else side_game.turn
                    return reward_pre_mask, bad_move, turn_pre_mask, -reward, side_game.turn
