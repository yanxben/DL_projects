import sys, time
import pickle
import numpy as np
from collections import namedtuple, Counter
from itertools import count
import random

import torch

from utils_replay_buffer import ReplayBuffer
from utils_save import save_model, load_model

from dqn_train import explore_step, train_step, evaluation_step

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def DQNLearningN(
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
    save_paths=None,
    symmetry=False,
    error_clip=False,
    grad_clip=False,
    prediction=False,
    action_mask=False
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

    assert load_paths is not None and save_paths is not None and len(load_paths) == len(save_paths)

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
            obs, done, reward = explore_step(envs[n], last_obs[n], Q[n], Q[model_countering[n]], policy_func, replay_buffer[n], t, action_mask)

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
                train_step(Q[n], target_Q[n], optimizer[n], replay_buffer[n], batch_size, gamma, symmetry, error_clip, grad_clip, prediction)

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