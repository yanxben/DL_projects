import random

import gym
import torch
import torch.optim as optim

from model import DQN_FCN
from game_utils import Game
from dqn_train import dqn_learing
from dqn_train import OptimizerSpec
#from utils.gym import get_env, get_wrapper_by_name
#from utils.schedule import LinearSchedule


BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 16
TARGET_UPDATE_FREQ = 10000
LEARNING_RATE = 2.5*1e-4
ALPHA = 0.95
EPS = 0.1

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def train_model(env):

    #def stopping_criterion(env):
    #    # notice that here t is the number of steps of the wrapped env,
    #    # which is different from the number of steps in the underlying env
    #    return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE),
    )

    #exploration = LinearSchedule(1000000, 0.5)

    # Construct an epilson greedy policy with given exploration schedule
    def epilson_greedy_policy(model, obs, t):
        sample = random.random()
        eps_threshold = EPS #exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)
            # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
            #return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
            return model(obs).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(2)]])



    #optimizer_spec,
    #policy_func,
    #exploration,
    #stopping_criterion=None,
    #replay_buffer_size=1000000,
    #batch_size=32,
    #gamma=0.99,
    #learning_starts=100000,
    #learning_freq=4,
    #frame_history_len=4,
    #target_update_freq=10000

    dqn_learing(
        env=env,
        q_func=DQN_FCN,
        optimizer_spec=optimizer_spec,
        policy_func=epilson_greedy_policy,
        stopping_criterion=None,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=1,
        target_update_freq=TARGET_UPDATE_FREQ
    )

if __name__ == '__main__':
    # Get Atari games.
    # Change the index to select a different game.
    #task = benchmark.tasks[3]

    # Run training
    #seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = Game()

    train_model(env)

