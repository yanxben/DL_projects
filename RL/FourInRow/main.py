import gym
import torch.optim as optim

from model import DQN_FCN
from game_utils import Game
from dqn_train import OptimizerSpec, dqn_learing
from utils.gym import get_env, get_wrapper_by_name
from utils.schedule import LinearSchedule


BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 16
TARGET_UPDATE_FREQ = 10000
LEARNING_RATE = 2.5*1e-4
ALPHA = 0.95
EPS = 0.1


def train_model(env):

    #def stopping_criterion(env):
    #    # notice that here t is the number of steps of the wrapped env,
    #    # which is different from the number of steps in the underlying env
    #    return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.5)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':
    # Get Atari games.
    # Change the index to select a different game.
    #task = benchmark.tasks[3]

    # Run training
    #seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = Game()

    train_model(env)

