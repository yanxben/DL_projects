import time

import random
import numpy as np

# CONSTANTS
BOARD_H, BOARD_W = [6, 7]


class Game:
    def __init__(self):
        self.BOARD_W = 7
        self.BOARD_H = 6
        self.state = None
        self.players = 2
        self.player = 0
        self.turn = 0
        self.done = 0
        self.bad_move_penalty = -1.01

    def reset(self, player=0): # TODO: allow random player start
        self.state = np.zeros([self.BOARD_H, self.BOARD_W, 3])
        self.state[0, :, 2] = 1
        self.player = player
        self.turn = 0
        self.done = 0
        return self.state[:,:,:2], self.player

    def step(self, col, player=None):
        if player is not None:
            assert player == self.player, 'Error, bad expected player step'

        if not _valid_action(self.state, col):
            self.done = 1
            reward = self.bad_move_penalty
        else:
            action = _get_action(self.state, col)
            self.state[action[0], action[1], self.player] = 1
            self.state[action[0], action[1], 2] = 0
            if action[0] < self.BOARD_H-1:
                self.state[action[0] + 1, action[1], 2] = 1

            self.turn += 1
            self.player = (self.player + 1) % 2
            self.done = _game_end(self.state, action)
            reward = 1 if self.done else 0
            if self.turn == self.BOARD_W * self.BOARD_H:
                self.done = 1

        return self.state[:,:,:2], reward, self.done, self.player

    def sim_step(self, col, old_state=None, old_turn=None, player=None):
        if old_state is None:
            assert player is None and old_turn is None, \
                'Error. Sim state - if old_state is not passed then player and old_turn are ignored'
            new_state = self.state.copy()
            player = self.player
            new_turn = self.turn = 1
        else:
            assert player is not None and old_turn is None, \
                'Error. Sim state - if old_state is passed then player and old_turn must be passes'
            new_state = old_state.copy()
            new_turn = old_turn + 1

        if not _valid_action(new_state, col):
            done = 1
            reward = self.bad_move_penalty
        else:
            action = _get_action(self.state, col)
            new_state[action[0], action[1], player] = 1
            new_state[action[0], action[1], 2] = 0
            if action[0] < self.BOARD_H-1:
                new_state[action[0] + 1, action[1], 2] = 1

            player = (player + 1) % 2
            done = _game_end(new_state, action)
            reward = 1 if done else 0
            if new_turn == self.BOARD_W * self.BOARD_H:
                done = 1

        return new_state[:,:,:2], reward, done, new_turn, player

    def rand_state(self, max_stage=None, final=False):
        if not max_stage:
            max_stage = random.choice(range(self.BOARD_W*self.BOARD_H))

        self.new_game()

        stage = 0
        find_final = False
        found_final = False
        while stage < self.BOARD_W * self.BOARD_H:
            if stage >= max_stage and not find_final:
                if not final:
                    break
                else:
                    find_final = True

            stage += 1

            action_list = []
            for col in range(self.BOARD_W):
                if not _valid_action(self.state, col):
                    continue
                action = _get_action(self.state, col)
                new_state, _ = self.sim_step(action)
                if _game_end(new_state, action):
                    if find_final:
                        found_final = True
                        break
                else:
                    action_list.append(action)

            if found_final:
                break
            if len(action_list) == 0:
                break
            else:
                chosen_action = random.choice(action_list)
                self.step(chosen_action)

        return self.state, self.player, stage

    def swap_state(self, player):
        return _swap_state(self.state)


def _swap_state(state):
    swapped_state = state.copy()
    swapped_state[:, :, :2] = swapped_state[:, :, 1::-1]
    return swapped_state


def _valid_action(state, col=None):
    if col is None:
        return np.max(state[:, :, 2], axis=0)
    return np.max(state[:, col, 2])


def _get_action(state, col):
    row = np.argmax(state[:, col, 2])
    return (row, col)


def _sim_step(old_state, col, player=0):
    new_state = old_state.copy()

    if not _valid_action(new_state, col):
        done = 1
        reward = -1
    else:
        action = _get_action(new_state, col)
        new_state[action[0], action[1], player] = 1
        new_state[action[0], action[1], 2] = 0
        if action[0] < BOARD_H-1:
            new_state[action[0] + 1, action[1], 2] = 1

        player = (player + 1) % 2
        done = _game_end(new_state, action)
        reward = 1 if done else 0
        if np.max(new_state[:,:,2]) == 0:
            done = 1

    return new_state[:,:,:2], reward, done, (new_state, player)


def _obs2state(obs):
    state = np.zeros([BOARD_H, BOARD_W, 3])
    state[:,:,:2] = obs.copy()
    merged_obs = obs[:,:,0] + obs[:,:,1]
    action_idx = merged_obs.argmin(axis=1)
    for col in range(BOARD_W):
        row = action_idx[col]
        if merged_obs[row, col] == 0:
            state[row, col, 2] = 1

    return state


def _game_end(state, last_action=None):
    if last_action is not None:
        player = np.argmax(state[last_action[0], last_action[1], 0:2])
        player_state = state[:, :, player].copy()
        # Check column
        offsets = np.array(range(4))
        if last_action[0] >= 3:
            if all(player_state[last_action[0] - offsets, last_action[1]] > 0):
                return True
        # Check row
        offsets = np.array(range(4))
        for shift in range(4):
            if all(last_action[1] + offsets - shift >= 0) and all(last_action[1] + offsets - shift < BOARD_W):
                if all(player_state[last_action[0], last_action[1] + offsets - shift] > 0):
                    return True
        # Check rising diagonal
        offsets = np.array(range(4))
        for shift in range(4):
            if all(last_action[1] + offsets - shift >= 0) and all(last_action[1] + offsets - shift < BOARD_W) and \
                    all(last_action[0] + offsets - shift >= 0) and all(last_action[0] + offsets - shift < BOARD_H):
                if all(player_state[last_action[0] + offsets - shift, last_action[1] + offsets - shift] > 0):
                    return True
        # Check decreasing diagonal
        offsets = np.array(range(4))
        for shift in range(4):
            if all(last_action[1] + offsets - shift >= 0) and all(last_action[1] + offsets - shift < BOARD_W) and \
                    all(last_action[0] - offsets + shift >= 0) and all(last_action[0] - offsets + shift < BOARD_H):
                if all(player_state[last_action[0] - offsets + shift, last_action[1] + offsets - shift] > 0):
                    return True
        return False
    else:
        for i in range(BOARD_W):
            row_i = np.argmax(state[:, i, 2]) - 1
            if row_i >= 0:
                action = (row_i, i)
                if _game_end(state, action):
                    return True
        return False


def _check_final_step(state):
    win_actions = [False] * BOARD_W
    lose_actions = [False] * BOARD_W
    for col in range(BOARD_W):
        if _valid_action(state, col):
            (row, _) = _get_action(state, col)
            for player in range(2):
                new_state = state.copy()
                new_state[row, col, player] = 1
                new_state[row, col, 2] = 0
                if row+1 < BOARD_H:
                    new_state[row+1, col, 2] = 1
                    end = _game_end(new_state, (row, col))
                    if player == 0:
                        win_actions[col] = end
                    else:
                        lose_actions[col] = end

    return win_actions, lose_actions


if __name__ == "__main__":

    from utils_plot import plot_state
    import matplotlib.pyplot as plt
    plt.interactive(False)

    game = Game()

    plt.figure()
    N = 4
    for i in range(N):
        for j in range(N):
            plt.subplot(N, N, i*N + j + 1)
            state, player, stage = game.rand_state()
            win_actions, lose_actions = _check_final_step(state)
            plot_state(state, stage, win_actions, lose_actions)

    plt.show()
