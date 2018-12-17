import time

import random
import numpy as np

from plot_utils import plot_state


# CONSTANTS
BOARD_H, BOARD_W = [6, 7]


class Game:
    def __init__(self):
        self.BOARD_W = 7
        self.BOARD_H = 6
        self.state = None
        self.player = 0

    def new_game(self):
        self.state = np.zeros([self.BOARD_H, self.BOARD_W, 3])
        self.state[0, :, 2] = 1
        self.player = 0
        return self.state, self.player

    def step(self, action):
        self.state[action[0], action[1], self.player] = 1
        self.state[action[0], action[1], 2] = 0
        if action[0] < self.BOARD_H-1:
            self.state[action[0] + 1, action[1], 2] = 1

        self.player = 1-self.player
        return self.state, self.player

    def sim_step(self, action):
        new_state = self.state.copy()
        new_state[action[0], action[1], self.player] = 1
        new_state[action[0], action[1], 2] = 0
        if action[0] < self.BOARD_H-1:
            new_state[action[0] + 1, action[1], 2] = 1

        return new_state, self.player

    def rand_state(self, max_stage=None):
        if not max_stage:
            max_stage = random.choice(range(self.BOARD_W*self.BOARD_H))

        self.new_game()

        stage = 0
        while stage < max_stage:
            #plt.imshow(self.state)
            #plt.title(stage)
            #plt.show(block=False)
            stage += 1

            action_list = []
            for col in range(self.BOARD_W):
                if not valid_action(self.state, col):
                    continue
                action = get_action(self.state, col)
                new_state, _ = self.sim_step(action)
                if not game_end(new_state, action):
                    action_list.append(action)

            if len(action_list) == 0:
                break
            else:
                chosen_action = random.choice(action_list)
                self.step(chosen_action)

        return self.state, self.player, stage


def valid_action(state, col):
    return np.max(state[:, col, 2])


def get_action(state, col):
    row = np.argmax(state[:, col, 2])
    return (row, col)


def game_end(state, last_action=None, debug=False):
    if debug:
        plt.imshow(state)
        plt.grid()
        plt.show(block=False)
        time.sleep(0.2)

    if last_action is not None:
        player = np.argmax(state[last_action[0], last_action[1], 0:2])
        player_state = state[:, :, player].copy()
        if debug:
            print('last action: {}' .format(last_action))
        # Check column
        if debug:
            print('Column')
        offsets = np.array(range(4))
        if last_action[0] >= 3:
            if debug:
                print([last_action[0] - offsets, last_action[1]])
            if all(player_state[last_action[0] - offsets, last_action[1]] > 0):
                return True
        # Check row
        if debug:
            print('Row')
        offsets = np.array(range(4))
        for shift in range(4):
            if debug:
                print([last_action[0], last_action[1] + offsets - shift])
            if all(last_action[1] + offsets - shift >= 0) and all(last_action[1] + offsets - shift < BOARD_W):
                if all(player_state[last_action[0], last_action[1] + offsets - shift] > 0):
                    return True
        # Check rising diagonal
        #print('Diag 1')
        offsets = np.array(range(4))
        for shift in range(4):
            #print([last_action[0] + offsets - shift, last_action[1] + offsets - shift])
            if all(last_action[1] + offsets - shift >= 0) and all(last_action[1] + offsets - shift < BOARD_W) and \
                    all(last_action[0] + offsets - shift >= 0) and all(last_action[0] + offsets - shift < BOARD_H):
                #print(player_state[last_action[0] + offsets - shift, last_action[1] + offsets - shift])
                if all(player_state[last_action[0] + offsets - shift, last_action[1] + offsets - shift] > 0):
                    #print('END')
                    return True
                #else:
                #    print('PASS')
            #else:
                #print('OUT')
        # Check decreasing diagonal
        #print('Diag 2')
        offsets = np.array(range(4))
        for shift in range(4):
            #print([last_action[0] - offsets + shift, last_action[1] + offsets - shift])
            if all(last_action[1] + offsets - shift >= 0) and all(last_action[1] + offsets - shift < BOARD_W) and \
                    all(last_action[0] - offsets + shift >= 0) and all(last_action[0] - offsets + shift < BOARD_H):
                #print(player_state[last_action[0] - offsets + shift, last_action[1] + offsets - shift])
                if all(player_state[last_action[0] - offsets + shift, last_action[1] + offsets - shift] > 0):
                    #print('END')
                    return True
                #else:
                    #print('PASS')
            #else:
                #print('OUT')
        return False
    else:
        for i in range(BOARD_W):
            row_i = np.argmax(state[:, i, 2]) - 1
            if row_i >= 0:
                action = (row_i, i)
                if game_end(state, action):
                    return True
        return False


def check_final_step(state):
    win_actions = [False] * BOARD_W
    lose_actions = [False] * BOARD_W
    for col in range(BOARD_W):
        if valid_action(state, col):
            (row, _) = get_action(state, col)
            for player in range(2):
                new_state = state.copy()
                new_state[row, col, player] = 1
                new_state[row, col, 2] = 0
                if row+1 < BOARD_H:
                    new_state[row+1, col, 2] = 1
                    end = game_end(new_state, (row, col))
                    if player == 1:
                        win_actions[col] = end
                    else:
                        lose_actions[col] = end

    return win_actions, lose_actions


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    game = Game()
    plt.interactive(False)

    plt.figure()
    N = 4
    for i in range(N):
        for j in range(N):
            plt.subplot(N, N, i*N + j + 1)
            state, player, stage = game.rand_state()
            win_actions, lose_actions = check_final_step(state)
            plot_state(state, stage, win_actions, lose_actions)

    plt.show()
    plt.show()
