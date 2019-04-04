import sys
import numpy as np

from utils_game import Game, _valid_action, _get_action, _swap_state, _sim_step

BOARD_W = 7
BOARD_H = 6
SEARCH_DEPTH = 2

PLAYER = 1
OPPONENT = -1
WIN = 1
LOSE = -1

# Method that runs the minimax algorithm and returns
# the move and score of each call.
#
def minimax(state, depth):
    action_arr = []
    for i in range(BOARD_W):
        if _valid_action(state, i):
            action_arr.append(i)

    if depth == 0 or len(action_arr) == 0:
        score = evaluateScore(state)
        return None, score

    best_score = None
    best_action = None

    for a in action_arr:
        _, reward, done, (new_state, _) = _sim_step(state, a)
        if done:
            score = reward
        else:
            # Recursive minimax call, with reduced depth
            _, score = minimax(_swap_state(new_state), depth - 1)
            score = -score

        if best_score is None or score > best_score:
            best_score = score
            best_action = a

    return best_action, best_score


# Method that runs the alphabeta algorithm and returns
# the move and score of each call.
#
def alphabeta(state, depth, beta, debug=False, txt=''):
    action_arr = []
    for i in [3, 2, 4, 1, 5, 0, 6]:  # Prioritize center for efficiency
        if _valid_action(state, i):
            action_arr.append(i)

    if depth == 0 or len(action_arr) == 0:
        score = evaluateScore(state)
        if debug:
            print('{} -> '' - score {} [alpha {} : beta {}]' .format(txt, score, float('-inf'), beta))
        return None, score, txt

    best_score = None
    best_action = None
    best_txt = txt
    alpha = float('-inf')

    for a in action_arr:
        _, reward, done, (new_state, _) = _sim_step(state, a)
        if done:
            score = reward
            new_txt = txt + str(a+1)
        else:
            # Recursive minimax call, with reduced depth
            _, score, new_txt = alphabeta(_swap_state(new_state), depth - 1, -alpha, debug, txt + str(a+1))
            score = -score

        if score > alpha:
            alpha = score

        if debug:
            print('{} -> {} - score {} [alpha {} : beta {}]' .format(txt, new_txt, score, alpha, beta))

        if best_score is None or score > best_score:
            best_score = score
            best_action = a
            best_txt = new_txt

        if best_score > beta:
            return best_action, best_score, best_txt

    if debug:
        print('{} -> {} best_score {} [alpha {} : beta {}]' .format(txt, best_txt, best_score, alpha, beta))

    return best_action, best_score, best_txt


# Method that calculates the heuristic value of a given
# board state. The heuristic adds a point to a player
# for each empty slot that could grant a player victory.
#
def evaluateScore(state):
    score_arr = []
    lose_flag1 = False
    lose_flag2 = False

    # Accumulate score over
    for col in range(BOARD_H):
        if _valid_action(state, col):
            row, _ = _get_action(state, col)
            score = scoreOfCoordinate(state, row, col)
            # If winner move return WIN
            if score[0] == WIN:
                return WIN
            # If two loser moves return LOSE
            if score[1] == WIN:
                if lose_flag1:
                    lose_flag2 = True
                lose_flag1 = True

            score_arr.append(score[0] - score[1])

    if lose_flag2:
        return LOSE

    # If no clear WIN or LOSE return average of values
    return sum(score_arr) / BOARD_W



def scoreOfCoordinate(state, row, col, debug=False):

    score = [0,0]
    offsets = np.array(range(4))

    for player in range(2):
        opponent = (player + 1) % 2
        player_state = state[:, :, player].copy()
        opponent_state = state[:, :, opponent].copy()

        # Check column
        if debug:
            print('Column')
        for shift in range(4):
            if debug:
                print([row + offsets - shift, col])
            if all(row + offsets - shift >= 0) and all(row + offsets - shift < BOARD_H):
                if not any(opponent_state[row + offsets - shift, col] > 0):
                    score[player] = max(score[player], sum(player_state[row + offsets - shift, col]) / 3)

        # Check row
        if debug:
            print('Row')
        for shift in range(4):
            if debug:
                print([row, col + offsets - shift])
            if all(col + offsets - shift >= 0) and all(col + offsets - shift < BOARD_W):
                if not any(opponent_state[row, col + offsets - shift] > 0):
                    score[player] = max(score[player], sum(player_state[row, col + offsets - shift]) / 3)

        # Check rising diagonal
        if debug:
            print('Diag 1')
        offsets = np.array(range(4))
        for shift in range(4):
            if debug:
                print([row + offsets - shift, col + offsets - shift])
            if all(col + offsets - shift >= 0) and all(col + offsets - shift < BOARD_W) and \
                    all(row + offsets - shift >= 0) and all(row + offsets - shift < BOARD_H):
                if not any(opponent_state[row + offsets - shift, col + offsets - shift] > 0):
                    score[player] = max(score[player], sum(player_state[row + offsets - shift, col + offsets - shift]) / 3)

        # Check decreasing diagonal
        if debug:
            print('Diag 2')
        offsets = np.array(range(4))
        for shift in range(4):
            if debug:
                print([row - offsets + shift, col + offsets - shift])
            if all(col + offsets - shift >= 0) and all(col + offsets - shift < BOARD_W) and \
                    all(row - offsets + shift >= 0) and all(row - offsets + shift < BOARD_H):
                if not any(opponent_state[row - offsets + shift, col + offsets - shift] > 0):
                    score[player] = max(score[player], sum(player_state[row - offsets + shift, col + offsets - shift]) / 3)

        return score


# Method that executes the first call of the minimax method and
# returns the move to be executed by the computer. It also verifies
# if any immediate wins or loses are present.
#
def bestMove(state, debug=False):
    move, _, _ = alphabeta(_swap_state(state), SEARCH_DEPTH, float('inf'), debug, '')
    return move


#
# Function that prints the game board, representing the player
# as a O and the computer as an X
#
def printBoard(state):
    for i in range(1, BOARD_W + 1):
        sys.stdout.write(" %d " % i)

    print("")
    print("_" * (BOARD_W * 3))
    for i in reversed(range(BOARD_H)):
        for j in range(BOARD_W):
            if state[i,j,0]:
                sys.stdout.write("|X|")
            elif state[i,j,1]:
                sys.stdout.write("|O|")
            else:
                sys.stdout.write("|-|")
        print("")

    print("_" * (BOARD_W * 3))
    print("")


# Method that provides the main flow of the game, prompting the user
# to make moves, and then allowing the computer to execute a move.
# After each turn, the method checks if the board is full or if a player
# has won.
#
def playGame():
    game = Game()

    moveHeights = [0] * BOARD_W
    player = PLAYER
    opponent = OPPONENT
    winner = 0
    gameOver = False
    remainingColumns = BOARD_W

    print("=========================")
    print("= WELCOME TO CONNECT 4! =")
    print("=========================\n")
    game.reset()
    printBoard(game.state)

    while True:

        while True:
            try:
                move = int(input("What is your move? (Choose from 1 to {})" .format(BOARD_W)))
            except ValueError:
                print("That wasn't a number! Try again.")
                continue
            if move < 1 or move > BOARD_W:
                print("That is not a valid move. Try again.")
            elif not _valid_action(game.state, move-1):
                print("The chosen column is already full. Try again.")
            else:
                break

        _, reward, done, _ = game.step(move-1)
        printBoard(game.state)

        if done and not reward:
            gameOver = True
        if gameOver:
            break

        if reward == WIN:
            winner = PLAYER
            break
        elif reward == LOSE:
            winner = OPPONENT
            break

        print("Now it's the computer's turn!")
        move = bestMove(game.state)
        if move == None:
            break
        _, reward, done, _ = game.step(move, player=1)
        printBoard(game.state)

        if done and not reward:
            gameOver = True
        if gameOver:
            break

        if reward == WIN:
            winner = OPPONENT
            break
        elif reward == LOSE:
            winner = PLAYER
            break

    return winner


#
# Main execution of the game. Plays the game until the user
# wishes to stop.
#

if __name__ == "__main__":
    playing = True
    while playing:
        winner = playGame()
        if winner == OPPONENT:
            print("Damn! You lost!")
        elif winner == PLAYER:
            print("Congratulations! You won!")
        else:
            print("The board is full. This is a draw!")

        while True:
            try:
                option = input("Do you want to play again? (Y/N)")
            except ValueError:
                print("Please input a correct value. Try again.")
                continue
            if option == 'Y' or option == 'y':
                break
            elif option == 'N' or option == 'n':
                playing = False
                break
            else:
                print("Please enter Y or N.")