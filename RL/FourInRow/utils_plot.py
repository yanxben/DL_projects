import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as patches


def plot_state(state, title=None, win_actions=None, lose_actions=None, action=None):
    if len(state.shape) == 4:
        state = np.squeeze(state, 0)

    # Plot image
    plt.imshow(state[::-1, :, [1,0,2]])

    # Plot grid
    xcoords = [0.5, 1.5, 2.5,  3.5, 4.5, 5.5, 6.5]
    ycoords = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    for xc in xcoords:
        plt.axvline(x=xc)
    for yc in ycoords:
        plt.axhline(y=yc)

    if title:
        plt.title(title)

    if lose_actions is not None and win_actions is not None:
        for col in range(len(lose_actions)):
            if lose_actions[col]:
                plt.plot([col+0.1], [5 - np.argmax(state[:, col, 2])], color='r', marker='o')

            if win_actions[col]:
                plt.plot([col-0.1], [5 - np.argmax(state[:, col, 2])], color='g', marker='o')

            if win_actions[col] or not any(win_actions) and lose_actions[col]:
                plt.plot([col], [5 - np.argmax(state[:, col, 2])], color='y', marker='.')

    if action:
        plt.plot([action], [5 - np.argmax(state[:, action, 2])], color='c', marker='o')


def plot_obs(obs, title=None):
    # Plot image
    state = np.zeros([obs.shape[0], obs.shape[1], 3])
    state[:, :, :2] = obs
    plt.imshow(state[::-1, :, [1, 0, 2]])

    # Plot grid
    xcoords = [0.5, 1.5, 2.5,  3.5, 4.5, 5.5, 6.5]
    ycoords = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    for xc in xcoords:
        plt.axvline(x=xc)
    for yc in ycoords:
        plt.axhline(y=yc)

    if title:
        plt.title(title)


def plot_stats(stats, prefix=''):
    """
    Statistic = {
        'TURNS_RATE': [],
        'ERROR_RATE': [],
        'MINIMAX_1': [],
        'MINIMAX_2': [],
        'MINIMAX_4': [],
        'MINIMAX_6': []
    )
    """
    turns_rate = stats['TURNS_RATE']
    error_rate = stats['ERROR_RATE']
    minimax1_stats = stats['MINIMAX_1']
    minimax2_stats = stats['MINIMAX_2']
    minimax4_stats = stats['MINIMAX_4']
    #minimax6_stats = stats['MINIMAX_6']

    epochs = list(range(1000, (len(turns_rate)+1)*1000, 1000))

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(epochs, turns_rate)
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('TURNS')
    plt.title('AVERAGE TURNS IN SELF-PLAY')

    plt.subplot(1,2,2)
    plt.plot(epochs, error_rate)
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('ERROR RATE')
    plt.title('VALIDATION ERROR')

    plt.savefig(prefix + 'TrainStats.png')
    plt.close()

    # Plot average win-lose rate and number or turns
    minimax1_wins = [minimax1_stats[i][0] for i in range(len(minimax1_stats))]
    minimax1_loss = [minimax1_stats[i][2] for i in range(len(minimax1_stats))]
    minimax1_bads = [minimax1_stats[i][3] for i in range(len(minimax1_stats))]
    minimax1_turn = [minimax1_stats[i][4] for i in range(len(minimax1_stats))]
    minimax1_avg_wins = [sum(minimax1_wins[i:i + 10])/10 for i in range(0, len(minimax1_wins) - 9)]
    minimax1_avg_loss = [sum(minimax1_loss[i:i + 10])/10 for i in range(0, len(minimax1_loss) - 9)]
    minimax1_avg_bads = [sum(minimax1_bads[i:i + 10])/10 for i in range(0, len(minimax1_bads) - 9)]
    minimax1_avg_turn = [sum(minimax1_turn[i:i + 10])/10 for i in range(0, len(minimax1_turn) - 9)]
    minimax2_wins = [minimax2_stats[i][0] for i in range(len(minimax2_stats))]
    minimax2_loss = [minimax2_stats[i][2] for i in range(len(minimax2_stats))]
    minimax2_bads = [minimax2_stats[i][3] for i in range(len(minimax2_stats))]
    minimax2_turn = [minimax2_stats[i][4] for i in range(len(minimax2_stats))]
    minimax2_avg_wins = [sum(minimax2_wins[i:i + 10])/10 for i in range(0, len(minimax2_wins) - 9)]
    minimax2_avg_loss = [sum(minimax2_loss[i:i + 10])/10 for i in range(0, len(minimax2_loss) - 9)]
    minimax2_avg_bads = [sum(minimax2_bads[i:i + 10])/10 for i in range(0, len(minimax2_bads) - 9)]
    minimax2_avg_turn = [sum(minimax2_turn[i:i + 10])/10 for i in range(0, len(minimax2_turn) - 9)]
    minimax4_wins = [minimax4_stats[i][0] for i in range(len(minimax4_stats))]
    minimax4_loss = [minimax4_stats[i][2] for i in range(len(minimax4_stats))]
    minimax4_bads = [minimax4_stats[i][3] for i in range(len(minimax4_stats))]
    minimax4_turn = [minimax4_stats[i][4] for i in range(len(minimax4_stats))]
    minimax4_avg_wins = [sum(minimax4_wins[i:i + 10])/10 for i in range(0, len(minimax4_wins) - 9)]
    minimax4_avg_loss = [sum(minimax4_loss[i:i + 10])/10 for i in range(0, len(minimax4_loss) - 9)]
    minimax4_avg_bads = [sum(minimax4_bads[i:i + 10])/10 for i in range(0, len(minimax4_bads) - 9)]
    minimax4_avg_turn = [sum(minimax4_turn[i:i + 10])/10 for i in range(0, len(minimax4_turn) - 9)]
    #minimax6_wins = [minimax6_stats[i][0] for i in range(len(minimax6_stats))]
    #minimax6_loss = [minimax6_stats[i][2] for i in range(len(minimax6_stats))]
    #minimax6_bads = [minimax6_stats[i][3] for i in range(len(minimax6_stats))]
    #minimax6_turn = [minimax6_stats[i][4] for i in range(len(minimax6_stats))]
    #minimax6_avg_wins = [sum(minimax6_wins[i:i + 10])/10 for i in range(0, len(minimax6_wins) - 9)]
    #minimax6_avg_loss = [sum(minimax6_loss[i:i + 10])/10 for i in range(0, len(minimax6_loss) - 9)]
    #minimax6_avg_bads = [sum(minimax6_bads[i:i + 10])/10 for i in range(0, len(minimax6_bads) - 9)]
    #minimax6_avg_turn = [sum(minimax6_turn[i:i + 10])/10 for i in range(0, len(minimax6_turn) - 9)]

    plt.figure()
    plt.subplot(2,2,1)
    plt.scatter(epochs[9:], minimax1_avg_wins, marker='.')
    plt.scatter(epochs[9:], minimax2_avg_wins, marker='.')
    plt.scatter(epochs[9:], minimax4_avg_wins, marker='.')
    #plt.scatter(epochs[9:], minimax6_avg_wins, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.title('AVERAGE WINS AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    plt.subplot(2,2,2)
    plt.scatter(epochs[9:], minimax1_avg_loss, marker='.')
    plt.scatter(epochs[9:], minimax2_avg_loss, marker='.')
    plt.scatter(epochs[9:], minimax4_avg_loss, marker='.')
    #plt.scatter(epochs[9:], minimax6_avg_loss, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.title('AVERAGE LOSSES AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    plt.subplot(2, 2, 3)
    plt.scatter(epochs[9:], minimax1_avg_bads, marker='.')
    plt.scatter(epochs[9:], minimax2_avg_bads, marker='.')
    plt.scatter(epochs[9:], minimax4_avg_bads, marker='.')
    #plt.scatter(epochs[9:], minimax6_avg_bads, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.title('AVERAGE BAD MOVES AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    plt.subplot(2,2,4)
    plt.scatter(epochs[9:], minimax1_avg_turn, marker='.')
    plt.scatter(epochs[9:], minimax2_avg_turn, marker='.')
    plt.scatter(epochs[9:], minimax4_avg_turn, marker='.')
    #plt.scatter(epochs[9:], minimax6_avg_turn, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.title('AVERAGE TURNS AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    plt.savefig(prefix + 'MiniMaxStats.png')
    plt.close()

