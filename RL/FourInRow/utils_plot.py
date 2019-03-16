import pylab
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as patches


def plot_state(state, title=None, win_actions=None, lose_actions=None, action=None, stats=None, stats_symm=None):
    plt.cla()
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
        plt.plot([action], [5 - ((np.argmax(state[:, action, 2]) - 1) % 6)], color='c', marker='o')

    if stats is not None:
        for col in range(len(stats[0])):
            plt.text(col - 0.2, 5 - 0.2, '{:.2f}'.format(stats[0][col]), fontsize=12, color='w')
            plt.text(col - 0.2, 5 + 0.2, '{:.2f}'.format(stats[1][col]), fontsize=12, color='w')
    if stats_symm is not None:
        for col in range(len(stats[0])):
                plt.text(col - 0.2, 4 - 0.2, '{:.2f}'.format(stats_symm[0][col]), fontsize=12, color='w')
                plt.text(col - 0.2, 4 + 0.2, '{:.2f}'.format(stats_symm[1][col]), fontsize=12, color='w')


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


def plot_stats(stats, path='./', prefix='', delta=10, stride=1, log_freq=1000):
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

    epochs = list(range(log_freq, (len(turns_rate)+1)*log_freq, log_freq))

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

    pylab.get_current_fig_manager().window.showMaximized()
    plt.pause(0.01)
    plt.savefig(path + prefix + 'TrainStats.png')
    plt.close()

    # Plot average win-lose rate and number or turns
    #delta=20
    #stride=2
    minimax1_wins = [minimax1_stats[i][0] for i in range(len(minimax1_stats))]
    minimax1_loss = [minimax1_stats[i][2] for i in range(len(minimax1_stats))]
    minimax1_bads = [minimax1_stats[i][3] for i in range(len(minimax1_stats))]
    minimax1_turn = [minimax1_stats[i][4] for i in range(len(minimax1_stats))]
    minimax1_avg_wins = [sum(minimax1_wins[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax1_wins) - stride*delta+1), stride)]
    minimax1_avg_loss = [sum(minimax1_loss[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax1_loss) - stride*delta+1), stride)]
    minimax1_avg_bads = [sum(minimax1_bads[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax1_bads) - stride*delta+1), stride)]
    minimax1_avg_turn = [sum(minimax1_turn[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax1_turn) - stride*delta+1), stride)]
    minimax2_wins = [minimax2_stats[i][0] for i in range(len(minimax2_stats))]
    minimax2_loss = [minimax2_stats[i][2] for i in range(len(minimax2_stats))]
    minimax2_bads = [minimax2_stats[i][3] for i in range(len(minimax2_stats))]
    minimax2_turn = [minimax2_stats[i][4] for i in range(len(minimax2_stats))]
    minimax2_avg_wins = [sum(minimax2_wins[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax2_wins) - stride*delta+1), stride)]
    minimax2_avg_loss = [sum(minimax2_loss[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax2_loss) - stride*delta+1), stride)]
    minimax2_avg_bads = [sum(minimax2_bads[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax2_bads) - stride*delta+1), stride)]
    minimax2_avg_turn = [sum(minimax2_turn[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax2_turn) - stride*delta+1), stride)]
    minimax4_wins = [minimax4_stats[i][0] for i in range(len(minimax4_stats))]
    minimax4_loss = [minimax4_stats[i][2] for i in range(len(minimax4_stats))]
    minimax4_bads = [minimax4_stats[i][3] for i in range(len(minimax4_stats))]
    minimax4_turn = [minimax4_stats[i][4] for i in range(len(minimax4_stats))]
    minimax4_avg_wins = [sum(minimax4_wins[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax4_wins) - stride*delta+1), stride)]
    minimax4_avg_loss = [sum(minimax4_loss[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax4_loss) - stride*delta+1), stride)]
    minimax4_avg_bads = [sum(minimax4_bads[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax4_bads) - stride*delta+1), stride)]
    minimax4_avg_turn = [sum(minimax4_turn[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax4_turn) - stride*delta+1), stride)]
    #minimax6_wins = [minimax6_stats[i][0] for i in range(len(minimax6_stats))]
    #minimax6_loss = [minimax6_stats[i][2] for i in range(len(minimax6_stats))]
    #minimax6_bads = [minimax6_stats[i][3] for i in range(len(minimax6_stats))]
    #minimax6_turn = [minimax6_stats[i][4] for i in range(len(minimax6_stats))]
    #minimax6_avg_wins = [sum(minimax6_wins[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax6_wins)/stride - delta+1))]
    #minimax6_avg_loss = [sum(minimax6_loss[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax6_loss)/stride - delta+1))]
    #minimax6_avg_bads = [sum(minimax6_bads[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax6_bads)/stride - delta+1))]
    #minimax6_avg_turn = [sum(minimax6_turn[i:i + stride*delta:stride])/delta for i in range(0, int(len(minimax6_turn)/stride - delta+1))]

    plt.figure()
    plt.subplot(2,2,1)
    #plt.scatter(epochs[delta-1::stride], minimax1_avg_wins, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax2_avg_wins, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax4_avg_wins, marker='.')
    plt.plot(epochs[stride*delta - 1::stride], minimax1_avg_wins)
    plt.plot(epochs[stride*delta - 1::stride], minimax2_avg_wins)
    plt.plot(epochs[stride*delta - 1::stride], minimax4_avg_wins)
    #plt.scatter(epochs[delta-1::stride], minimax6_avg_wins, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.ylim((0, 2))
    plt.title('AVERAGE WINS AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    plt.subplot(2,2,2)
    #plt.scatter(epochs[delta-1::stride], minimax1_avg_loss, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax2_avg_loss, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax4_avg_loss, marker='.')
    plt.plot(epochs[stride*delta - 1::stride], minimax1_avg_loss)
    plt.plot(epochs[stride*delta - 1::stride], minimax2_avg_loss)
    plt.plot(epochs[stride*delta - 1::stride], minimax4_avg_loss)
    #plt.scatter(epochs[delta-1::stride], minimax6_avg_loss, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.ylim((0, 2))
    plt.title('AVERAGE LOSSES AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    plt.subplot(2, 2, 3)
    #plt.scatter(epochs[delta-1::stride], minimax1_avg_bads, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax2_avg_bads, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax4_avg_bads, marker='.')
    plt.plot(epochs[stride*delta - 1::stride], minimax1_avg_bads)
    plt.plot(epochs[stride*delta - 1::stride], minimax2_avg_bads)
    plt.plot(epochs[stride*delta - 1::stride], minimax4_avg_bads)
    #plt.scatter(epochs[delta-1::stride], minimax6_avg_bads, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.ylim((0, 2))
    plt.title('AVERAGE BAD MOVES AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    plt.subplot(2,2,4)
    #plt.scatter(epochs[delta-1::stride], minimax1_avg_turn, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax2_avg_turn, marker='.')
    #plt.scatter(epochs[delta-1::stride], minimax4_avg_turn, marker='.')
    plt.plot(epochs[stride*delta - 1::stride], minimax1_avg_turn)
    plt.plot(epochs[stride*delta - 1::stride], minimax2_avg_turn)
    plt.plot(epochs[stride*delta - 1::stride], minimax4_avg_turn)
    #plt.scatter(epochs[delta-1::stride], minimax6_avg_turn, marker='.')
    plt.grid()
    plt.xlabel('EPOCHS')
    plt.ylabel('RATE')
    plt.ylim((0, 42))
    plt.title('AVERAGE TURNS AGAINST MINIMAX')
    plt.legend(['MINIMAX 1', 'MINIMAX 2', 'MINIMAX 4'])  #, 'MINIMAX 6'])

    pylab.get_current_fig_manager().window.showMaximized()
    plt.pause(0.01)
    plt.savefig(path + prefix + 'MiniMaxStats.png')
    plt.close()


if __name__ == "__main__":
    import pickle

    with open('./checkpoints/model_5_01_lr_5e6_symmetry_good/Statistics.pkl', 'rb') as f:
        stats = pickle.load(f)

    plot_stats(stats, path='./plots/', prefix='model_5_01_lr_5e6_symmetry_good_', delta=10, stride=1)
