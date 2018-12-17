import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.patches as patches


def plot_state(state, title=None, win_actions=None, lose_actions=None):
    if len(state.shape) == 4:
        state = np.squeeze(state, 0)

    # Plot image
    plt.imshow(state[::-1,:,:])

    # Plot grid
    xcoords = [0.5, 1.5, 2.5,  3.5, 4.5, 5.5, 6.5]
    ycoords = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    for xc in xcoords:
        plt.axvline(x=xc)
    for yc in ycoords:
        plt.axhline(y=yc)

    if title:
        plt.title(title)

        if lose_actions and win_actions:
            for col in range(len(lose_actions)):
                if lose_actions[col]:
                    plt.plot([col+0.1], [5 - np.argmax(state[:, col, 2])], color='r', marker='.')

                if win_actions[col]:
                    plt.plot([col-0.1], [5 - np.argmax(state[:, col, 2])], color='g', marker='.')

                if win_actions[col] or not any(win_actions) and lose_actions[col]:
                    plt.plot([col], [5 - np.argmax(state[:, col, 2])], color='y', marker='.')
