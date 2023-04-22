import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, N=10):
    '''
    rewards - values to be plotted (and averaged)
    N (int) - number of adjacent datapoints to be averaged
    '''

    if N > len(rewards):
        N = len(rewards)
    moving_average = np.convolve(rewards, np.ones(N)/N, mode='valid')

    # account for convolution output being smaller than data length
    #adding estimates for first few datapoints
    for i in range(N//2-1):
        old_datum = np.sum(rewards[:i+N//2]) / (i+N//2)
        moving_average = np.insert(moving_average, i, old_datum)

    #adding estimates for last few datapoints
    begin = N//2
    if N%2 == 0:
        begin -= 1
    for i in range(begin, -1, -1):
        old_datum = np.sum(rewards[-i-N//2:]) / (i+N//2)
        moving_average = np.append(moving_average, old_datum)

    plt.plot(rewards, label="Reward")
    plt.plot(moving_average, linestyle='--', label="Moving Average")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
