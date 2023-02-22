'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        # DO NOT TOUCH
        M = len(X_bar)
        weights = X_bar[:, 3]
        weights = weights / weights.sum()
        r = np.random.uniform(0, 1 / len(X_bar))
        c = weights[0]
        i = 0
        X_bar_resampled = []
        for m in range(1, M + 1):
            u = r + (m - 1) * (1 / M)
            while u > c:
                i = i + 1
                if i > len(weights):
                    break
                c = c + weights[i - 1]
            X_bar_resampled.append(X_bar[i - 1])
        X_bar_resampled = np.array(X_bar_resampled)
        return X_bar_resampled

    def random_sampler(self, X_bar):
        int_indices = np.random.choice(list(range(0, len(X_bar))), len(X_bar), list(X_bar[:, 3]))
        X_bar = X_bar[int_indices]
        return X_bar

