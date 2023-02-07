'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import random


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
        X_bar_resampled =  np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        size = X_bar.shape[0]
        X_bar_resampled = []
        # r = random.randrange(0, (1/size)*10000)
        # r = r/10000
        r = np.random.uniform(0, (1/size), 1)
        c = X_bar[0][3]
        i = 1
        for m in range(1,size+1):
            u = r + (m-1)*(1/size)
            while u > c:
                i += 1
                c += X_bar[i-1][3]
            X_bar_resampled.append(np.expand_dims((X_bar[i-1]),axis=0))

        X_bar_resampled_2 = np.vstack(X_bar_resampled)
        # import ipdb
        # ipdb.set_trace()

        return X_bar_resampled_2
