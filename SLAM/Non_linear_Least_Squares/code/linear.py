'''
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import time
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import argparse
import ipdb
import matplotlib.pyplot as plt
from solvers import *
from utils import *


def create_linear_system(odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param odoms Odometry measurements between i and i+1 in the global coordinate system. Shape: (n_odom, 2).
    \param observations Landmark measurements between pose i and landmark j in the global coordinate system. Shape: (n_obs, 4).
    \param sigma_odom Shared covariance matrix of odometry measurements. Shape: (2, 2).
    \param sigma_observation Shared covariance matrix of landmark measurements. Shape: (2, 2).

    \return A (M, N) Jacobian matrix.
    \return b (M, ) Residual vector.
    where M = (n_odom + 1) * 2 + n_obs * 2, total rows of measurements.
          N = n_poses * 2 + n_landmarks * 2, length of the state vector.
    '''

    n_odom = len(odoms)
    n_obs = len(observations)

    M = (n_odom + 1) * 2 + n_obs * 2
    N = n_poses * 2 + n_landmarks * 2

    A = np.zeros((M, N))
    b = np.zeros((M, ))

    # Prepare Sigma^{-1/2}.
    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    # The prior is just a reference frame, it also has some uncertainty, but no measurement
    # Hence the measurement function which estimates the prior is just a identity function
    # i.e h_p(r_t) = r_t. Since no measurements exist, the b matrix will have only zeros (already the case)

    # Here we also define the uncertainty in prior is same as odom uncertainty
    A[0:2, 0:2] = sqrt_inv_odom @ np.eye(2)

    # no need to update b (already zeros)

    # TODO: Then fill in odometry measurements
    """
    The A matrix structure is shown in the theory section. Along the rows, it has:
        - predicted prior (of size 1)
        - predicted odom measurements (of size n_odom)
        - predicted landmark measurements (of size n_obs)

    We will also follow the same order
    """

    H_odom = np.array([[-1,0,1,0], [0,-1,0,1]], dtype=np.float32)
    H_land = np.array([[-1,0,1,0], [0,-1,0,1]], dtype=np.float32)

    A_fill_odom = sqrt_inv_odom @ H_odom

    for i in range(n_odom):
        # declare an offset for i to include the prior term (which only occurs once along rows)
        j = i+1

        # A[2*j : 2*j+2 , 2*j : 2*j+4] = sqrt_inv_odom @ H_odom
        A[2*j : 2*j+2, 2*i : 2*i+4] = A_fill_odom
        b[2*j : 2*j + 2] = sqrt_inv_odom @ odoms[i]

    # TODO: Then fill in landmark measurements
    A_fill_land = sqrt_inv_obs @ H_land # H_land like H_odom is also a 2x4 matrix

    for i in range(n_obs):
        # observations = (52566,4) # (pose_index, landmark_index, measurement_x, measurement_y)
        # Therefore we need to check which pose is associated with which landmark
        p_idx = int(observations[i,0])
        l_idx = int(observations[i,1])
        # offset to account for prior (offset only along rows) + all odom measurements above
        j = i + n_odom + 1

        A[2*j : 2*j+2, 2*p_idx : 2*p_idx+2] = A_fill_land[0:2, 0:2]
        A[2*j : 2*j+2, 2*(n_poses + l_idx):2*(n_poses + l_idx)+2] = A_fill_land[0:2, 2:4]
        b[2*j : 2*j+2] = sqrt_inv_obs @ observations[i,2:4]

    return csr_matrix(A), b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='path to npz file', nargs='?',
                        default='/home/sush/CMU/SLAM-and-Robot-Autonomy/SLAM/Non_linear_Least_Squares/data/2d_linear.npz')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help=
        'Number of repeats in evaluation efficiency. Increase to ensure stablity.'
    )
    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']

    # Verify that gt_traj = (1000,2) and gt_landmarks = (100,2) shapes for 2d_linear.npz
    print(gt_traj.shape)
    print(gt_landmarks.shape)

    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', label='gt trajectory')
    plt.scatter(gt_landmarks[:, 0],
                gt_landmarks[:, 1],
                c='b',
                marker='+',
                label='gt landmarks')
    plt.legend()
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odoms = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']
    """
    The shapes of above values for 2d_linear.npz are:
    odoms = (999,2) which makes sense since there are 1000 robot poses
    observations = (52566,4) # (pose_index, landmark_index, measurement_x, measurement_y)
    sigma_odom = (2,2)
    sigma_landmark = (2,2)
    """

    # Build a linear system
    A, b = create_linear_system(odoms, observations, sigma_odom,
                                sigma_landmark, n_poses, n_landmarks)

    # Solve with the selected method
    for method in args.method:
        print(f'Applying {method}')

        total_time = 0
        total_iters = args.repeats
        for i in range(total_iters):
            start = time.time()
            x, R = solve(A, b, method)
            end = time.time()
            total_time += end - start
        print(f'{method} takes {total_time / total_iters}s on average')

        if R is not None:
            plt.spy(R)
            plt.show()

        traj, landmarks = devectorize_state(x, n_poses)

        # Visualize the final result
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
