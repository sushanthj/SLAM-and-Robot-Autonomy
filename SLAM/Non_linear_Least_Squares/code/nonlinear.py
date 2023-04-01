'''
Initially written by Ming Hsiao in MATLAB
Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import argparse
import matplotlib.pyplot as plt
import ipdb
from solvers import *
from utils import *


def warp2pi(angle_rad):
    """
    Warps an angle in [-pi, pi]. Used in the update step.
    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad = angle_rad - 2 * np.pi * np.floor(
        (angle_rad + np.pi) / (2 * np.pi))
    return angle_rad


def init_states(odoms, observations, n_poses, n_landmarks):
    '''
    Initialize the state vector given odometry and observations.
    '''
    traj = np.zeros((n_poses, 2))
    landmarks = np.zeros((n_landmarks, 2))
    landmarks_mask = np.zeros((n_landmarks), dtype=bool)

    for i in range(len(odoms)):
        traj[i + 1, :] = traj[i, :] + odoms[i, :]

    for i in range(len(observations)):
        pose_idx = int(observations[i, 0])
        landmark_idx = int(observations[i, 1])

        if not landmarks_mask[landmark_idx]:
            landmarks_mask[landmark_idx] = True

            pose = traj[pose_idx, :]
            theta, d = observations[i, 2:]

            landmarks[landmark_idx, 0] = pose[0] + d * np.cos(theta)
            landmarks[landmark_idx, 1] = pose[1] + d * np.sin(theta)

    return traj, landmarks


def odometry_estimation(x, i):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from (odometry between pose i and i+1)
    \return odom Odometry (\Delta x, \Delta y) in the shape (2, )
    '''
    # TODO: return odometry estimation
    odom = np.zeros((2, ))

    odom[0] = x[2*(i+1)] - x[2*i]
    odom[1] = x[2*(i+1)+1] - x[(2*i)+1]
    # ipdb.set_trace()

    return odom


def bearing_range_estimation(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return obs Observation from pose i to landmark j (theta, d) in the shape (2, )
    '''
    # TODO: return bearing range estimations
    obs = np.zeros((2, ))

    # given the robot pose and landmark location, get the bearing estimate (see theory)
    y_dist = x[(2*n_poses)+(2*j)+1] - x[(2*i)+1]
    x_dist = x[(2*n_poses)+(2*j)] - x[(2*i)]
    obs[0] = warp2pi(np.arctan2(y_dist, x_dist))
    obs[1] = np.sqrt(x_dist**2 + y_dist**2)

    return obs


def compute_meas_obs_jacobian(x, i, j, n_poses):
    '''
    \param x State vector containing both the pose and landmarks
    \param i Index of the pose to start from
    \param j Index of the landmark to be measured
    \param n_poses Number of poses
    \return jacobian Derived Jacobian matrix in the shape (2, 4)
    '''
    # TODO: return jacobian matrix
    jacobian = np.zeros((2, 4))

    y_dist = x[(2*n_poses)+(2*j)+1] - x[(2*i)+1]
    x_dist = x[(2*n_poses)+(2*j)] - x[(2*i)]

    sensor_range = np.sqrt(x_dist**2 + y_dist**2)

    jacobian[0,0] = y_dist/(sensor_range**2)
    jacobian[0,1] = -x_dist/(sensor_range**2)
    jacobian[0,2] = -y_dist/(sensor_range**2)
    jacobian[0,3] = x_dist/(sensor_range**2)

    jacobian[1,0] = -x_dist/sensor_range
    jacobian[1,1] = -y_dist/sensor_range
    jacobian[1,2] = x_dist/sensor_range
    jacobian[1,3] = y_dist/sensor_range

    return jacobian


def create_linear_system(x, odoms, observations, sigma_odom, sigma_observation,
                         n_poses, n_landmarks):
    '''
    \param x State vector x at which we linearize the system.
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

    sqrt_inv_odom = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
    sqrt_inv_obs = np.linalg.inv(scipy.linalg.sqrtm(sigma_observation))

    # TODO: First fill in the prior to anchor the 1st pose at (0, 0)
    # The prior is just a reference frame, it also has some uncertainty, but no measurement
    # Hence the measurement function which estimates the prior is just a identity function
    # i.e h_p(r_t) = r_t. Since no measurements exist, the b matrix will have only zeros (already the case)

    # Here we also define the uncertainty in prior is same as odom uncertainty
    A[0:2, 0:2] = sqrt_inv_odom @ np.eye(2)

    # no need to update b (already zeros)

    H_odom = np.array([[-1,0,1,0], [0,-1,0,1]], dtype=np.float32)
    A_fill_odom = sqrt_inv_odom @ H_odom

    # TODO: Then fill in odometry measurements
    for i in range(n_odom):
        # declare an offset for i to include the prior term (which only occurs once along rows)
        j = i+1

        A[2*j : 2*j+2, 2*i : 2*i+4] = A_fill_odom
        b[2*j : 2*j + 2] = sqrt_inv_odom @ (odom[i] - odometry_estimation(x,i))

    # TODO: Then fill in landmark measurements
    for i in range(n_obs):
        p_idx = int(observations[i,0])
        l_idx = int(observations[i,1])
        Al = sqrt_inv_obs @ compute_meas_obs_jacobian(x, p_idx, l_idx, n_poses)

        # offset again to account for prior
        j = n_odom+1+i
        A[2*j:2*j+2, 2*p_idx:2*p_idx+2] = Al[0:2,0:2]
        A[2*j:2*j+2,2*(n_poses+l_idx):2*(n_poses+l_idx)+2] = Al[0:2,2:4]

        b[2*j:2*j+2] = sqrt_inv_obs @ warp2pi(observations[i,2:4] - bearing_range_estimation(x,p_idx,l_idx,n_poses))

    return csr_matrix(A), b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='?', default='../data/2d_nonlinear.npz')
    parser.add_argument(
        '--method',
        nargs='+',
        choices=['default', 'pinv', 'qr', 'lu', 'qr_colamd', 'lu_colamd'],
        default=['default'],
        help='method')

    args = parser.parse_args()

    data = np.load(args.data)

    # Plot gt trajectory and landmarks for a sanity check.
    gt_traj = data['gt_traj']
    gt_landmarks = data['gt_landmarks']
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-')
    plt.scatter(gt_landmarks[:, 0], gt_landmarks[:, 1], c='b', marker='+')
    plt.show()

    n_poses = len(gt_traj)
    n_landmarks = len(gt_landmarks)

    odom = data['odom']
    observations = data['observations']
    sigma_odom = data['sigma_odom']
    sigma_landmark = data['sigma_landmark']

    # Initialize: non-linear optimization requires a good init.
    for method in args.method:
        print(f'Applying {method}')
        traj, landmarks = init_states(odom, observations, n_poses, n_landmarks)
        print('Before optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)

        # Iterative optimization
        x = vectorize_state(traj, landmarks)
        for i in range(10):
            A, b = create_linear_system(x, odom, observations, sigma_odom,
                                        sigma_landmark, n_poses, n_landmarks)
            dx, _ = solve(A, b, method)
            x = x + dx
        traj, landmarks = devectorize_state(x, n_poses)
        print('After optimization')
        plot_traj_and_landmarks(traj, landmarks, gt_traj, gt_landmarks)
