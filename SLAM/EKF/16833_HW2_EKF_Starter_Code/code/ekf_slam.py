"""
Initially written by Ming Hsiao in MATLAB
Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import re
import matplotlib.pyplot as plt
import math
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)

def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    param angle_rad = Input angle in radians
    return angle_rad_warped = Warped angle to [-\pi, \pi].
    """
    if angle_rad > np.pi:
        while (angle_rad > np.pi):
            angle_rad -= 2*np.pi

    else:
        while (angle_rad < -np.pi):
            angle_rad += 2*np.pi

    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    Here we find:
    1. Number of landmarks (k)
    2. landmark states (just position (2k,1)) which will get stacked onto
       robot pose
    3. Covariance of landmark pose estimations pg 249. (or pg314) Line 17. (H_t @ Σ̄_t @ H_t.T)

    input1 init_measure    :  Initial measurements of form (beta0, l0, beta1,...) (2k,1)
    input2 init_measure_cov:  Initial covariance matrix of shape (2, 2) per landmark given parameters.
    input3 init_pose       :  Initial pose vector of shape (3, 1).
    input4 init_pose_cov   :  Initial pose covariance of shape (3, 3) given parameters.

    return1 k              : Number of landmarks.
    return2 landmarks      : Numpy array of shape (2k, 1) for the state.
    return3 landmarks_cov  : Numpy array of shape (2k, 2k) for the uncertainty.

    Note. landmark_cov was the 2x2 covariance we found in the theory section H_l
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k)) # H_l from theory section

    # to get H_l, we need [x_t, y_t and theta_t] from robot state vector
    x_t = init_pose[0][0]
    y_t = init_pose[1][0]
    theta_t = init_pose[2][0]

    # to find the covaraince of all landmark poses H_l, we need to iterate over each landmark
    # and start filling in landmark_cov array initialized above. (we'll fill in diagonal
    # components only)
    for l_i in range(k):
        # l_i is the i'th landmark
        # init_measure.shape = (2k,1)
        beta = init_measure[l_i*2][0]
        l_range = init_measure[l_i*2 + 1][0]

        # need to find landmark location in global coords (l_x, l_y) to find H_l
        l_x = x_t + l_range*(np.cos(beta+theta_t))
        l_y = y_t + l_range*(np.sin(beta+theta_t))

        landmark[l_i*2][0] = l_x
        landmark[l_i*2+1][0] = l_y

        # define q which will be used in finding H_l (mentioned in theory section)
        q = (l_x - x_t)**2 + (l_y - y_t)**2
        q_root = math.sqrt(q)

        H_l = np.array([ [((l_x - x_t)/q_root), ((l_y - y_t)/q_root)],
                         [-((l_y - y_t)/q_root), ((l_x - x_t)/q_root)]])
        #NOTE: we are ignoring multiplying H_l * F_x_t (line15 in EKF pg. 257)

        assert(H_l.shape == (2,2))

        landmark_cov[l_i*2:l_i*2+2, l_i*2:l_i*2+2] = H_l @ init_measure_cov @ H_l.T

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    

    return X_pred, P_pred


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''

    return X_pre, P_pre


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    plt.waitforbuttonpress(0)


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    """
    The data file is extracted as arr and is given to init_landmarks below

    The data file has 2 types of entries:
    1. landmarks [β 1 r 1 β 2 r 2 · · · ], where β_i , r_i correspond to landmark
    2. control inputs in form of [d, alpha] (d = translation along x-axis)
    """
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    # pose vector is initialized to zero
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    """
    measure = all landmarks
    measure_cov = known sensor covariance
    pose = initialized to (0,0)
    pose_cov = how much we trust motion model = fixed
    """
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov) # basically H_t in for-loop of pg 204

    # Setup state vector X by stacking pose and landmark states
    # X = [x_t, y_t, thetha_t, landmark1(range), landmark1(bearing), landmark2(range)...]
    X = np.vstack((pose, landmark))

    # Setup covariance matrix P by expanding pose and landmark covariances
    """
    - The covariance matrix for a state vector = [x,y,thetha] would be 3x3
    - However, since we also add landmarks into the state vector, we need to add that as well
    - Since there are 2*k landmarks, we create a new matrix encapsulating pose_cov and landmark_cov

    - this new cov matrix (constructed by np.block) is:

        [[pose_cov,        0     ],
         [    0,     landmark_cov]]

    """
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
