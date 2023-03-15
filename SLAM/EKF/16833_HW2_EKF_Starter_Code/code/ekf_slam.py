"""
Initially written by Ming Hsiao in MATLAB
Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np
import re
import matplotlib.pyplot as plt
import math
import ipdb
from copy import deepcopy
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
    R = np.array([[float(np.cos(theta)), -float(np.sin(theta))],
                  [float(np.sin(theta)), float(np.cos(theta))]])
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
    3. Covariance of landmark pose estimations (see theory)

    input1 init_measure    :  Initial measurements of form (beta0, l0, beta1,...) (2k,1)
    input2 init_measure_cov:  Initial covariance matrix (2, 2) per landmark given parameters.
    input3 init_pose       :  Initial pose vector of shape (3, 1).
    input4 init_pose_cov   :  Initial pose covariance of shape (3, 3) given parameters.

    return1 k              : Number of landmarks.
    return2 landmarks      : Numpy array of shape (2k, 1) for the state.
    return3 landmarks_cov  : Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2*k, 1))
    landmark_cov = np.zeros((2*k, 2*k))

    x_t = init_pose[0][0]
    y_t = init_pose[1][0]
    theta_t = init_pose[2][0]

    # to find the covaraince of all landmark poses, we need to iterate over each landmark
    # and start filling in landmark_cov array initialized above. (we'll fill in diagonal
    # components only)
    for l_i in range(k):
        # l_i is the i'th landmark
        # init_measure.shape = (2k,1)
        beta = init_measure[l_i*2][0]
        l_range = init_measure[l_i*2 + 1][0]

        # need to find landmark location in global coords (l_x, l_y) to find H_l
        l_x = x_t + l_range*(float(np.cos(beta+theta_t)))
        l_y = y_t + l_range*(float(np.sin(beta+theta_t)))

        landmark[l_i*2][0] = l_x
        landmark[l_i*2+1][0] = l_y

        # Note, L here is the derivative of (l_x,l_y) vector (sensor model) w.r.t beta and theta
        # G_l is the derivative of same sensor model w.r.t state varialbes (x,y,theta)
        L = np.array([[-l_range*float(np.sin(beta+theta_t)), float(np.cos(beta+theta_t))],
                      [l_range*float(np.cos(beta+theta_t)), float(np.sin(beta+theta_t))]])

        # G_l represents the robot pose aspect of landmark measurement
        # therefore when measuring covariance, it will use robot pose covariance
        G_l = np.array([[1, 0, -l_range*float(np.sin(theta_t + beta))],
                        [0, 1, l_range*float(np.cos(theta_t + beta))]])

        # See theory, L below was derived w.r.t to measurement. Therefore,
        # during covariance calculation it will use measurement_covariance
        # Similarly, G defined w.r.t state variables (x,y,theta) therefore uses pose_covariance
        pred_landmark_cov = (G_l @ init_pose_cov @ G_l.T) + (L @ init_measure_cov @ L.T)

        assert(pred_landmark_cov.shape == (2,2))

        landmark_cov[l_i*2:l_i*2+2, l_i*2:l_i*2+2] = pred_landmark_cov

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance shape (3, 3) in the (x, y, theta) space.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    # TODO: Predict new position (mean) using control inputs (only geometrical, no cov here)
    theta_curr = X[2][0]

    d_t = control[0][0] # control input in robot's local frame's x-axis
    alpha_t = control[1][0]

    P_pred = deepcopy(P)
    pos_cov = deepcopy(P[0:3,0:3])

    X_pred = np.zeros(shape=X.shape)
    X_pred[0][0] += d_t*np.cos(theta_curr)
    X_pred[1][0] += d_t*np.sin(theta_curr)
    X_pred[2][0] += alpha_t

    X_pred = X_pred + X

    # TODO: Predict new uncertainity (covariance) using motion model noise, find G_t and R_t
    # NOTE: G_t needs to be mulitplied with P viz of shape (3 + 2k, 3+ 2k), because it has
    # pose and measurement cov. IN THIS STEP OF PREDICTION WE ONLY UPDATE POSE COV
    # Therefore G_t and R_t can be 3x3 (3 variables in state vector)

    G_t = np.array([[1, 0,  -d_t*float(np.sin(theta_curr))],
                    [0, 1,   d_t*float(np.cos(theta_curr))],
                    [0, 0,                1               ]])

    rotation_matrix_z = np.array([[float(np.cos(theta_curr)), -float(np.sin(theta_curr)), 0],
                                  [float(np.sin(theta_curr)),  float(np.cos(theta_curr)), 0],
                                  [           0,                            0,            1]])

    pose_pred_cov = (G_t @ pos_cov @ G_t.T) + \
                    (rotation_matrix_z @ control_cov @ rotation_matrix_z.T)

    # update just the new predicted covariance in robot pose, measurement pose is left untouched
    P_pred[0:3,0:3] = pose_pred_cov

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

    Since we have a measurement, we will have to update both pose and measure covariances, i.e.
    the entire P_pre will be updated.

    Here we use the H_p and H_l described in the theory section. H_l and H_p will be combined
    to form H_t (the term in the EKF Algorithm in Probablistic Robotics). This H_t term
    will be defined for each landmark and stored in a massive matrix

    Q viz measurement covariance will need to be added to the H_t of each landmark, therefore
    it too will also be stored in a huge diagonal matrix
    '''
    # Q needs to be added to (Ht @ P_pre @ (Ht.T)) = (2*k, 2*k), therefore must be same shape
    Q = np.zeros(shape=(2*k, 2*k))

    # stack all predicted measurements into one large vector
    z_t = np.zeros(shape=(2*k, 1))

    # H_t as discussed above will be a large diagonal matrix where we'll stack H_p and H_l
    # side-by-side horizontally (making H_t 2x5 for each landmark). This will then be stacked
    # vertically, but again as a diagonal matrix.
    # H_t.T will also be multiplied with P_pre (3+2k, 3+2k). Therefore this needs to
    # also have 3+2k columns therefore the other column should be 2k rows since
    # (H_p concat with H_l) = 2x5
    H_t = np.zeros(shape=(2*k, 3+(2*k)))

    # iterate through every measurement, assuming every measurement captures every landmark
    num_measurements = k
    for i in range(num_measurements):
        # since we have a predicted pose already X_pre[0:3] we'll use that as our
        # linearization point

        # define the predicted pose of robot and landmark in global frame
        pos_x = X_pre[0][0] # robot pose_x in global frame
        pos_y = X_pre[1][0] # robot pose_y in global frame
        pos_theta = X_pre[2][0] #  bearing in global frame
        l_x = X_pre[3+i*2][0] # landmark i in global frame
        l_y = X_pre[4+i*2][0] # landmark i in global frame

        # convert predicted poses to local frame
        l_x_offset = l_x - pos_x
        l_y_offset = l_y - pos_y

        # use predicted pose of robot and landmark to get predicted measurements
        i_bearing = warp2pi(np.arctan2(l_y_offset, l_x_offset) - pos_theta) # bearing of i-th l
        i_range = math.sqrt(l_x_offset**2 + l_y_offset**2) # range of i-th landmark
        z_t[2*i][0] = i_bearing
        z_t[2*i+1][0] = i_range

        # Jacobian of measurement function (h(β,r) in theory) w.r.t pose (x,y,theta)
        # Note here we define h(β,r), whereas in theory it is h(r,β), hence rows are interchanged
        H_p = np.array([[(l_y_offset/(i_range**2)), (-l_x_offset/(i_range**2)), -1],
                        [(-l_x_offset/i_range)    , (-l_y_offset/i_range),       0],],
                        dtype=np.float64)

        # Note here we define h(β,r), whereas in theory it is h(r,β)
        H_l = np.array([[(-l_y_offset/(i_range**2)), (l_x_offset/(i_range**2))],
                        [(l_x_offset/i_range)      , (l_y_offset/i_range)     ]])

        # H_t[2*i : 2*i+2, 2*i : 2*i+3] = H_p
        H_t[2*i : 2*i+2, 0:3] = H_p
        H_t[2*i : 2*i+2, 3+2*i : 5+2*i] = H_l

        Q[i*2:i*2+2, i*2:i*2+2] = measure_cov

    # Now after obtaining H_t and Q_t, find Kalman gain K
    K = P_pre @ H_t.T @ np.linalg.inv((H_t @ P_pre @ H_t.T) + Q)

    # Update pose(mean) and noise(covariance) using K #! SHOULD I SUM THE DIFFERENCES IN MEAS?
    X_updated = np.zeros(shape=X_pre.shape)
    X_updated = X_pre + K @ (measure - z_t) # (measure - z_t) = (actual - prediction)

    P_updated = (np.eye(2*k+3) - (K @ H_t)) @ P_pre

    return X_updated, P_updated


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance
       given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)

    e_dist = np.zeros(shape=l_true.shape, dtype=float)
    m_dist = np.zeros(shape=l_true.shape, dtype=float)

    for i in range(k):
        euclidean_dist = 0

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
    1. landmarks [β_1 r_1 β_2 r_2 · · · ], where β_i , r_i correspond to landmark
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

            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
