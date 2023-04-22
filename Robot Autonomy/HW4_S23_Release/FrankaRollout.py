import mujoco as mj
import numpy as np
from mujoco import viewer
from mujoco.glfw import glfw
import time
import copy as cp
from plotter import plot_rewards
from scipy import optimize
import ipdb
np.random.seed(100)


class FrankaSim():
    """This class runs a forward sim using the given policy and estimates the reward.
    It learns the parameters for a normal distribution over the initial and final joint
    positions and initial and final velocities for the selected joints.
    """

    def __init__(self, path_to_mjcf, list_of_joints):

        # Load model and data for the mujoco setup
        self.model = mj.MjModel.from_xml_path(path_to_mjcf) # Load MJCF model
        self.model.body_mass[-2] = 0.2 # Update block mass, cannot add in MJCF due to MuJoCo bug
        self.data = mj.MjData(self.model) #Create the MuJoCo data object

        # Simulation parameters
        self.trajectory_start_time = 0.0 # Simulation start time
        self.trajectory_final_time = 25.0 # Maximum permitted trajectory time
        self.block_settle_tolerance = 1e-5 # tolerance to detect if the block is still, if yes then the motion has been completed and simulation can terminate early

        self.maximum_simulation_time = 20 # seconds, maximum time the simulation will run if other termination criteria not satisfied
        self.maximum_trajectory_time = 15 # maximum time in seconds for any one joint trajectory durations

        # Initial robot configuration
        self.initial_config = np.array([0, 0.335, 0, -2.56, 0, 2.8, 0.782])
        self.initial_velocities = np.array([0,0,0,0,0,0,0])
        self.joint_final_times = np.empty((len(list_of_joints),), dtype=float)
        self.joint_final_times_max = np.empty((4,len(list_of_joints)))
        self.no_of_active_joints = len(list_of_joints)

        # Active joint boolean indexing
        self.active_joints = np.array([True if i in list_of_joints else False for i in range(7)], dtype=bool)

        # Joint Limits
        self.qmin=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.qmax=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Initialize the mean and covariance
        self.policyMu = np.array([0.335,0,1.66,0,-2.56,0,-0.34,0,5,7])
        self.policyCov = np.diag([1 for i in range((4*len(list_of_joints))+2)])

        # Sampled joint position boolean indexing
        self.positions_in_sample = np.array([True if (i%2 ==0 and i<4*len(list_of_joints)) else False for i in range((4*len(list_of_joints))+2)])

        # Policy trajectory coefficients, a 4xno_of_joints matrix, each column has the 4 coefficients for a
        # third order polynomial trajectory for one of the joints, each joint has its own column of coefficients
        self.coeffs = np.empty((4, len(list_of_joints)))


    def check_joint_limits(self, sample):
        """This method takes in a sample from the distribution and clips the sampled joints values to 
        be within the Franka's joint limits"""

        q_min = np.repeat(self.qmin[self.active_joints],2)
        q_max = np.repeat(self.qmax[self.active_joints],2)

        updated_sample = np.empty(sample.shape, dtype=float)
        updated_sample[0:] = sample[0:]

        updated_sample[self.positions_in_sample] = np.clip(updated_sample[self.positions_in_sample], q_min, q_max)

        return updated_sample

    def update_trajectory_parameters_from_policy(self):

        """This function samples a vector from the normal distribution, computes the polynomial trajectory
        and returns the sample that was drawn."""

        # sample the boundary constraints from the policy
        sample = np.random.multivariate_normal(self.policyMu, self.policyCov)

        # clip sample to valid configurations
        updated_sample = self.check_joint_limits(sample)

        # For each active joint, compute the coefficients of the cubic polynomial that defines the trajectory
        for k in range(np.count_nonzero(self.active_joints)):
            q_i =  updated_sample[k*4]    # Initial joint angle
            dq_i = updated_sample[k*4+1]    # Initial joint velocity

            q_f = updated_sample[k*4+2]      # Final joint angle
            dq_f = updated_sample[k*4+3]     # Final joint velocity

            t_i = self.trajectory_start_time     # Initial time
            t_f = updated_sample[k + -1*(np.count_nonzero(self.active_joints))] if (updated_sample[k + -1*(np.count_nonzero(self.active_joints))] < self.maximum_trajectory_time) else self.maximum_trajectory_time     # Final time
            t_f = -1*t_f if t_f < 0 else t_f

            self.joint_final_times[k] = t_f
            self.joint_final_times_max[:,k] = np.array([t_f**3, t_f**2, t_f**1, 1])


            # compute the trajectory coefficients for current joint
            b = np.array([q_i, dq_i, q_f, dq_f])

            A = np.array([[t_i**3, t_i**2, t_i, 1],
                        [3*t_i**2, 2*t_i, 1, 0],
                        [t_f**3, t_f**2, t_f, 1],
                        [3*t_f**2, 2*t_f, 1, 0]])

            self.coeffs[:,k] = np.matmul(np.linalg.inv(A), b)

        return updated_sample


    def sample_policy(self):
        """Placeholder method to correspond to the pseudocode"""

        sample = self.update_trajectory_parameters_from_policy()
        return sample

    def get_desired_joint_vals(self, time):

        """This method uses the cubic polynomial coefficients and the passed in time to compute
        the desired joint positions and velocities for this instant. This method is called by
        the controller"""

        t_vec = np.array([time**3, time**2, time**1, 1])
        t_mat = np.repeat(np.expand_dims(t_vec, axis=1), self.no_of_active_joints, axis=1)
        t_mat = np.clip(t_mat,np.zeros(self.joint_final_times_max.shape),self.joint_final_times_max)

        vel_coeffs = np.array([3,2,1])
        vel_coeffs = np.tile(np.expand_dims(vel_coeffs, axis=1), [1,self.no_of_active_joints])
        desired_joint_positions = self.initial_config
        desired_joint_velocities = self.initial_velocities
        desired_joint_positions[self.active_joints] = np.sum(np.multiply(t_mat, self.coeffs), axis=0)
        desired_joint_velocities[self.active_joints] = np.sum(np.multiply(np.multiply(t_mat[1:, :], vel_coeffs), self.coeffs[0:3,:]), axis=0)


        return desired_joint_positions, desired_joint_velocities


    def compute_control_signal(self, model, data, time):

        """This is the control callback that is provided to MuJoCo's simulator during each
        policy rolloout. It gets the desired joint positions and velocities and uses a PID
        control to track the motion."""

        # Get the desired joint positions and velocities for this time instant
        desired_positions, desired_velocities = self.get_desired_joint_vals(time)

        # Instantite a handle to the desired body on the robot
        body = data.body("hand")

        # Desired gain on position error (K_p)
        Kp = np.eye(7,7)*1000

        # Desired gain on velocity error (K_d)
        Kd = np.eye(7,7)*100

        # Set the actuator control torques
        return data.qfrc_bias[:7] + Kp@(desired_positions-data.qpos[:7]) + Kd@(desired_velocities-data.qvel[:7])


    def control_callback(self, model, data):

        """This is the control callback that is provided to MuJoCo's simulator during each
        policy rolloout. It gets the desired joint positions and velocities and uses a PID
        control to track the motion."""

        # Get the current simulator time from MuJoCo
        time = data.time

        # Set the actuator control torques
        data.ctrl[:7] = self.compute_control_signal(model, data, time)

    def get_reward(self, block_x_position):
        """This method computes the reward based on the blocks final position"""
        return 1.0 - abs(0.9-block_x_position)


    def policy_rollout(self):
        """This method runs the simulator to execute the sampled motion and returns
        the reward."""

        # Get the initial state of the arm based on policy and set keyframe configuration
        desired_initial_pos, desired_initial_vels = self.get_desired_joint_vals(0)
        # print(desired_initial_pos.shape)
        # print(desired_initial_vels)
        self.model.key_time[0] = 0
        self.model.key_qpos[0,:7] = desired_initial_pos
        self.model.key_qvel[0,:7] = desired_initial_vels
        self.model.key_qpos[0,7:9] = np.array([0.025, 0.025])

        # Setting the initial position for the arm and block pushing task
        mj.mj_resetDataKeyframe(self.model, self.data, 0)   # Propogate key through joints and read gripper center for block placement
        mj.mj_forward(self.model, self.data)
        center_of_gripper_site = self.data.site("block_site")
        finger = self.data.body("hand")
        self.model.key_qpos[0,0:9] = np.array([0, 0.335, 0, -2.56, 0, 2.8, 0.782, 0, 0])
        self.model.key_qpos[0,9:12] = np.array([0.5, 0, 0.025])

        rot_quat = np.empty((4,))
        mj.mju_mat2Quat(rot_quat, center_of_gripper_site.xmat)
        self.model.key_qpos[0,12:16] = np.array([1, 0, 0, 0])
        self.model.key_ctrl[0,7] = 0

        # Reset the simulation state to the starting keyframe
        mj.mj_resetDataKeyframe(self.model, self.data, 0)

        # Set the control callback
        mj.set_mjcb_control(self.control_callback)

        # Framerate
        framerate = 60

        # Instantiate a handle to the block
        block = self.data.body("block")
        prev_block_pos = block.xpos

        frames = []

        max_width = 200
        max_height = 100

        scene = mj.MjvScene(self.model, maxgeom=10000)

        cam = mj.MjvCamera()    # create the camera view
        opt = mj.MjvOption()    # set the visualization options

        # Init GLFW, create window, make OpenGL context current
        glfw.init()
        window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # Set the initial camera position
        cam.azimuth = 90 ; cam.elevation = -45 ; cam.distance =  5
        cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])

        context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)


        while not glfw.window_should_close(window):
            time_prev = self.data.time
            prev_block_pos = block.xpos.copy()
            while (self.data.time - time_prev < 1.0/60.0):
                mj.mj_step(self.model, self.data)


            if ((self.data.time >= 3)) and (np.linalg.norm(prev_block_pos - block.xpos) < self.block_settle_tolerance):
                print("Time exceeded without block motion!")
                break

            if ((self.data.time >= self.maximum_simulation_time)):
                print("Maximum simulation time exceeded!")
                break


            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
            mj.mjv_updateScene(self.model, self.data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)

        glfw.terminate()
        print(self.get_reward(block.xpos[0]))

        return self.get_reward(block.xpos[0])


    def extract_top_K_samples(self, K, sample_params, rewards_list):
        # COOL PYTHON STUFF
        """
        We have two lists, sample_params and rewards_list. We need to sort sample_params
        based on rewards list. To do so, we zip it, sort it, and get extract only the params
        """
        # when sorting, ascending is default and we need descending
        top_K_samples = sorted(list(zip(sample_params, rewards_list)), key=lambda x:x[1], reverse=True)
        top_K_params = [np.expand_dims(x[0], axis=0) for x in top_K_samples[0:K]]
        return top_K_params

    ########################################################################
    ################################# CEM ##################################
    ########################################################################

    def CEM(self):
        """This method implements the CEM algorithm and plots the average reward 
        after all iterations."""

        #initialize policy
        self.policyMu = np.array([0.335,0,1.66,0,-2.56,0,-0.34,0,5,5])
        self.policyCov = np.diag([1 for i in range((4*self.no_of_active_joints)+2)])

        # TODO:

        num_updates = 5
        num_samples = 15
        rewards_to_plot = []

        for i in range(num_updates):
            sample_params = []
            rewards_list = []

            for j in range(num_samples):
                param_sample = self.sample_policy()
                one_reward = self.policy_rollout()
                sample_params.append(param_sample)
                rewards_list.append(one_reward)
                rewards_to_plot.append(one_reward)

            top_k_params = self.extract_top_K_samples(5, sample_params, rewards_list)
            # update policy mean and cov
            stacked = np.vstack(top_k_params)
            self.policyMu = np.mean(stacked, axis=0)
            # np.cov takes in data in foll format:
            # variables along axis=0
            # observations along axis=1
            self.policyCov = np.cov(stacked.T)

            print("mean shape", self.policyMu.shape)
            print("cov shape", self.policyCov.shape)

        plot_rewards(rewards=rewards_to_plot)


    ########################################################################
    ################################# REPS #################################
    ########################################################################

    #Function for computing eta parameters
    def computeEta(self,returns,epsilon):
        """This method returns the optimal eta value for the weighting step in REPS"""

        #Make more numerically stable by removing max return
        R = returns - np.max(returns)

        #Define dual function to be optimized
        def dual_function(eta):
            #TODO:
            df = 0 # Correct this line
            return df

        #Perform optimization of dual function
        eta = optimize.minimize(dual_function, 1,bounds=[(0.00000001,10000)]).x
        return eta[0]

    def REPS(self):
        """This method implements the REPS algorithm and plots the average reward
        after all iterations."""

        #initialize policy
        self.policyMu = np.array([0.335,0,1.66,0,-2.56,0,-0.34,0,5,7])
        self.policyCov = np.diag([1 for i in range((4*self.no_of_active_joints)+2)])

        # TODO: