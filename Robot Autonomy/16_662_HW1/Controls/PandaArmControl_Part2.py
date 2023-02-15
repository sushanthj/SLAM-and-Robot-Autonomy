import mujoco as mj
from mujoco import viewer
import numpy as np
import math
import quaternion


# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque_sine_board.xml"

################################# Control Callback Definitions #############################

def get_board_control(model, data):

    # Instantite a handle to the desired body on the robot
    omega = 0.4
    data.ctrl[7] = 0.15*np.sin(omega*data.time) + 0.15

# Control callback for gravity compensation
def gravity_comp(model, data):

    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data):

    # Copy the code that you used to implement the controller from Part1 here
    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    end_effector = data.body("hand")

    # Get the Jacobian for the desired location on the robot (The end-effector)

    # id is just getting the ID of the body handle which we created above to use in downstream functions
    id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")
    """
    Here it's 8 DOF because they've added another actuator in the xml file

    If we have 8 DOF, then the jacobian of rotations jacr would be 3x8
    This is because it would map the 3 rotations (roll,pitch,yaw) to the 8 DOF, therefore 3x8
    Same logic is applied to get jacobian of positions jacp of size 3x8
    """
    # init a dummy jacobian matrix of the right size (model.nv gives the DOF required)
    jacp = np.zeros((3,model.nv))
    jacr = np.zeros((3,model.nv))
    print(jacr.shape)

    # update the jacobians using the below function (note id can = 7 if we didn't have a handle)
    # id = 7 works because 7 = the end effector whose jacobian we are interested in
    mj.mj_jacBody(model, data, jacp, jacr, id)

    jacp = jacp[:,0:-1]
    jacr = jacr[:,0:-1]

    # # get the final jacobian by combining jacp and jacr
    final_jacobian = np.concatenate((jacp, jacr), axis=0) # will be 6xmodel.nv shape
    print(final_jacobian.shape)

    # # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # # variables

    # # Specify the desired force as a wrench which is a 6x1 vector
    wrench_des = np.array([15,0,0,0,0,0])

    # # Compute the required control input using desired force values (using Jacobian transpose method)
    required_ctrl_inp = np.matmul(final_jacobian.T, wrench_des)

    orientation_curr = end_effector.xquat
    pos_curr = end_effector.xpos

    # Set the control inputs (modify data similar to position control)
    data.ctrl[:7] = data.qfrc_bias[:7] + required_ctrl_inp

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

    # Update control inputs to the whiteboard
    get_board_control(model, data)

# Control callback for an impedance controller
def impedance_control(model, data):
    """
    Impedence Control Basics:

    1. Impedecne controller are interaction controller
    2. This means it's not suitable for position control but can only exert forces
    3. Therefore the only output we command is how much force our robot will exert
    4. However, here we construct our wrench differently as = stiffness*pos_error + damping*vel_error
    5. Impedence control allows for compliant movement, where motion is not restrictive in one
       direction but can be made restrictive in other direction. We can also try this out
       ourselves by making Kp a diagonal matrix and making the gains high for all other
       directions other than x-axis and then double clicking the robot in simulator to move it
    """

    # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    """
    Here pos_des, vel_des, and orientation_des is all w.r.t to the end effector

    Doing stuff like data.qvel or data.qpos only gives joint velocities or joint angles
    in Generalized coordinates (which is not at all cartesian)

    The overall position will be 1x6 vector and so will velocity (positional + angular)
    The position is a 1x6 vector because it contains [x,y,z,roll,pitch,yaw]

    However we don't care about angular aspect of position in this question
    """

    # Instantite a handle to the desired body on the robot
    end_effector = data.body("hand")
    # id is just getting the ID of the body handle which we created above to use in downstream functions
    id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "hand")

    # Set the desired position (we'll set it to a 1x3 matrix now,
    #                           but we'll combine with rotations (zero vector here) later to make it a 1x6)
    # Also, we are appending 1 to the desired position because even after we reach the board
    # i.e. steady_state, we will have to
    pos_des = np.array([0.595 + 1, 0, 0.595])

    # Set the desired velocities (positional vel only therfore 3x1)
    vel_des = np.zeros((3,1))

    # Set the desired orientation (Use numpy quaternion manipulation functions)
    # convert desired_(roll,pitch,yaw) to desired_quaternion
    orientation_des = np.array([-0.5,0.5,-0.5,0.5])

    # Get the current orientation, position and vel of end effector
    orientation_curr = end_effector.xquat
    pos_curr = end_effector.xpos
    vel_curr = np.zeros((6,1))
    mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, id, vel_curr, True)
    # for impedence control we need only the first 3 elements
    vel_curr = vel_curr[0:3,:]
    print(vel_curr.shape)

    # Get orientation error (Not using this for impedence control)
    # orientation_error = orientation_des - orientation_curr
    # print("orientation error in quaternion", orientation_error)
    # orientation_error = quaternion.from_float_array(orientation_error)
    # orientation_error_axis_angle = quaternion.as_rotation_vector(orientation_error)
    # print("orientation error in axis angle", orientation_error_axis_angle)
    # orientation_error_blank = np.array([0,0,0])

    # Get the errors
    # since vel_des is zero, and current vel will be high due to error in position
    # this vel_error will be negative and will act as a damper on the whole system
    vel_error = vel_des - vel_curr
    # expanding vel_error to 6x1
    vel_error_final = np.zeros((6,1))
    vel_error_final[0:3,0] = vel_error.squeeze()
    pos_error = pos_des - pos_curr
    # expanding pos_error to 6x1
    pos_error_final = np.zeros((6,1))
    pos_error_final[0:3,0] = pos_error.squeeze()

    print("pos error final shape", pos_error_final.shape)
    print("vel error final shape", vel_error_final.shape)

    # Compute the impedance control input torques
    stiffness = 15
    damping = 20

    # Get the Jacobian at the desired location on the robot
    # init a dummy jacobian matrix of the right size (model.nv gives the DOF required)
    jacp = np.zeros((3,model.nv))
    jacr = np.zeros((3,model.nv))

    # update the jacobians using the below function (note id can = 7 if we didn't have a handle)
    # id = 7 works because 7 = the end effector whose jacobian we are interested in
    mj.mj_jacBody(model, data, jacp, jacr, id)

    jacp = jacp[:,0:-1]
    jacr = jacr[:,0:-1]

    # get the final jacobian by combining jacp and jacr
    final_jacobian = np.concatenate((jacp, jacr)) # will be 6xmodel.nv shape

    # Set the control inputs (specific to impedence control)
    required_ctrl_inp = np.squeeze(final_jacobian.T @ (damping*vel_error_final + stiffness*pos_error_final))

    data.ctrl[:7] = data.qfrc_bias[:7] + required_ctrl_inp

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

    # Update control inputs to the whiteboard
    get_board_control(model, data)

def position_control(model, data):

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    desired_joint_positions = np.array([0,0,0,-1.57079,0,1.57079,-0.7853])

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0,0,0,0,0,0,0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp*(desired_joint_positions-data.qpos[:7]) + Kd*(np.array([0,0,0,0,0,0,0])-data.qvel[:7])


####################################### MAIN #####################################


if __name__ == "__main__":

    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    mj.set_mjcb_control(force_control) #TODO:

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(force_sensor_max_time/model.opt.timestep))

    # Launch the simulate viewer
    viewer.launch(model, data)

    # Save recorded force and time points as a csv file
    force = np.reshape(force, (5000, 1))
    time = np.reshape(time, (5000, 1))
    plot = np.concatenate((time, force), axis=1)
    np.savetxt('force_vs_time.csv', plot, delimiter=',')