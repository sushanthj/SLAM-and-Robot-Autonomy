from RobotUtil import *
from Franka import *

def main():
    # Check if two boxes are colliding w.r.t a reference box
    ref_box_rpyxyz = np.array([0,0,0,0,0,0])
    ref_box_dims = np.array([3,1,2])
    # convert roll pitch yaw and translation to a tranformation matrix
    ref_box_pose = rpyxyz2H(ref_box_rpyxyz[0:3], ref_box_rpyxyz[3:6])

    # get user defined test boxes
    test_box_poses, test_box_dims = get_test_boxes()

    # convert BBox poses and dimensions to get corners and axes
    ref_box_corners, ref_box_axes = BlockDesc2Points(ref_box_pose, ref_box_dims)
    test_box_corners = []
    test_box_axes = []

    for i in range(len(test_box_poses)):
        corner, axes = BlockDesc2Points(test_box_poses[i], test_box_dims[i])
        test_box_corners.append(corner)
        test_box_axes.append(axes)

    # Do collision check for each test_box and ref_box
    collision_results = []
    for i in range(len(test_box_poses)):
        collision_results.append(CheckBoxBoxCollision(ref_box_corners, ref_box_axes,
                                                      test_box_corners[i], test_box_axes[i]))

    print(collision_results)

    # Visulaize the bounding boxes for the robot arm's links at home position (all joints at 0)
    Arm = FrankArm()
    Arm.PlotCollisionBlockPoints(ang=[0.,0.,0.,0.,0.,0.,0.])

def get_test_boxes():
    """
    Custom Test Boxes to test collision checks

    Returns:
        test_box_poses : list of transformation matrices for each box
         test_box_dims : list of dimensions for each box
    """
    test_boxes_rpyxyz = np.zeros((8,6))
    test_boxes_dims = []

    #                                  r,p,y            x,y,z
    test_boxes_rpyxyz[0,:] = np.array([0,0,0,           0,1,0])
    test_boxes_rpyxyz[1,:] = np.array([1,0,1.5,         1.5,-1.5,0])
    test_boxes_rpyxyz[2,:] = np.array([0,0,0,           0,0,-1])
    test_boxes_rpyxyz[3,:] = np.array([0,0,0,           3,0,0])
    test_boxes_rpyxyz[4,:] = np.array([0.5,0,0.4,       -1,0,-2])
    test_boxes_rpyxyz[5,:] = np.array([-0.2,0.5,0,      1.8,0.5,1.5])
    test_boxes_rpyxyz[6,:] = np.array([0,0.785,0.785,   0,-1.2,0.4])
    test_boxes_rpyxyz[7,:] = np.array([0,0,0.2,         -0.8,0,-0.5])

    test_boxes_poses = []
    for row in range(test_boxes_rpyxyz.shape[0]):
        test_boxes_poses.append(rpyxyz2H(test_boxes_rpyxyz[row,0:3], test_boxes_rpyxyz[row,3:6]))

    test_boxes_dims = [[0.8,0.8,0.8], [1,3,3], [2,3,1], [3,1,1], [2,0.7,2], [1,3,1],
                       [1,1,1], [1,0.5,0.5]]

    return test_boxes_poses, test_boxes_dims

if __name__ == "__main__":
    main()