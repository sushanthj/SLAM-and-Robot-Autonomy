import numpy as np
import math


def rpyxyz2H(rpy, xyz):
    Ht = [[1, 0, 0, xyz[0]],
          [0, 1, 0, xyz[1]],
          [0, 0, 1, xyz[2]],
          [0, 0, 0, 1]]

    Hx = [[1, 0, 0, 0],
          [0, math.cos(rpy[0]), -math.sin(rpy[0]), 0],
          [0, math.sin(rpy[0]), math.cos(rpy[0]), 0],
          [0, 0, 0, 1]]

    Hy = [[math.cos(rpy[1]), 0, math.sin(rpy[1]), 0],
          [0, 1, 0, 0],
          [-math.sin(rpy[1]), 0, math.cos(rpy[1]), 0],
          [0, 0, 0, 1]]

    Hz = [[math.cos(rpy[2]), -math.sin(rpy[2]), 0, 0],
          [math.sin(rpy[2]), math.cos(rpy[2]), 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]

    H = np.matmul(np.matmul(np.matmul(Ht, Hz), Hy), Hx)

    return H


def R2axisang(R):
    ang = math.acos((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)
    Z = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if Z == 0:
        return [1, 0, 0], 0.
    x = (R[2, 1] - R[1, 2]) / Z
    y = (R[0, 2] - R[2, 0]) / Z
    z = (R[1, 0] - R[0, 1]) / Z

    return [x, y, z], ang


def BlockDesc2Points(H, Dim):
    # takes in an object pose (specified by a homogeneous transformation matrix, H)
    # and object dimensions (length(x)/width(y)/height(z))
    # and returns the bounding box of the object and the axes given by the rotation matrix.

    center = H[0:3, 3]
    axes = [H[0:3, 0], H[0:3, 1], H[0:3, 2]]

    corners = [
        center,
        center + (axes[0] * Dim[0] / 2.) + (axes[1] * Dim[1] / 2.) + (axes[2] * Dim[2] / 2.),
        center + (axes[0] * Dim[0] / 2.) + (axes[1] * Dim[1] / 2.) - (axes[2] * Dim[2] / 2.),
        center + (axes[0] * Dim[0] / 2.) - (axes[1] * Dim[1] / 2.) + (axes[2] * Dim[2] / 2.),
        center + (axes[0] * Dim[0] / 2.) - (axes[1] * Dim[1] / 2.) - (axes[2] * Dim[2] / 2.),
        center - (axes[0] * Dim[0] / 2.) + (axes[1] * Dim[1] / 2.) + (axes[2] * Dim[2] / 2.),
        center - (axes[0] * Dim[0] / 2.) + (axes[1] * Dim[1] / 2.) - (axes[2] * Dim[2] / 2.),
        center - (axes[0] * Dim[0] / 2.) - (axes[1] * Dim[1] / 2.) + (axes[2] * Dim[2] / 2.),
        center - (axes[0] * Dim[0] / 2.) - (axes[1] * Dim[1] / 2.) - (axes[2] * Dim[2] / 2.)
    ]
    # returns corners of BB and axes
    return corners, axes


def CheckPointOverlap(pointsA, pointsB, axis):
    # TODO: check if sets of points projected on axis are overlapping

    # print("Shape of A:", pointsA.shape)
    # print("Shape of B:", pointsB.shape)
    # print("Shape of axis:", axis.shape)

    # print()

    status = False

    pointsA = np.asarray(pointsA)
    pointsB = np.asarray(pointsB)

    # print("Shape of A:", pointsA.shape)
    # print("Shape of B:", pointsB.shape)

    # print("Points A:", pointsA)
    # print("Points B:", pointsB)
    # print("Axis:", axis)

    # print(np.linalg.norm(axis))

    norm_axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) != 0 else axis   # 1x3
    project_axis = np.column_stack(
        (norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis, norm_axis))

    projectionA_scals = np.dot(axis, pointsA.T)   # 1x3 x 3x9

    projectionA = project_axis * projectionA_scals  # 3x9 * 1x9 -> 3x9

    projectionB_scals = np.dot(axis, pointsB.T)  # 1x3 x 3x9

    projectionB = project_axis * projectionB_scals   # 3x9 * 1x9 -> 3x9

    # print("Projected Shape of A:", projectionA.shape)

    # why do we need to compare them in 3 different axis???
    if (np.min(projectionA[0, :]) > np.max(projectionB[0, :]) or np.max(projectionA[0, :]) < np.min(
            projectionB[0, :])) or (
            np.min(projectionA[1, :]) > np.max(projectionB[1, :]) or np.max(projectionA[1, :]) < np.min(
            projectionB[1, :])) or (
            np.min(projectionA[2, :]) > np.max(projectionB[2, :]) or np.max(projectionA[2, :]) < np.min(
            projectionB[2, :])):
        # print("There is a separation")
        # print("Points A:", pointsA)
        # print("Points B:", pointsB)

        # print("Points projected on axis", axis)
        # print("Projected Points A:", projectionA)
        # print("Projected Points B:", projectionB)

        return False

    return True


def CheckBoxBoxCollision(pointsA, axesA, pointsB, axesB):
    # Sphere check

    if np.linalg.norm(pointsA[0] - pointsB[0]) > (
            np.linalg.norm(pointsA[0] - pointsA[1]) + np.linalg.norm(pointsB[0] - pointsB[1])):
        return False

    # TODO - SAT cuboid-cuboid collision check

    # Checking for collision along surface normals

    for i in range(3):
        if not CheckPointOverlap(pointsA, pointsB, axesA[i]):
            return False

    for j in range(3):
        if not CheckPointOverlap(pointsA, pointsB, axesB[j]):
            return False

    # Checking for edge-edge collisions
    for i in range(3):
        for j in range(3):
            if not CheckPointOverlap(pointsA, pointsB, np.cross(axesA[i], axesB[j])):
                return False

    return True


# # Testing Script for Evaluating cuboid-cuboid colilision checker:
# refData = np.array([0, 0, 0, 0, 0, 0, 3, 1, 2])
# H_ref = rpyxyz2H(refData[3:6], refData[0:3])
# corners_ref, axes_ref = BlockDesc2Points(H_ref, refData[6:9])
# # print('axes_ref = ', axes_ref)
#
# testCaseData = np.array([[   0,    1,    0,    0,     0,     0,  0.8,  0.8,  0.8],
#                          [ 1.5, -1.5,    0,    1,     0,   1.5,    1,    3,    3],
#                          [   0,    0,   -1,    0,     0,     0,    2,    3,    1],
#                          [   3,    0,    0,    0,     0,     0,    3,    1,    1],
#                          [  -1,    0,   -2,  0.5,     0,   0.4,    2,  0.7,    2],
#                          [ 1.8,  0.5,  1.5, -0.2,   0.5,     0,    1,    3,    1],
#                          [   0, -1.2,  0.4,    0, 0.785, 0.785,    1,    1,    1],
#                          [-0.8,    0, -0.5,    0,     0,   0.2,    1,  0.5,  0.5]])
# counter = 1
# for arr in testCaseData:
#     xyz = arr[0:3]
#     rpy = arr[3:6]
#     lwh = arr[6:9]
#     H = rpyxyz2H(rpy, xyz)
#     corners, axes = BlockDesc2Points(H, lwh)
#     flag = CheckBoxBoxCollision(corners_ref, axes_ref, corners, axes)
#     print("Test Case ", counter, ", whether collision = ", flag)
#     counter += 1



