#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from a5_utils import normalize_points
from UZ_utils_a5 import *


def draw_epiline(sp: gs.GridSpec, l,h,w):
    # l: line equation (vector of size 3)
    # h: image height
    # w: image width
    print(l)

    x0, y0 = map(int, [0, -l[2]/l[1]])
    x1, y1 = map(int, [w-1, -(l[2]+l[0]*w)/l[1]])

    plt.plot([x0,x1],[y0,y1],'r')

    plt.ylim([0,h])
    plt.gca().invert_yaxis()


# 1b
def calc_disparity():
    f = 2.5
    T = 120
    d = [f * T / p_z for p_z in range(1, 1000)]

    plt.plot(range(1, 1000), d)
    plt.xlabel("p_z [mm]")
    plt.ylabel("disparity [mm]")
    plt.show()


# 2b
# pairs: at least 8 pairs of points from 2 images
def fundamental_matrix(pairs):
    # normalize points
    print("before normalization:")
    print(pairs)
    left_pts, T_1 = normalize_points(pairs[:, 0:2])
    right_pts, T_2 = normalize_points(pairs[:, 2:4])
    print("after normalization:")

    pairs = np.hstack((left_pts, right_pts))
    print(pairs)

    # construct matrix A
    A = []
    for pair in pairs:
        u_1 = pair[0]
        v_1 = pair[1]
        u_2 = pair[3]
        v_2 = pair[4]
        A.append([u_1 * u_2, u_2 * v_1, u_2, u_1 * v_2, v_1 * v_2, v_2, u_1, v_1, 1])
    A = np.array(A)

    # svd
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape((3, 3))

    # rank 2 constraint
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    print("S with last eigenvalue 0:", S)
    F = U @ np.diag(S) @ V
    # print(F)
    # transform back to original coordinates
    F = T_2.T @ F @ T_1
    return F 


# 2c
def test_2b():
    pairs = np.loadtxt("data/epipolar/house_points.txt")
    image_1 = cv2.imread("data/epipolar/house1.jpg")
    image_2 = cv2.imread("data/epipolar/house2.jpg")
    images = [image_1, image_2]
    # columns are x, y, x', y'
    F = fundamental_matrix(pairs)
    print("F: \n", F)

    points_1 = pairs[:, 0:2]        # 10 x 2
    points_2 = pairs[:, 2:4]
    points = [points_1, points_2]
    # convert points to homogeneous coordinates
    points_1 = np.hstack((points_1, np.ones((points_1.shape[0], 1))))   # 10 x 3
    points_2 = np.hstack((points_2, np.ones((points_2.shape[0], 1))))   # 10 x 3
    
    # epipolar lines for each point
    lines_2 = (F @ points_1.T).T
    lines_1 = (F.T @ points_2.T).T
    lines = [lines_1, lines_2]
    print("lines_1: \n", lines_1)
    print("lines_2: \n", lines_2)

    fig = plt.figure(figsize=(5, 10))
    grid = gs.GridSpec(nrows=1, ncols=2)
    for i in range(2):
        sp = fig.add_subplot(grid[i])
        sp.imshow(images[i])
        for line in lines[i]:
            draw_epiline(sp, line, images[i].shape[0], images[i].shape[1])
        sp.scatter(points[i][:, 0], points[i][:, 1], c='r', s=30)
    plt.show()


# 2c
def reprojection_error(F: np.ndarray, p1, p2):
    # epipolar lines for each point
    l1 = F @ p1
    l2 = F.T @ p2
    
    # distance from point to line
    d1 = np.abs(l1 @ p2) / np.sqrt(l1[0]**2 + l1[1]**2)
    d2 = np.abs(l2 @ p1) / np.sqrt(l2[0]**2 + l2[1]**2)
    return (d1 + d2) / 2


def test_2c():
    # 1
    p1 = np.array([85, 233, 1])        # left image
    p2 = np.array([67, 219, 1])        # right image
    
    F = np.array([[-0.000000885211824, -0.000005615918803, 0.001943109518320],
                    [0.000009392818702, 0.000000616883199, -0.012006630150442],
                    [-0.001203084137613, 0.011037006977740, -0.085317335867129]])
    # print("F:\n", F.shape, F)

    err = reprojection_error(F, p1, p2)
    print("err = ", err)

    # 2
    pairs = np.loadtxt("data/epipolar/house_points.txt")
    errors = []
    for pair in pairs:
        p1 = np.array([pair[0], pair[1], 1])
        p2 = np.array([pair[2], pair[3], 1])
        err = reprojection_error(F, p1, p2)
        errors.append(err)
    print("avg err = ", np.mean(errors))


# 3a
# pairs: correspondences between 2 images in homogeneous coordinates
# P_1, P_2: projection matrices of size 3x4
# returns triangulated 3D points
def triangulate(pairs: np.ndarray, P_1: np.ndarray, P_2: np.ndarray):
    # convert pairs to homogeneous coordinates
    pairs = np.hstack((pairs[:, 0:2], np.ones((pairs.shape[0], 1)), pairs[:, 2:4], np.ones((pairs.shape[0], 1))))

    # triangulate each pair
    points_3d = []
    for pair in pairs:
        # extract 2D points
        x_1 = pair[0:3]
        x_2 = pair[3:6]

        # construct A matrix
        x_1_x = np.array([[0, -x_1[2], x_1[1]],
                        [x_1[2], 0, -x_1[0]],
                        [-x_1[1], x_1[0], 0]])
        x_2_x = np.array([[0, -x_2[2], x_2[1]],
                        [x_2[2], 0, -x_2[0]],
                        [-x_2[1], x_2[0], 0]])
        prod1 = x_1_x @ P_1
        prod2 = x_2_x @ P_2

        # combine first 2 lines of prod1 and prod2 into A
        A = np.vstack((prod1[0:2], prod2[0:2]))     # A is 4x4 matrix now
        
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        X = X / X[3]
        points_3d.append(X)
    return np.array(points_3d)


def test_3a():
    pairs = np.loadtxt("data/epipolar/house_points.txt")
    P_1 = np.loadtxt("data/epipolar/house1_camera.txt")
    P_2 = np.loadtxt("data/epipolar/house2_camera.txt")
    image_1 = cv2.imread("data/epipolar/house1.jpg")
    image_2 = cv2.imread("data/epipolar/house2.jpg")

    points_1 = pairs[:, 0:2]        # 10 x 2
    points_2 = pairs[:, 2:4]
    points = [points_1, points_2]
    data = [image_1, image_2, pts]

    pts = triangulate(pairs, P_1, P_2)
    print(pts)

    # transformation matrix for easier interpretation in 3d
    T = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]])
    pts = (T @ pts[:, 0:3].T).T
    print("transformed points:\n", pts)

    fig = plt.figure(figsize=(5, 15))
    grid = gs.GridSpec(nrows=1, ncols=3)
    for i in range(len(data)):
        if i < 2:
            sp = fig.add_subplot(grid[i])
            sp.imshow(data[i])
            sp.scatter(points[i][:, 0], points[i][:, 1], c='r', s=10)
            for j in range(len(points[i])):
                plt.text(points[i][j, 0], points[i][j, 1], str(j))
        else:
            sp = fig.add_subplot(grid[i], projection='3d')
            sp.mouse_init(rotate_btn=1, zoom_btn=3)
            sp.scatter(data[i][:, 0], data[i][:, 1], data[i][:, 2], c='r', s=10)
            # label dots
            for j in range(len(data[i])):
                sp.text(data[i][j, 0], data[i][j, 1], data[i][j, 2], str(j))
            sp.set_xlabel("x")
            sp.set_ylabel("y")
            sp.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    # calc_disparity()
    # test_2b()
    # test_2c()
    test_3a()
