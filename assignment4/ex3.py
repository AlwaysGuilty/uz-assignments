#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from a4_utils import *
from UZ_utils_a4 import *
from ex1 import hessian_points, harris_points, nms
from ex2 import find_matches


# display images in a 1xN grid
def display_images(*images):
    # display images in a grid
    n = len(images)
    fig = plt.figure(figsize=(5, 5 * n))
    grid = gs.GridSpec(nrows=1, ncols=n)
    for i in range(n):
        sp = fig.add_subplot(grid[i])
        sp.imshow(images[i], cmap="gray")
    plt.show()


# Homography
# x = [x_t, y_t]
# p = [p1, p2, p3, p4]
# p1, p2 for rotation and scale, p3 and p4 for translation
def estimate_homography(feats_1: np.ndarray, feats_2: np.ndarray):
    # approximates homography between 2 images using a given set if matched feature points
    # construct matrix A
    A = np.array([])
    for i in range(len(feats_1)):
        x_r, y_r = feats_1[i]
        x_t, y_t = feats_2[i]
        A_i = np.array([
            [x_r, y_r, 1, 0, 0, 0, -x_t*x_r, -x_t*y_r, -x_t],
            [0, 0, 0, x_r, y_r, 1, -y_t*x_r, -y_t*y_r, -y_t]
        ])
        A = np.vstack((A, A_i)) if A.size else A_i 
    print("A:\n", A)

    # solve for h
    U, S, V = np.linalg.svd(A)
    h = V[-1] / V[-1, -1]
    print("h:\n", h, h.shape)

    # construct H
    H = np.reshape(h, (3, 3))
    print("H:\n", H)

    return H


# set can be wither "newyork" or "graf"
def test_a(set: str):
    assert set == "newyork" or set == "graf", "set must be either newyork or graf"

    filenames = []
    paths = []
    images = []
    corr = None

    if set == "newyork":
        filenames = [set + "_a.jpg", set + "_b.jpg"]
    else:
        filenames = [set + "_a.jpg", set + "_b.jpg"]

    paths = [os.path.join("data", set, filenames[0]), os.path.join("data", set, filenames[1])]
    titles = [filenames[0], filenames[1], "transformed_" + filenames[0]]
    corr = np.loadtxt(os.path.join("data", set, set + ".txt"))

    for file in paths:
        I: np.ndarray = imread_gray(file, "float64")
        images.append(I)
    # load corresponding feature points
    # columns: x1, y1, x2, y2
    

    # display_matches(images[0], corr[:, 0:2], images[1], corr[:, 2:4])
    
    H = estimate_homography(corr[:, 0:2], corr[:, 2:4])

    transformed = cv2.warpPerspective(images[0], H, (images[0].shape[1], images[0].shape[0]))
    images.append(transformed)
    

    fig = plt.figure(figsize=(5, 15))
    grid = gs.GridSpec(nrows=1, ncols=3)
    for i in range(len(images)):
        sp = fig.add_subplot(grid[i])
        sp.imshow(images[i], cmap="gray")
        sp.set_title(titles[i])
    plt.show()


# b
# RANSAC
def ransac():
    filenames = ["data/newyork/newyork_a.jpg", "data/newyork/newyork_b.jpg"]
    images = []
    for filename in filenames:
        images.append(imread_gray(filename, "float64"))
    feats_1, feats_2 = find_matches(images[0], images[1], display=False)

    # ransac loop
    # iterations
    k = 50
    threshold = 2
    min_error = 10000000
    H_best = None
    feats_1_best = None
    feats_2_best = None
    while k > 0:
        idxs = np.random.randint(0, len(feats_1), 4)
        rand_selected_1 = feats_1[idxs]
        rand_selected_2 = feats_2[idxs]
        print("rand_selected_1:\n", rand_selected_1)
        print("rand_selected_2:\n", rand_selected_2)
        # estimate homography
        H = estimate_homography(rand_selected_1, rand_selected_2)

        feats_1_transformed = H @ np.vstack((feats_1.T, np.ones((1, feats_1.shape[0]))))
        feats_1_transformed = feats_1_transformed[0:2, :] / feats_1_transformed[2, :]
        feats_1_transformed = feats_1_transformed.T
        print("feats_1_transformed:\n", feats_1_transformed)
        print("feats_2:\n", feats_2)

        # error calc
        err_vector = np.linalg.norm(feats_1_transformed - feats_2, axis=1)
        print("err:\n", err_vector, len(err_vector))

        # find inliers
        inliers = np.where(err_vector < threshold)[0]
        print("inliers:\n", inliers)

        if len(inliers) > len(feats_1) * 0.5:
            H = estimate_homography(feats_1[inliers], feats_2[inliers])
            new_feats_1_transformed = H @ np.vstack((feats_1.T, np.ones((1, feats_1.shape[0]))))
            new_feats_1_transformed = new_feats_1_transformed[0:2, :] / new_feats_1_transformed[2, :]
            new_feats_1_transformed = new_feats_1_transformed.T
            new_err_vector = np.linalg.norm(new_feats_1_transformed - feats_2, axis=1)
            if np.mean(new_err_vector) < min_error:
                H_best = H
                feats_1_best = feats_1[inliers]
                feats_2_best = feats_2[inliers]
                min_error = np.mean(new_err_vector)
        k -= 1


    # transform image newyork_a with H_best
    transformed_newyork_a = cv2.warpPerspective(images[0], H_best, (images[0].shape[1], images[0].shape[0]))

    fig = plt.figure(figsize=(10, 10))
    grid = gs.GridSpec(nrows=2, ncols=2)
    sp = fig.add_subplot(grid[0, :])
    display_matches(images[0], feats_1_best, images[1], feats_2_best, show=False)
    sp = fig.add_subplot((grid[1, 0]))
    sp.imshow(images[1], cmap="gray")
    sp.set_title("newyork_b")
    sp = fig.add_subplot((grid[1, 1]))
    sp.imshow(transformed_newyork_a, cmap="gray")
    sp.set_title("transformed newyork_a")
    plt.show()


if __name__ == "__main__":
    # test_a("newyork")
    # test_a("graf")
    ransac()