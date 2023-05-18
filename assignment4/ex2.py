#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from a4_utils import *
from UZ_utils_a4 import *
from ex1 import hessian_points, harris_points, nms


# feat_pts_1 -> features of image 1
# feat_pts_2 -> features of image 2
# feat_pts_1 and 2 may not be of the same size
def find_correspondences(feat_pts_1: np.ndarray, feat_pts_2: np.ndarray):    
    # debug
    print(f"len 1: {len(feat_pts_1)}")
    print(f"len 2: {len(feat_pts_2)}")

    m1 = np.array([feat_pts_1 for _ in range(len(feat_pts_2))]).reshape((len(feat_pts_2), len(feat_pts_1), 16*16))
    m2 = np.array([[feats for _ in feat_pts_1] for feats in feat_pts_2]).reshape((len(feat_pts_2), len(feat_pts_1), 16*16))

    # hellinger distances between harris points
    # diffs = (np.sqrt(m1) - np.sqrt(m2)) ** 2
    d_hell: np.ndarray = np.sqrt(0.5 * np.sum((np.sqrt(m1) - np.sqrt(m2)) ** 2, axis=2))
    print("d_hell shape:", d_hell.shape)

    # return list of pairs [a, b], where a is index from the first list and b is index from the second.
    pairs = []
    for i in range(len(d_hell[0])):
        col = d_hell[:, i]
        min_d = np.min(col)
        min_idx = np.where(col == min_d)
        # print("found min indices: ", min_idx)
        min_idx = min_idx[0]
        # print("min idx: ", min_idx, [i, min_idx[0]])
        pairs.append([i, min_idx[0]])

    # debug
    print(f"n pairs: {len(pairs)}")
    
    # for each descriptor from feat_pts_1 find the closest descriptor from feat_pts_2

    return pairs


def test_a():
    sigma = 3       # reduce it to 2?
    tresh = 1e-6
    r = 15
    graf_a = "data/graf/graf_a_small.jpg"
    graf_b = "data/graf/graf_b_small.jpg"
    files = [graf_a, graf_b]

    images = []
    harris_pts = []
    treshd = []
    nmss = []       # suppressed non-maxima
    feat_pts = []   # feature points from both images
    descs = []      # descriptors of dimenson N_deatures * bins**2


    # fig = plt.figure(figsize=(5, 10))
    # grid = gs.GridSpec(nrows=1, ncols=2)

    for i in range(len(files)):
        image = imread_gray(files[i], "float64")
        images.append(image)

        harris = harris_points(image, sigma)
        harris_pts.append(harris)
        
        tr = np.where(harris > tresh, harris, 0)
        treshd.append(tr)
        
        n = nms(tr, r, tresh)
        nmss.append(n)

        feat = np.nonzero(n)        # tuple of arrays
        # print("feature pts. dims:", zip(feat[0], feat[1]))
        feat_pts.append(feat)
    
        # Y = feat[0], X = feat[1]
        # returns N_features x bins**2 array of descriptors
        desc = simple_descriptors(image, feat[0], feat[1])
        descs.append(desc)
        
    # fins correspondences between images 0 and 1
    # return list of pairs [a, b], where a is index from descs[0] and b is index from descs[1]
    # basically returns indices of feature points in corresponsing feat array
    # a je index feature pointa v feat_pts[0], b je index feature pointa v feat_pts[1]
    pairs = find_correspondences(descs[0], descs[1])
    print("pairs:\n", pairs)
    
    
    # pts is Nx2 array of feature points
    feats_1 = np.asarray(feat_pts[0]).T
    feats_2 = np.asarray(feat_pts[1]).T
    feats_1 = np.flip(feats_1, axis=1)
    feats_2 = np.flip(feats_2, axis=1)
    print("feats_1:\n", feats_1)
    print("feats_2:\n", feats_2)
    print("feats_1 shape:", feats_1.shape)
    print("feats_2 shape:", feats_2.shape)

    # sort feats_2 according to pairs
    feats_2_sorted = np.array([])
    for i in range(len(feats_1)):
        new_idx = pairs[i][1]
        feats_2_sorted = np.vstack((feats_2_sorted, feats_2[new_idx])) if feats_2_sorted.size else feats_2[new_idx]

    print("feats_2_sorted:\n", feats_2_sorted)

    display_matches(images[0], feats_1, images[1], feats_2_sorted)



# 2b
def find_matches(I1: np.ndarray, I2: np.ndarray, display: bool = True):
    # execute Hessian feature point detector
    sigma = 3
    images = [I1, I2]
    sigmas = [sigma, sigma]
    rs = [15, 15]
    # treshs = [0.004, 0.004]       # hessian
    treshs = [1e-6, 1e-6]           # harris
    feat_pts = []
    descs = []
    
    for i in range(len(images)):
        I = images[i]
        sigma = sigmas[i]
        points = harris_points(I, sigma)
        # points = hessian_points(I, sigma)
        
        points = np.where(points > treshs[i], points, 0)
        points = nms(points, rs[i], treshs[i])
        
        points = np.nonzero(points)

        feat_pts.append(points)

        # find simple descriptors
        desc = simple_descriptors(I, points[0], points[1])
        descs.append(desc)

        
    # find correspondences
    # Y = feat[0], X = feat[1] and vice versa
    # returns N_features x bins**2 array of descriptors
    pairs_1 = find_correspondences(descs[0], descs[1])
    pairs_2 = find_correspondences(descs[1], descs[0])
    print("pairs_1:\n", pairs_1)
    print("pairs_2:\n", pairs_2)

    # find only symmetric correspondences
    pairs = []
    for i in range(len(pairs_1)):
        if pairs_1[i][1] == pairs_2[pairs_1[i][1]][0] and pairs_1[i][0] == pairs_2[pairs_1[i][1]][1]:
            pairs.append(pairs_1[i])
    print("symmetric pairs:\n", pairs)

    feats_1 = np.asarray(feat_pts[0]).T
    feats_2 = np.asarray(feat_pts[1]).T
    feats_1 = np.flip(feats_1, axis=1)
    feats_2 = np.flip(feats_2, axis=1)
    # print("feats_1:\n", feats_1)
    # print("feats_2:\n", feats_2)
    # print("feats_1 shape:", feats_1.shape)
    # print("feats_2 shape:", feats_2.shape)

    # sort feats_2 according to pairs
    feats_1_sorted = np.array([])
    feats_2_sorted = np.array([])
    for i in range(len(pairs)):
        feats_1_idx = pairs[i][0]
        feats_2_idx = pairs[i][1]
        feats_1_sorted = np.vstack((feats_1_sorted, feats_1[feats_1_idx])) if feats_1_sorted.size else feats_1[feats_1_idx]

        feats_2_sorted = np.vstack((feats_2_sorted, feats_2[feats_2_idx])) if feats_2_sorted.size else feats_2[feats_2_idx]
    print("feats_1_sorted:\n", feats_1_sorted)
    print("feats_2_sorted:\n", feats_2_sorted)

    if display:
        display_matches(images[0], feats_1_sorted, images[1], feats_2_sorted)

    return (feats_1_sorted, feats_2_sorted)
    # pretty accurate, but not perfect


def test_b():
    filenames = ["data/graf/graf_a_small.jpg", "data/graf/graf_b_small.jpg"]
    images = []
    for filename in filenames:
        images.append(imread_gray(filename, "float64"))
    _, _ = find_matches(images[0], images[1])


if __name__ == "__main__":
    # test_a()
    test_b()
