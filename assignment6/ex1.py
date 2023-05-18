#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from a6_utils import *
from UZ_utils_a6 import *


def pca(data: np.ndarray):
    mu: np.ndarray = np.mean(data, axis=1)
    print("mu:\n", mu, mu.shape)
    N: int = data.shape[1]
    X_d: np.ndarray = data - mu[:, np.newaxis]
    print("centered data:\n", X_d)

    # covariance matrix
    C: np.ndarray = (1 / (N - 1)) * (X_d @ X_d.T)
    print("cov:\n", C)

    return (mu, C)


def draw_pca(points: np.ndarray, mu: np.ndarray, C: np.ndarray, U: np.ndarray, S: np.ndarray):
    fig = plt.figure(figsize=(5, 10))
    grid = gs.GridSpec(nrows=1, ncols=2)

    sp = fig.add_subplot(grid[0], xlim=(-1, 7), ylim=(-1, 7))
    sp.scatter(points[0, :], points[1, :])
    sp.scatter(mu[0], mu[1], c="r")

    # scaled eigenvectors to the standard deviations along the eigenvectors
    e1 = U[:, 0] * np.sqrt(S[0])
    e2 = U[:, 1] * np.sqrt(S[1])

    # draw cov ellipse
    drawEllipse(mu, C)

    # plot eigenvectors and scale the length of the vectors to the standard deviations along the eigenvectors
    # plt.quiver(mu[0], mu[1], e1[0], e1[1], color="r", scale=scale_e1)   # ????
    # plt.quiver(mu[0], mu[1], e2[0], e2[1], color="g", scale=S[0])
    sp.arrow(mu[0], mu[1], e1[0], e1[1], color="r", length_includes_head=True)
    sp.arrow(mu[0], mu[1], e2[0], e2[1], color="g", length_includes_head=True)

    # plot cumulative graph of eigenvalues and normalize it so that the largest value is 1
    sp = fig.add_subplot(grid[1], xlim=(0, 1), ylim=(0, 1))
    sp.plot(np.cumsum(S) / np.sum(S))
    sp.set_xlabel("eigenvalues")
    sp.set_ylabel("cumulative sum of eigenvalues")

    plt.show()

    # we lose 20% info. if not including the 2nd eigenvector


def test_pca():
    points_a: np.ndarray = np.array([[3, 3, 7, 6], [4, 6, 6, 4]])
    points_b: np.ndarray = np.loadtxt("data/points.txt")

    points_b = np.transpose(points_b)

    mu, C = pca(points_b)

    # svd of covariance matrix
    U, S, V = np.linalg.svd(C)
    print("U:\n", U)
    print("S:\n", S)

    # draw
    # draw_pca(points_b, mu, C, U, S)

    # e
    # project points_b to PCA space
    projected_pts = U.T @ (points_b - mu[:, np.newaxis])
    print("projected pts:\n", projected_pts)
    projected_pts[1, :] = 0
    # project the points back to the cartesian space
    # diminished_U = np.copy(U)
    # diminished_U[1, :] = 0
    projected_pts = U @ projected_pts + mu[:, np.newaxis]
    # draw_pca(projected_pts, mu, C, U, S)
    # data gets projected onto the first eigenvector

    # f
    q = np.array([6, 6])
    # calculate distance between q and the closest points in points_b
    dists = np.linalg.norm(points_b - q[:, np.newaxis], axis=0)
    closest_idx = np.argmin(dists)
    # the closest is the point with index 2 (5, 4)
    print("closest point:", points_b[:, closest_idx])

    new_pts = np.hstack((points_b, q[:, np.newaxis]))
    new_pts = U.T @ (new_pts - mu[:, np.newaxis])
    new_pts[1, :] = 0
    new_pts = U @ new_pts + mu[:, np.newaxis]
    draw_pca(new_pts, mu, C, U, S)
    print("new points:\n", new_pts)

    # which point is the closest now to q?
    print("new q:\n", new_pts[:, -1])
    dists = np.linalg.norm(new_pts[:, :-1] - new_pts[:, -1][:, np.newaxis], axis=0)
    closest_idx = np.argmin(dists)
    print("closest point now:", new_pts[:, closest_idx])


# ex 2
def dual_pca(data: np.ndarray):
    mu: np.ndarray = np.mean(data, axis=1)
    print("mu:\n", mu, mu.shape)
    N: int = data.shape[1]
    print("N:", N)
    X_d: np.ndarray = data - mu[:, np.newaxis]
    print("centered data:\n", X_d)

    # dual covariance matrix
    C: np.ndarray = (1 / (N - 1)) * (X_d.T @ X_d)
    print("dual cov mat:\n", C)

    # svd of dual covariance matrix
    U, S, _ = np.linalg.svd(C)

    U = X_d @ U @ np.diag(np.sqrt(1 / (S * (N - 1))))
    print("S:\n", S)

    return (mu, C, U, S)


def test_dual_pca():
    points: np.ndarray = np.loadtxt("data/points.txt")
    points = np.transpose(points)
    mu, C, U, S = dual_pca(points)
    print("U:\n", U)
    draw_pca(points, mu, C, U, S)       # ellipse is not oriented correctly

    # 2b
    projected_pts = U.T @ (points - mu[:, np.newaxis])
    projected_pts = U @ projected_pts + mu[:, np.newaxis]

    draw_pca(projected_pts, mu, C, U, S)


# 3a
def prepare_data(set: int):
    assert set < 4 and set > 0, "set must be 1, 2 or 3"

    filenames = [f"{i:03}.png" for i in range(1, 65)]
    images = np.array([])
    dims = None
    
    for filename in filenames:
        path = os.path.join("data", "faces", str(set), filename)
        img = imread_gray(path, "float64")
        if dims is None:
            dims = img.shape
        img = np.reshape(img, (img.shape[0] * img.shape[1], 1))
        images = np.hstack((images, img)) if images.size else img

    return images, dims


# 3b
def apply_dual_pca(data: np.ndarray):
    mu: np.ndarray = np.mean(data, axis=1)
    N: int = data.shape[1]
    X_d: np.ndarray = data - mu[:, np.newaxis]
    C: np.ndarray = (1 / (N - 1)) * (X_d.T @ X_d)
    U, S, _ = np.linalg.svd(C)
    print(S)
    epsilon = 1e-15
    U = X_d @ U @ np.diag(np.sqrt(1 / ((S + epsilon) * (N - 1))))
    
    return (U, mu)


def test_3bc(set: int):
    assert set < 4 and set > 0, "set must be 1, 2 or 3"
    data, dims = prepare_data(set)
    U, mu = apply_dual_pca(data)        # data in allpixels * 64
    print("data dims:", data.shape)
    print("mu dims:", mu.shape)
    print("mu shaped dims:", mu[:, np.newaxis].shape)
    print("U dims:", U.shape)
    images = []
    fig = plt.figure(figsize=(5, 25))
    grid = gs.GridSpec(nrows=1, ncols=5)
    for i in range(5):
        image = U[:, i]
        image = np.reshape(image, (dims[0], dims[1]))
        images.append(image)
        sp = fig.add_subplot(grid[i])
        sp.imshow(image, cmap="gray")
    plt.show()
    # what do these images represent numerically and in context of faces?

    # project the first image to the PCA space and reconstruct it
    print("1 img dims:", data[:, 0].shape)
    changed_img = np.copy(data[:, 0])
    changed_img = np.reshape(changed_img, (dims[0] * dims[1], 1))
    changed_img[4074] = 0
    projected = U.T @ (changed_img - mu[:, np.newaxis])
    projected[0:4] = 0
    reconstructed = U @ projected + mu[:, np.newaxis]
    print("projected dims:", projected.shape)
    print("reconstructed dims:", reconstructed.shape)

    # plot the original image, image with 1 pixel changed and the reconstructed image
    to_display = [data[:, 0], changed_img, reconstructed]
    fig = plt.figure(figsize=(5, 15))
    grid = gs.GridSpec(nrows=1, ncols=3)
    for i in range(len(to_display)):
        sp = fig.add_subplot(grid[i])
        sp.imshow(np.reshape(to_display[i], (dims[0], dims[1])), cmap="gray")
    plt.show()

    # c
    img = np.copy(data[:, 0])
    img = np.reshape(img, (dims[0] * dims[1], 1))
    
    fig = plt.figure(figsize=(5, 30))
    grid = gs.GridSpec(nrows=1, ncols=6)
    idxs = [2 ** i for i in range(5, -1, -1)]
    print(idxs)
    for i in range(len(idxs)):
        projected = U.T @ (img - mu[:, np.newaxis])
        projected[idxs[i]:] = 0
        reconstructed = U @ projected + mu[:, np.newaxis]
        sp = fig.add_subplot(grid[i])
        sp.imshow(np.reshape(reconstructed, (dims[0], dims[1])), cmap="gray")
        sp.set_title(str(idxs[i]))
    plt.show()




if __name__ == "__main__":
    test_pca()
    # test_dual_pca()
    # test_3bc(1)