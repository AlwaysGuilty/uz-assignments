#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from UZ_utils_a3 import *

# b
# derived 1D gaussian kernel g(x)', returns a row
def gaussdx(sigma: float) -> np.ndarray:
    std3 = np.ceil(3 * sigma)
    D = np.arange(-std3, std3 + 1)
    D = (-1.0 / (np.sqrt(2 * np.pi) * sigma ** 3)) * D * np.exp(-(D ** 2) / (2 * (sigma ** 2)))
    D = D / np.sum(np.abs(D))
    D = D.reshape([1, D.shape[0]])
    return D

# 1D gaussian kernel, returns a row
def gauss(std: float) -> np.ndarray:
    std3 = np.ceil(3 * std)
    G = np.arange(-std3, std3 + 1)
    G = (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-(G ** 2) / (2 * (std ** 2)))
    G = G / np.sum(G)
    G = G.reshape([1, G.shape[0]])
    return G

# c
def c():
    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1

    std = 5
    G = gauss(std)
    D = gaussdx(std)
    D = np.flip(D)
    G_T = G.reshape([G.shape[1], 1])
    D_T = D.reshape([D.shape[1], 1])

    a = cv2.filter2D(impulse, -1, G)
    a = cv2.filter2D(a, -1, G_T)
    b = cv2.filter2D(impulse, -1, G)
    b = cv2.filter2D(b, -1, D_T)
    c = cv2.filter2D(impulse, -1, D)
    c = cv2.filter2D(c, -1, G_T)
    d = cv2.filter2D(impulse, -1, G_T)
    d = cv2.filter2D(d, -1, D)
    e = cv2.filter2D(impulse, -1, D_T)
    e = cv2.filter2D(e, -1, G)

    # display images
    images = [impulse, a, b, c, d, e]
    titles = ["Impulse", "G, Gt", "G, Dt", "D, Gt", "Gt, D", "Dt, G"]
    fig = plt.figure(figsize=(10, 15))
    grid = gs.GridSpec(nrows=2, ncols=3)
    for i in range(len(images)):
        sp = fig.add_subplot(grid[i % 2, i // 2])
        sp.imshow(images[i], cmap="gray")
        sp.set_title(titles[i])
    plt.show()

    # Order of operations is not important, because of convolution's properties 

# d
# returns a list of 2 elements:
#   0: I_x
#   1: I_y
def get_partial_derivatives(I: np.ndarray, sigma: float) -> list:
    G = gauss(sigma)
    D = gaussdx(sigma)
    D = np.flip(D)
    G_T = G.reshape([G.shape[1], 1])
    D_T = D.reshape([D.shape[1], 1])
    
    I_x = cv2.filter2D(I, -1, G_T)
    I_x = cv2.filter2D(I_x, -1, D)
    I_y = cv2.filter2D(I, -1, G)
    I_y = cv2.filter2D(I_y, -1, D_T)
    
    return [I_x, I_y]

# expects partial first order derivatives I_x and I_y
# returns list of 3 elements of partial derivatives of second order
#   0: I_xx
#   1: I_yy
#   2: I_xy 
def get_partial_2nd_derivatives(I_x: np.ndarray, I_y: np.ndarray, sigma: float) -> list:
    G = gauss(sigma)
    D = gaussdx(sigma)
    D = np.flip(D)
    G_T = G.reshape([G.shape[1], 1])
    D_T = D.reshape([D.shape[1], 1])

    I_xx = cv2.filter2D(I_x, -1, G_T)
    I_xx = cv2.filter2D(I_xx, -1, D)
    I_yy = cv2.filter2D(I_y, -1, G)
    I_yy = cv2.filter2D(I_yy, -1, D_T)

    I_xy = cv2.filter2D(I_x, -1, G)
    I_xy = cv2.filter2D(I_xy, -1, D_T)

    return [I_xx, I_yy, I_xy]


# Expects grayscale image
# Returns list of 2 elements
#   0: derivative magnitudes M
#   1: derivative angles fi
def gradient_magnitude(I: np.ndarray, sigma: float) -> list:
    d = get_partial_derivatives(I, sigma)
    I_x = d[0]
    I_y = d[1]
    M = np.sqrt(I_x ** 2 + I_y ** 2)
    fi = np.arctan2(I_y, I_x)
    return [M, fi]


def test_d():
    sigma = 1
    # I = imread_gray_cv2("images/museum.jpg", "float64")
    path = "images/museum.jpg"

    # assert os.path.exists(path)

    I = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    I = np.asarray(I)
    I = I.astype(np.float64) / 255
    d = get_partial_derivatives(I, sigma)
    I_x = d[0]
    I_y = d[1]
    dd = get_partial_2nd_derivatives(I_x, I_y, sigma)
    mag_dir = gradient_magnitude(I, sigma)
    
    images = [I] + d + dd + mag_dir
    titles = ["Original", "I_x", "I_y", "I_xx", "I_yy", "I_xy", "I_mag", "I_dir"]
    fig = plt.figure(figsize=(10, 20))
    grid = gs.GridSpec(nrows=2, ncols=4)
    for i in range(len(images)):
        sp = fig.add_subplot(grid[i % 2, i // 2])
        sp.imshow(images[i], cmap="gray")
        sp.set_title(titles[i])
    plt.show()

if __name__ == "__main__":
    test_d()
