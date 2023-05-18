#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from a4_utils import *
from UZ_utils_a4 import *


# a
# computes determinants of hessian
def hessian_points(I: np.ndarray, sigma: float) -> np.ndarray:
    Ix_Iy = get_partial_derivatives(I, sigma)
    Ixx_Iyy_Ixy = get_2nd_derivatives(Ix_Iy[0], Ix_Iy[1], sigma)

    det = Ixx_Iyy_Ixy[0] * Ixx_Iyy_Ixy[1] - Ixx_Iyy_Ixy[2] ** 2
    return det


# r     neighbourhood size
def nms(det: np.ndarray, r: int, tresh: float) -> np.ndarray:
    padded = np.copy(det) 
    padded = np.pad(padded, ((r, r), (r, r)), "constant", constant_values=((0, 0), (0, 0)))
    pxs = np.argwhere(det)      # does not include padding
    for i, j in pxs:
        # these indices are wrong when parametrized with r
        i_start = i
        i_end = i + 2 * r + 1
        j_start = j
        j_end = j + 2 * r + 1
        curr_response = det[i, j]
        neighbourhood = padded[i_start:i_end, j_start:j_end]
        neighbourhood[r, r] = 0     # set center element to 0 to exclude it from calculating neighbourhood max
        max = np.amax(neighbourhood)
        if curr_response > max and curr_response > tresh:
            # neighbourhood[i_start:i_end, j_start:j_end] = 0
            neighbourhood[r, r] = curr_response
    out = padded[r:len(padded) - r + 1, r:len(padded[0]) - r + 1]
    return out



def test_ab(case: str):
    assert case == "a" or case == "b", "case must be either 'a' or 'b'"

    test_points_path = "data/test_points.jpg"
    graf_a_path = "data/graf/graf_a.jpg"
    I = imread_gray(graf_a_path, "float64")

    sigmas = [3, 6, 9]
    tresh, r = 0, 0
    if case == "a":
        tresh = 0.004
        r = 15       # neighbourhood size
    else:
        # tresh = 1e-6
        tresh = 1.35e-6
        # tresh = 0
        r = 24
    fig = plt.figure(figsize=(10, 15))
    grid = gs.GridSpec(nrows=2, ncols=3)
    for i in range(len(sigmas)):
        out = None
        treshd = None
        if case == "a":
            out = hessian_points(I, sigmas[i])
        else:
            out = harris_points(I, sigmas[i])
        treshd = np.where(out > tresh, out, 0)
        treshd = nms(treshd, r, tresh)
        # non_zero = np.argwhere(treshd)
        # print("non zero pts after nms:\n", non_zero, len(non_zero))
        if case == "b":
            treshd = np.where(treshd > tresh, treshd, 0)
        sp = fig.add_subplot(grid[0, i])
        # sp.imshow(out, cmap="gray")
        sp.imshow(out)
        sp.set_title(f"sigma = {sigmas[i]}")

        sp = fig.add_subplot(grid[1, i])
        sp.imshow(I, cmap="gray")
        points = np.nonzero(treshd)
        sp.scatter(points[1], points[0], c="#ff0000", marker="x", linewidths=1)
    
    plt.show()

# Q: What kind of structures in the image are detected by the algorithm?
# A: Curvatures
# Q: How does the parameter sigma affect the result?
# A: The bigger the sigma, the bigger the detected feature 


# b
def harris_points(I: np.ndarray, sigma: float):
    sigma_gauss = sigma * 1.6
    alpha = 0.06
    d = get_partial_derivatives(I, sigma)
    I_x = d[0]
    I_y = d[1]
    G = gauss(sigma_gauss)
    G_T = np.transpose(G)
    # UL = upper left, LP = lower right element in the auto-correlation matrix
    UL = cv2.filter2D(np.power(I_x, 2), -1, G)
    UL = cv2.filter2D(UL, -1, G_T)
    antidiag = cv2.filter2D(I_x * I_y, -1, G)
    antidiag = cv2.filter2D(antidiag, -1, G_T)
    LR = cv2.filter2D(np.power(I_y, 2), -1, G)
    LR = cv2.filter2D(LR, -1, G_T)
    det = UL * LR - np.power(antidiag, 2)
    trace = UL + LR
    out = det - alpha * np.power(trace, 2)
    return out

# Feature points do not appear on the same structures in the image compared to the Hessian method.


if __name__ == "__main__":
    # test_ab("a")
    test_ab("b")
