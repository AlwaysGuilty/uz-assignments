#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from UZ_utils_a3 import *
from ex1 import gradient_magnitude


# a
# theta: treshold
def findedges(I: np.ndarray, sigma: float, theta: float) -> np.ndarray:
    mag_dir = gradient_magnitude(I, sigma)
    I_mag = mag_dir[0]
    I_e = np.where(I_mag > theta, 1, 0)
    # print(mag_dir[1])
    return I_e

def test_ab():
    sigma = 1
    theta_1 = 0.05
    theta_2 = 0.16
    I = imread_gray_cv2("images/museum.jpg", "float64")
    mag_dir = gradient_magnitude(I, sigma)
    I_e_1 = findedges(I, sigma, theta_1)
    I_e_2 = findedges(I, sigma, theta_2)
    n = nms(I_e_2, mag_dir[0], mag_dir[1])

    images = [I, I_e_1, I_e_2, n]
    titles = ["Original", f"Thresholded (thr = {theta_1})", f"Thresholded (thr = {theta_2})", f"Nonmax. supp. (thr = {theta_2})"]
    fig = plt.figure(figsize=(10, 10))
    grid = gs.GridSpec(nrows=2, ncols=2)
    for i in range(len(images)):
        sp = fig.add_subplot(grid[i % 2, i // 2])
        sp.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        sp.set_title(titles[i])
    plt.show()

    # Q: Can we set the parameter so that all the edges in the image are clearly visible?
    # A: Not quite. Theta_1 shows a lot of edges, almost all of them, but some are still missing


# b
# non-maxima suppression
# expects a tresholded image of edges (binary image of 0s and 1s), gradient magnitudes and direction vectors 
def nms(I: np.ndarray, I_mag: np.ndarray, I_dir: np.ndarray) -> np.ndarray:
    I_copy = np.copy(I)
    pxs = np.argwhere(I_copy)
    for i, j in pxs:
        theta = I_dir[i, j]
        mag = I_mag[i, j]
        
        # check if out of bounds    -> pad the image with 0s instead of writing this ugly code
        # borders for i
        # if px_i == 0:
        #     i_start = px_i
        #     i_end = px_i + 2
        # elif px_i == len(I) - 1:
        #     i_start = px_i - 1
        #     i_end = px_i + 1
        # else:
        #     i_start = px_i - 1
        #     i_end = px_i + 2
        
        # # borders for j
        # if px_j == 0:
        #     j_start = px_j
        #     j_end = px_j + 2
        # elif px_j == len(I[0]) - 1:
        #     j_start = px_j - 1
        #     j_end = px_j + 1
        # else:
        #     j_start = px_j - 1
        #     j_end = px_j + 2

        # let's skip border pixels for now
        if i == 0 or i == len(I) - 1 or j == 0 or j == len(I[0]) - 1:
            continue

        i_start = i - 1
        i_end = i + 2
        j_start = j - 1
        j_end = j + 2

        # komentar asistenta:
        # lahko bi uporabil modulo operacijo nad theto da bi dobil nek lookuptable za kote
        # look left and right
        if theta > - np.pi / 8 and theta <= np.pi / 8 or theta <= - (7 * np.pi) / 8 or theta > (7 * np.pi) / 8:
            slice = I_mag[i, j_start:j_end]
            max = np.amax(slice)
            if max > mag:
                I_copy[i, j] = 0
        # look up and down
        elif theta > (3 * np.pi) / 8 and theta <= (5 * np.pi) / 8 or theta <= - (3 * np.pi) / 8 and theta > - (5 * np.pi) / 8:
            slice = I_mag[i_start:i_end, j]
            max = np.amax(slice)
            if max > mag:
                I_copy[i, j] = 0
        # look at antidiagonal (x = y)
        elif theta > np.pi / 8 and theta <= (3 * np.pi) / 8 or theta > - (7 * np.pi) / 8 and theta <= - (5 * np.pi) / 8:
            slice = [I_mag[i, j], I_mag[i - 1, j - 1], I_mag[i + 1, j + 1]]
            max = np.amax(slice)
            if max > mag:
                I_copy[i, j] = 0
        # diagonal (x = -y)
        else:
            slice = [I_mag[i, j], I_mag[i - 1, j + 1], I_mag[i + 1, j - 1]]
            max = np.amax(slice)
            if max > mag:
                I_copy[i, j] = 0
    return I_copy

if __name__ == "__main__":
    test_ab()