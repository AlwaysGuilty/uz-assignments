#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from UZ_utils_a3 import *
from a3_utils import *
from ex1 import gradient_magnitude
from ex2 import findedges, nms


# expects I, binary image of 0's and 1's representing edges?
def hough_find_lines(I: np.ndarray, rho_bins: int, theta_bins: int, treshold: float) -> np.ndarray:
    # get only edges
    pxs = np.nonzero(I)
    # get diagonal of the image (max Rho)
    rho_max = np.sqrt(I.shape[0] ** 2 + I.shape[1] ** 2) 
    H = np.zeros((rho_bins, theta_bins))
    thetas = np.linspace(-np.pi / 2, np.pi / 2, num=theta_bins)
    for x, y in zip(pxs[0], pxs[1]):
        rhos = y * np.cos(thetas) + x * np.sin(thetas)
        rhos = (rho_bins - 1) * ((rhos + rho_max) / (2 * rho_max))
        for i in range(len(thetas)):
            H[int(rhos[i]), i] += 1
            # H[i, int(rhos[i])] += 1
    # H = np.where(H > treshold, H, 0)
    return H
    

def a():
    xs = [10, 30, 50, 80]
    ys = [10, 60, 20, 90]
    fig = plt.figure(figsize=(10, 10))
    grid = gs.GridSpec(nrows=2, ncols=2)
    for i in range(4):
        I = np.zeros((100, 100))
        I[xs[i], ys[i]] = 1
        H = hough_find_lines(I, 300, 300, 0)
        title = f"x = {xs[i]}, y = {ys[i]}"
        sp = fig.add_subplot(grid[i % 2, i // 2])
        sp.imshow(H)
        sp.set_title(title)
    plt.show()


# b, c, d
# case can be either "b", "c" or "d"
# e_treshold:   treshold for edge extraction
# acc_treshold  treshold for extracting only certain thetas and rhos from accumulator matrix
def test_bcd(case: str, e_treshold: float):
    sigma = 1
    rho_bins = 500
    theta_bins = 500
    syntetic = np.zeros((100, 100))
    syntetic[10, 10] = 1
    syntetic[10, 20] = 1
    path_oneline = "images/oneline.png"
    path_rectangle = "images/rectangle.png"
    # assert os.path.exists(path_oneline)
    oneline = imread_gray_cv2(path_oneline, "float64")
    rectangle = imread_gray_cv2(path_rectangle, "float64")
    oneline_mag_dir = gradient_magnitude(oneline, sigma)
    rectangle_mag_dir = gradient_magnitude(rectangle, sigma)
    oneline_e = findedges(oneline, sigma, e_treshold)
    rectangle_e = findedges(rectangle, sigma, e_treshold)
    oneline_e_nms = nms(oneline_e, oneline_mag_dir[0], oneline_mag_dir[1])
    rectangle_e_nms = nms(rectangle_e, rectangle_mag_dir[0], rectangle_mag_dir[1])
    images = [syntetic, oneline, rectangle]
    # edges = [syntetic, oneline_e, rectangle_e]
    edges = [syntetic, oneline_e_nms, rectangle_e_nms]
    titles = ["Synthetic", "oneline.png", "rectangle.png"]
    # acc_tresholds = [0.9, 0.9, 0.6]
    acc_tresholds = [1, 390, 200]

    fig = plt.figure(figsize=(5, 15))
    grid = gs.GridSpec(nrows=1, ncols=3)    
    for i in range(len(images)):
        sp = fig.add_subplot(grid[i])
        H = hough_find_lines(edges[i], rho_bins, theta_bins, 0)
        if case == "b":    
            sp.imshow(H)
        elif case == "c":
            n = nonmaxima_suppression_box(H)
            sp.imshow(n)
        elif case == "d":
            H = nonmaxima_suppression_box(H)
            H = np.where(H > acc_tresholds[i], H, 0)
            # extract pairs that are larger than acc_treshold
            # extract binned values of rhos and thetas
            pairs = np.argwhere(H > 0.0)
            # reverse thetas
            thetas = (pairs[:, 1] / (theta_bins - 1)) * np.pi - (np.pi / 2)
            # reverse rhos
            rho_max = np.sqrt(edges[i].shape[0] ** 2 + edges[i].shape[1] ** 2)
            rhos = ((pairs[:, 0] / (rho_bins - 1)) * (2 * rho_max)) - rho_max
            sp.imshow(images[i], cmap="gray")
            # sp.imshow(H, cmap="gray")
            for rho, theta in zip(rhos, thetas):
                draw_line(rho, theta, images[i].shape[0], images[i].shape[1])
        else:
            print("test_bd: invalid case")
        sp.set_title(titles[i])
    plt.show()


# c
def nonmaxima_suppression_box(I: np.ndarray) -> np.ndarray:
    I_cpy = np.copy(I) 
    I_pad = np.pad(I_cpy, ((1, 1), (1, 1)), "constant", constant_values=((0, 0), (0, 0)))
    pxs = np.argwhere(I_cpy > 0.0)    
    for i, j in pxs:
        # skip edge pixels
        # if i == 0 or i == len(I) - 1 or j == 0 or j == len(I[0]) - 1:
        #     continue
        # plus 1 to adjust for the padding
        i_start = i - 1 + 1
        i_end = i + 2 + 1
        j_start = j - 1 + 1
        j_end = j + 2 + 1
        neighbourhood = I_pad[i_start:i_end, j_start:j_end]
        max = np.amax(neighbourhood)
        # sets pixels in the neighbourhood that are not max to 0
        I_pad[i_start:i_end, j_start:j_end] = np.where(neighbourhood < max, 0, max)
        # get all maxes, if there are more 
        maxs = np.argwhere(neighbourhood > 0.0)
        # print(maxs)
        if I_pad[i + 1, j + 1] == max:
            neighbourhood = 0
            I_pad[i + 1, j + 1] = max
        else:
            neighbourhood = 0
            I_pad[i_start + maxs[0][0], j_start + maxs[0][1]] = max
    out = I_pad[1:len(I_pad) - 1, 1:len(I_pad[0]) - 1]
    return out


def e():
    sigma = 1
    e_treshold = 0.17
    rho_bins = 400
    theta_bins = 400
    top_n = 10      # how many top bins we select
    bricks_filename = "bricks.jpg"
    pier_filename = "pier.jpg"
    filenames = [bricks_filename, pier_filename]
    data_type = np.dtype([("index", np.ndarray, 2), ("value", np.float64)])
    fig = plt.figure(figsize=(10, 10))
    grid = gs.GridSpec(nrows=2, ncols=2)
    for i in range(len(filenames)):
        # I = imread_gray_cv2("images/" + filenames[i], "float64")
        I = cv2.imread("images/" + filenames[i], 1)
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        I_gray = cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)
        I_gray = np.asarray(I_gray)
        I_gray = I_gray.astype(np.float64) / 255
        mag_dir = gradient_magnitude(I_gray, sigma)
        e = findedges(I_gray, sigma, e_treshold)
        n = nms(e, mag_dir[0], mag_dir[1])
        H = hough_find_lines(n, rho_bins, theta_bins, 0)
        H_n = nonmaxima_suppression_box(H)

        sp = fig.add_subplot(grid[0, i])
        sp.imshow(H)
        # sp.imshow(H_n)
        sp.set_title(filenames[i])

        print(f"i={i}\nmax: {np.amax(H_n)}\nmin: {np.amin(H_n[np.nonzero(H_n)])}")      # debug
        # H_n_tresh = np.where(H > acc_tresholds[i], H, 0)
        indices = np.argwhere(H_n)          # indices and corresponding values
        values = H_n[np.nonzero(H_n)]
        pairs = list(zip(indices, values))
        pairs = np.array(pairs, dtype=data_type)
        
        pairs = np.sort(pairs, order="value")
        pairs = np.flip(pairs)
        pairs = pairs[:top_n]

        print(pairs)
        pairs_indices = pairs["index"]

        thetas = (pairs_indices[:, 1] / (theta_bins - 1)) * np.pi - (np.pi / 2)
        rho_max = np.sqrt(e.shape[0] ** 2 + e.shape[1] ** 2)
        rhos = ((pairs_indices[:, 0] / (rho_bins - 1)) * (2 * rho_max)) - rho_max

        sp = fig.add_subplot(grid[1, i])
        sp.imshow(n)
        for rho, theta in zip(rhos, thetas):
            draw_line(rho, theta, I.shape[0], I.shape[1])
    plt.show()


if __name__ == "__main__":
    # a()
    # test_bcd("b", 0.16)
    # test_bcd("c", 0.16)
    test_bcd("d", 0.16)
    # e()

