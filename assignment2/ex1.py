#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

from UZ_utils_a2 import *
from a2_utils import *
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

# a
def myhist3(I: np.ndarray, n_bins: int) -> np.ndarray:
    # reduction of image to n_bins
    I = np.floor(I * n_bins).astype(np.uint8)   # og is float, convert to uint8
    H = np.zeros((n_bins, n_bins, n_bins))
    for row in I:
        for pixel in row:
            # gledamo koliko je barve kot celote v sliki, ne koliko je katere izmed njenih komponent
            H[pixel[0], pixel[1], pixel[2]] += 1      
    H = H / np.sum(H)
    return H

# b
"""
measure can be one of the following: 'l2', 'chi-square', 'intersection' or 'hellinger'
"""
def compare_histograms(H_1: np.ndarray, H_2: np.ndarray, measure: str):
    result = -1
    # check dims
    if not H_1.shape == H_2.shape:
        print("Histogram dimensions do not match")
    # euclidian distance
    if measure == "l2":
        result = np.sqrt(np.sum((H_1 - H_2) ** 2))
    elif measure == "chi-square":
        epsilon_0 = 10 ** (-10)
        result = 0.5 * np.sum(((H_1 - H_2) ** 2) / (H_1 + H_2 + epsilon_0))
    elif measure == "intersection":
        result = 1 - np.sum(np.amin([H_1, H_2], axis = 0))
    elif measure == "hellinger":
        result = np.sqrt(0.5 * np.sum((np.sqrt(H_1) - np.sqrt(H_2)) ** 2))
    else:
        print("Invalid measure method")
    return result

# c
# to vse bi lahko slo v loop
def c():
    n_bins = 8
    I_1 = imread_cv2("dataset/object_01_1.png", "float64", max=1)
    I_2 = imread_cv2("dataset/object_02_1.png", "float64", max=1)
    I_3 = imread_cv2("dataset/object_03_1.png", "float64", max=1)
    H_1 = myhist3(I_1, n_bins).reshape(-1, order='F')
    H_2 = myhist3(I_2, n_bins).reshape(-1, order='F')
    H_3 = myhist3(I_3, n_bins).reshape(-1, order='F')
    imshow(I_1, (2, 3, 1), "Image 1")
    imshow(I_2, (2, 3, 2), "Image 2")
    imshow(I_3, (2, 3, 3), "Image 3")
    l2_1_1 = compare_histograms(H_1, H_1, "l2")
    l2_1_2 = compare_histograms(H_1, H_2, "l2")
    l2_1_3 = compare_histograms(H_1, H_3, "l2")
    chi_1_1 = compare_histograms(H_1, H_1, "chi-square")
    chi_1_2 = compare_histograms(H_1, H_2, "chi-square")
    chi_1_3 = compare_histograms(H_1, H_3, "chi-square")
    int_1_1 = compare_histograms(H_1, H_1, "intersection")
    int_1_2 = compare_histograms(H_1, H_2, "intersection")
    int_1_3 = compare_histograms(H_1, H_3, "intersection")
    hel_1_1 = compare_histograms(H_1, H_1, "hellinger")
    hel_1_2 = compare_histograms(H_1, H_2, "hellinger")
    hel_1_3 = compare_histograms(H_1, H_3, "hellinger")
    print(f"L2:\n\t1: {l2_1_1}\n\t2: {l2_1_2}\n\t3: {l2_1_3}")
    print(f"Chi-square:\n\t1: {chi_1_1}\n\t2: {chi_1_2}\n\t3: {chi_1_3}")
    print(f"Intersection:\n\t1: {int_1_1}\n\t2: {int_1_2}\n\t3: {int_1_3}")
    print(f"Hellinger:\n\t1: {hel_1_1}\n\t2: {hel_1_2}\n\t3: {hel_1_3}")
    hist_show(H_1, n_bins ** 3, (2, 3, 4), f"f2(h1,h1) = {np.round(l2_1_1, 2):.2f}")
    hist_show(H_2, n_bins ** 3, (2, 3, 5), f"f2(h1,h2) = {np.round(l2_1_2, 2):.2f}")
    hist_show(H_3, n_bins ** 3, (2, 3, 6), f"f2(h1,h3) = {np.round(l2_1_3, 2):.2f}")
    plt.show()

    # Q: Which image (object_02_1.png or object_03_1.png) is more similar
    # to image object_01_1.png considering the L2 distance? How about the other three
    # distances? We can see that all three histograms contain a strongly expressed component
    # (one bin has a much higher value than the others). Which color does this
    # bin represent?
    # A: Considering L2 distance, Image 1 is more similar to Image 3,
    # since the Euclidian distance between them is shorter. 
    # In this case, other distances tell us the same.
    # The bin that is strongly expressed in all three histograms, represents the darkest pixels, i.e. black.

# d sort and display distance graphs before and after sorting
def d_sort(data, n_bins, method):
    n_data = len(data)

    # copy unsorted array
    method_dists = np.copy(data[method])    # not sorted
    
    # sort data
    data.sort(order=method)

    # make figure
    fig_1 = plt.figure(num=1, figsize=(5, 10))
    gs_1 = gs.GridSpec(nrows=1, ncols=2)

    # plot unsorted dists
    sp_1 = fig_1.add_subplot(gs_1[0])
    sp_1.plot(range(n_data), method_dists)
    for i in range(n_data):
        if method_dists[i] in data[method][:5]:
            sp_1.plot(i, method_dists[i], "o", color="orange", markerfacecolor="none")
    
    # plot sorted dists
    sp_2 = fig_1.add_subplot(gs_1[1])
    sp_2.plot(range(n_data), data[method])
    for i in range(5):
        sp_2.plot(i, data[method][i], "o", color="orange", markerfacecolor="none")

    # display images, their hists and distances for a given method
    fig_2 = plt.figure(num=2, figsize=(5, 15))
    gs_2 = gs.GridSpec(nrows=2, ncols=6)
    for i in range(6):
        dist = data[i][method]
        sp_im = fig_2.add_subplot(gs_2[0, i])
        sp_im.imshow(data[i]["image"])
        sp_im.set_title(data[i]["name"])
        sp_h = fig_2.add_subplot(gs_2[1, i])
        sp_h.bar(range(n_bins ** 3), data[i]["hist"], width=3)
        sp_h.set_title(f"dist={np.round(dist, 2):.2f}")
    plt.show(block=True)

# d
def d(path: str, n_bins: int):
    if not path.endswith("/"):
        path += "/"
    # list files
    dir = np.asarray(os.listdir(path))
    n_files = len(dir)

    # make data type for our use and a data structure
    data_el = np.dtype([("image", np.ndarray), ("hist", np.ndarray), ("l2", np.float64), ("chi-square", np.float64), ("intersection", np.float64), ("hellinger", np.float64), ("name", np.str_, 15)])
    data = np.empty(n_files, dtype=data_el)
    
    # fill the data structure with images, names and hists
    for i in range(n_files):
        filename = dir[i]
        I = imread_cv2(path + filename, "float64", max=1)
        H = myhist3(I, n_bins).reshape(-1, order='F')
        el = [(I, H, None, None, None, None, filename)]
        el = np.array(el, dtype=data_el).astype(data_el)
        data[i] = el

    # calc all distances for all hists for a given image
    image = 19
    image_hist = data[image]["hist"]
    for i in range(n_files):
        hist = data[i]["hist"]
        data[i]["l2"] = compare_histograms(image_hist, hist, "l2")
        data[i]["chi-square"] = compare_histograms(image_hist, hist, "chi-square")
        data[i]["intersection"] = compare_histograms(image_hist, hist, "intersection")
        data[i]["hellinger"] = compare_histograms(image_hist, hist, "hellinger")
    
    # sort and display
    d_sort(data, n_bins, "hellinger")
    # d_sort(data, n_bins, "l2")
    # d_sort(data, n_bins, "intersection")
    # d_sort(data, n_bins, "chi-square")

    # Q: Which distance is in your opinion best suited for image retrieval?
    # A: Hellinger.
    # Q: How does the retrieved sequence change if you use different number of bins? 
    # Is the execution time affected by the number of bins?
    # A: More bins -> longer execution time. Distances also change,
    # because with less bins, you group more colours together, meaning it computes different values

if __name__ == "__main__":
    # c()
    d("dataset/", 8)
