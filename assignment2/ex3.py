#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

from UZ_utils_a2 import *
from a2_utils import *
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from ex2 import gauss

# a
def gaussfilter(I: np.ndarray, std: float) -> np.ndarray:
    F = np.copy(I)
    k = gauss(std)
    k = k.reshape([1, k.shape[0]])
    F = cv2.filter2D(F, -1, k)
    F = cv2.filter2D(F, -1, k.T)
    return F

def test_a():
    I = imread_gray_cv2("images/lena.png", "uint8")
    I_g = np.copy(I)
    I_g = gauss_noise(I_g)
    I_sp = sp_noise(I, max=255)
    I_g_f = gaussfilter(I_g, 1)
    I_sp_f = gaussfilter(I_sp, 1)
    fig = plt.figure(figsize=(10, 15))
    grid = gs.GridSpec(nrows=2, ncols=3)
    images = [I, I_g, I_sp]
    images_names = ["Original", "Gaussian Noise", "Salt and Pepper"]
    # og and noised
    for i in range(len(images)):
        sp = fig.add_subplot(grid[0, i])
        sp.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        sp.set_title(images_names[i])
    # filtered
    filtered = [I_g_f, I_sp_f]
    filtered_names = ["Filtered Gaussian Noise", "Filtered Salt and Pepper"]
    for i in range(len(filtered)):
        sp = fig.add_subplot(grid[1, i + 1])
        sp.imshow(filtered[i], cmap="gray", vmin=0, vmax=255)
        sp.set_title(filtered_names[i])
    plt.show()

    # Q: Which noise is better removed using the Gaussian filter?
    # A: Gaussian.

# b image sharpening
def b():
    filter = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - (1/9) * np.ones([3, 3])
    image = imread_gray_cv2("images/museum.jpg", "uint8")
    sharpened = cv2.filter2D(image, -1, filter)

    fig = plt.figure(figsize=(5, 10))
    grid = gs.GridSpec(nrows=1, ncols=2)
    sp_1 = fig.add_subplot(grid[0])
    sp_1.imshow(image, cmap="gray")
    sp_1.set_title("Original")
    sp_2 = fig.add_subplot(grid[1])
    sp_2.imshow(sharpened, cmap="gray")
    sp_2.set_title("Sharpened")
    plt.show()

# c median
# I: input signal -> has to be 1D
# w: filter width, has to be an odd number
def simple_median(I: np.ndarray, w: int) -> np.ndarray:
    if not len(I.shape) == 1:
        print("Invalid I dimensions")
        return I
    F = np.copy(I)
    w = w // 2
    # F = np.pad(F, pad_width=w)
    for i in range(w, len(F) - w):
        F[i] = np.median(I[i - w:i + w + 1])
    return F

# test median
def test_c():
    signal = np.zeros(40)
    signal[11:20] = 1
    signal_sp = sp_noise_1d(signal, max=3)
    gauss_k = gauss(2)
    signal_sp_gauss = cv2.filter2D(signal_sp, -1, gauss_k)
    # if width is set to 1, it wont work. It has to be at least 2 
    signal_sp_median = simple_median(signal_sp, 2)

    plots = [signal, signal_sp, signal_sp_gauss, signal_sp_median]
    titles = ["Original", "Corrupted", "Gauss", "Median"]
    fig = plt.figure(figsize=(5, 20))
    grid = gs.GridSpec(nrows=1, ncols=4)
    for i in range(len(plots)):
        sp = fig.add_subplot(grid[i])
        sp.plot(range(len(plots[i])), plots[i])
        sp.set_ylim(0, 4)
        sp.set_title(titles[i])
    plt.show()

    # Gaussian filter takes average from the filter's window, which can change a lot
    # if there is SP noise. That doesn't happen if we are using median filter instead
    # of Gaussian.

    # Q: Which filter performs better at this specific task? 
    # A: Median is better. Reason: Look 5 lines above.
    # In comparison to Gaussian filter that can be applied multiple times in any
    # order, does the order matter in case of median filter? 
    # A: Yes.
    # Q: What is the name of filters like this?
    # A: Nonlinear filters.

if __name__ == "__main__":
    # test_a()
    # b()
    test_c()