#! "d:/FRI/3_letnik/ai/uz/Scripts/python.exe"

"""
Before the first run, you need to have all necessary Python packages installed. For
that we highly recommend firstly creating Virtual Environment, to have your
development environment seperated from other projects (https://docs.python.org/3/tutorial/venv.html).

In system terminal then run: "pip install numpy opencv-python matplotlib Pillow"
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image


"""
path: str
data_type: str of one of two values: float64 or uint8
max: takes only 1 or 255, must be specified when working with floats
"""
def imread(path: str, data_type: str, max: int | None) -> np.ndarray:
    I = cv2.imread(path, 1)
    # I = cv2.imread(path)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    I = np.asarray(I)
    if data_type == "float64":
        I = I.astype(np.float64)
        if not max == 1 and not max == 255:
            print("Invalid max") 
        if max == 1:
            I = I / 255
    elif data_type == "uint8":
        I = I.astype(np.uint8)
    else:
        print("invalid argument: data_type")
    return I


"""
path: str
data_type: str of one of two values: float64 or uint8
"""
def imread_gray(path: str, data_type: str) -> np.ndarray:
    I = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    I = np.asarray(I)
    if data_type == "float64":
        I = I.astype(np.float64) / 255
        # I = I.astype(np.float64)
    elif data_type == "uint8":
        I = I.astype(np.uint8)
    else:
        print("invalid argument: data_type")
    return I


def imshow(img:np.ndarray, plot=None, title=None, cbar=True, c_map=None) -> None:
    """
    Shows an image. Image can be of types:
    - type uint8, in range [0, 255]
    - type float, in range [0, 1]
    """

    if plot is not None:
        plt.subplot(plot[0], plot[1], plot[2])

    if len(img.shape) == 3:
        plt.imshow(img)  # if type of data is "float", then values have to be in [0, 1]
    else:
        plt.imshow(img, cmap="gray")
        #plt.set_cmap('gray')  # also "hot", "nipy_spectral"
        if cbar is None or cbar == True:
            plt.colorbar()
    
    if title is not None:
        plt.title(title)
    
    if c_map is not None:
        plt.set_cmap(c_map)

    # plt.show()

def hist_show(H: np.ndarray, n_bins: int, plot=None, title=None): 
    if plot is not None:
        plt.subplot(plot[0], plot[1], plot[2])
    plt.bar(range(n_bins), H, width=3)
    if title is not None:
        plt.title(title)

def signal_show(*signals):
    """
    Plots all given 1D signals in the same plot.
    Signals can be Python lists or 1D numpy array.
    """
    for s in signals:
        if type(s) == np.ndarray:
            s = s.squeeze()
        plt.plot(s)
    plt.show()


def convolve(I: np.ndarray, *ks):
    """
    Convolves input image I with all given kernels.

    :param I: Image, should be of type float64 and scaled from 0 to 1.
    :param ks: 2D Kernels
    :return: Image convolved with all kernels.
    """
    for k in ks:
        k = np.flip(k)  # filter2D performs correlation, so flipping is necessary
        I = cv2.filter2D(I, cv2.CV_64F, k)
    return I


if __name__ == '__main__':  # False if this file is imported to another file and executed from there.

    # For this to work you need to put an image "image.jpg" to where you run the script.

    # Read an image in rgb (check the code in "UZ_utils" for details).
    I = imread('image.jpg')  # see the definition above
    imshow(I, 'rgb')
    print(I.shape, I.dtype, np.max(I))

    # gray
    I = imread_gray('image.jpg')  # see the definition above
    imshow(I, 'gray')
    print(I.shape, I.dtype, np.max(I))

    # Image filtering.
    I = np.zeros((20, 20), dtype=float)
    I[5, 5] = 1
    k = np.array([[1, 2, 3, 4, 5]], dtype=float)
    I = convolve(I, k, k.T)  # see the definition above
    imshow(I)

    # Plotting 1D signal.
    x = np.arange(0, 10, .1)  # From 0 to 10 with 0.1 step.
    y = np.sin(x)
    signal_show(y, 2 * y)  # see the definition above
