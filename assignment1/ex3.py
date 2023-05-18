#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

from ast import iter_child_nodes
from UZ_utils import *
import numpy as np
from matplotlib import pyplot as plt



if __name__ == "__main__":
    plt.clf()

    # a
    I = imread_gray("./images/mask.png")
    imshow(I, (3, 2, 1), "mask.png")

    n = 5
    se = np.ones((n, n), np.uint8)
    I_eroded = cv2.erode(I, se)
    I_dilated = cv2.dilate(I, se)
    imshow(I_eroded, (3, 2, 2), "eroded")
    imshow(I_dilated, (3, 2, 3), "dilated")
    I_opened = cv2.dilate(I_eroded, se)
    I_closed = cv2.erode(I_dilated, se)
    imshow(I_opened, (3, 2, 4), "opened")
    imshow(I_closed, (3, 2, 5), "closed")

    # Q: Based on the results, which order of erosion and dilation operations produces opening and which closing
    # A: opening: first erosion, then dilation, closing: first dilation, then erosion

    plt.show()
