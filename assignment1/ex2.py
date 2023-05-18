#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt

def convert2grayscale(I:np.ndarray) -> np.ndarray:
    h, w, c = I.shape
    red_c = I[:,:,0]
    green_c = I[:,:,1]
    blue_c = I[:,:,2]
    gray = (red_c + green_c + blue_c) / 3
    return gray.reshape((h, w))

def mask_1(I:np.ndarray, tresh:int) -> np.ndarray:
    I_1 = np.copy(I)
    I_1[I_1 < tresh] = 0
    I_1[I_1 >= tresh] = 1
    return I_1

def mask_2(I:np.ndarray, tresh:int) -> np.ndarray:
    mask = np.copy(I)
    mask = np.where(mask < tresh, 0, 1)
    return mask

# 2 b
def myhist(I:np.ndarray, bins:int) -> np.ndarray:
    I = I.astype(np.uint8)
    I = I.reshape(-1)
    H = np.zeros(bins)
    bin_count = np.bincount(I, minlength=bins)
    interval = int(np.floor(255 / bins))
    for i in range(bins):
        H[i] += sum(bin_count[i * interval:i * interval + interval])
    return H


# morphologic operations
def open(I:np.ndarray, se:np.ndarray) -> np.ndarray:
    return cv2.dilate(cv2.erode(I, se), se)

def close(I:np.ndarray, se:np.ndarray) -> np.ndarray:
    return cv2.erode(cv2.dilate(I, se), se)


if __name__ == "__main__":
    
    plt.clf()

    # 2
    # a
    # I = imread_cv2("./images/bird.jpg")
    # I = convert2grayscale(I)
    I = imread_gray_cv2("./images/bird.jpg")
    #I = imread_gray("./images/bird.jpg")
    imshow(I, (2, 3, 1), "bird.jpg")
    I_2 = mask_1(I, 80)
    imshow(I_2, (2, 3, 2), "mask 1")
    I_3 = mask_2(I, 50)
    imshow(I_3, (2, 3, 3), "mask 2")


    # b

    bins = 255
    H = myhist(I, bins)
    plt.subplot(2, 3, 4)
    plt.bar(range(bins), H / sum(H))
    plt.title("histogram for bird.jpg")

    bins = 20
    H_1 = myhist(I, bins)
    plt.subplot(2, 3, 5)
    plt.bar(range(bins), H_1 / sum(H_1))
    plt.title("histogram for bird.jpg 20 with bins")

    # Q: The histograms are usually normalized by dividing the result by the sum of cells. Why is that?
    # A: To get the percentage of pixels that fall in a specific bin

    plt.show()


    # 3
    # a

    I = imread_gray_cv2("./images/mask.png")
    I /= 255
    plt.clf()
    imshow(I, (3, 3, 1), "mask.png")

    n = 5
    se_sq_5 = np.ones((n, n), np.uint8)     # SE square of size 5
    se_el_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))      # SE ellipse of size 5
    n = 3
    se_sq_3 = np.ones((n, n), np.uint8)                                 # SE square of size 3
    se_el_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))      # SE ellipse of size 3

    I_eroded = cv2.erode(I, se_sq_5)
    I_dilated = cv2.dilate(I, se_sq_5)
    imshow(I_eroded, (3, 3, 2), "eroded")
    imshow(I_dilated, (3, 3, 3), "dilated")
    I_opened = open(I, se_sq_5)
    I_closed = close(I, se_sq_5)
    imshow(I_opened, (3, 3, 4), "opened")
    imshow(I_closed, (3, 3, 5), "closed")

    # Q: Based on the results, which order of erosion and dilation operations produces opening and which closing
    # A: opening: first erosion, then dilation, closing: first dilation, then erosion


    # b

    mask1 = cv2.dilate(I_2.astype(np.uint8), se_sq_3)
    mask2 = cv2.dilate(I_2.astype(np.uint8), se_sq_5)
    mask3 = cv2.dilate(I_2.astype(np.uint8), se_el_3)
    mask4 = cv2.dilate(I_2.astype(np.uint8), se_el_5)
    for i in range(6):
        mask4 = cv2.dilate(mask4, se_el_5)
    for i in range(6):
        mask4 = cv2.erode(mask4, se_sq_5)

    imshow(mask1, (3, 3, 6), "dilation sq3")
    imshow(mask2, (3, 3, 7), "dilation sq5")
    imshow(mask3, (3, 3, 8), "dilation el3")
    imshow(mask4, (3, 3, 9), "final")

    plt.show()


    # d

    plt.clf()

    eagle = imread_gray_cv2("./images/eagle.jpg")
    imshow(eagle, (3, 3, 1), "eagle.jpg")
    bins = 255
    H_eagle = myhist(eagle, bins)
    plt.subplot(3, 3, 2)
    plt.bar(range(bins), H_eagle / sum(H_eagle))
    plt.title("histogram for eagle.jpg")
    
    eagle_mask = mask_2(eagle, 180)
    for i in range(3):
        eagle_mask = cv2.erode(eagle_mask.astype(np.uint8), se_sq_3)
    for i in range(3):
        eagle_mask = cv2.dilate(eagle_mask.astype(np.uint8), se_sq_3)
    eagle_mask = np.invert(eagle_mask)
    imshow(eagle_mask, (3, 3, 3), "eagle mask", cbar=False)

    # Q: Why is the background included in the mask and not the object? How
    # would you fix that in general? (just inverting the mask if necessary doesn’t count)
    # A: Here, the object is darker than the background, which is the opposite to the situation
    # we had with the bird.jpg, that's why background is white in the mask and the object is black.
    # Fix: Maybe count pixels that are < 0.5 / > 0.5 to see if the image is more dark/light and then
    # decide on how to mask 


    # e

    coins = imread_cv2("./images/coins.jpg")
    coins_gray = imread_gray_cv2("./images/coins.jpg")
    imshow(coins, (3, 3, 4), "coins.jpg")
    imshow(coins_gray, (3, 3, 5), "gray coins")
    
    H_coins = myhist(coins_gray, bins)
    plt.subplot(3, 3, 6)
    plt.bar(range(bins), H_coins / sum(H_coins))
    plt.title("histogram for coins.jpg")
    
    coins_mask = mask_2(coins_gray, 225)
    coins_mask = coins_mask.astype(np.uint8)
    coins_mask = 1 - coins_mask
    # coins_mask = cv2.erode(coins_mask, se_sq_3)
    imshow(coins_mask, (3, 3, 7), "coins mask")

    components = cv2.connectedComponentsWithStats(coins_mask)
    stats = components[2]
    for i in range(1, len(stats)):
        if stats[i, 4] > 700:
            x = stats[i, 0]
            y = stats[i, 1]
            w = stats[i, 2]
            h = stats[i, 3]
            coins[y - 10:y + h + 10, x - 10:x + w + 10, :] = 1.0
            coins_mask[y:y + h, x:x + w] = 0
    imshow(coins_mask, (3, 3, 8), "modified mask")
    imshow(coins, (3, 3, 9), "modified original image")

    plt.show()