#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

from UZ_utils import *
import numpy as np
# import cv2
from matplotlib import pyplot as plt

# a

I = imread("./images/umbrellas.jpg")
h, w, c = I.shape
print(h, w, c, I.dtype)
# plt.subplot(3, 2, 1)
imshow(I, (3, 2, 1), "Original Image")

# b 

# iteratively
# I_1 = np.copy(I)
# for i in range(len(I_1)):
#     for j in range(len(I_1[i])):
#         gray = 0.
#         for c in range(3):
#             gray += I_1[i, j, c]
#         gray = gray/3
#         I_1[i, j] = np.array([gray, gray, gray])    
# imshow(I_1)

# vector
I_2 = np.copy(I)
red_c = I_2[:,:,0]
green_c = I_2[:,:,1]
blue_c = I_2[:,:,2]
gray = (red_c + green_c + blue_c) / 3
I_2 = gray.reshape((h, w))  # not needed
imshow(I_2, (3, 2, 2), "Grayscale")

# c

cutout = I[130:260, 240:450, 1]
plt.subplot(3, 2, 3)
imshow(cutout, (3, 2, 3), "cutout")

# pyplot defaults to viridis color map, but we use the grayscale
# Q: Why do we use different color maps?
# A: Sometimes to emphasize values that are not normal
# different color maps are interpolating colors differently


# d

I_3 = np.copy(I)
I_3[130:260, 240:450, :] = 1 - I_3[130:260, 240:450, :]
imshow(I_3, (3, 2, 4), "inverted")

# Q: How is inverting a grayscale value defined for uint8?
# A: uint8 -> [0, 255] -> substract pixel values from 255

# e

I_2 = I_2 * 63
I_2 = I_2.astype(np.uint8)
imshow(I_2, (3, 2, 5), "grayscale uint8")

plt.subplot(3, 2, 6)
plt.imshow(I_2, vmax=255, cmap="gray")
# ^ basically same as I_2 * 4, bacause 63*4 = 252 (not same as 255 levels, but looks the same)
plt.colorbar()
plt.title("modified vmax")


plt.show()
