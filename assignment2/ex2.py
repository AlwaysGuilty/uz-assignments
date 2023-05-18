#! "D:/FRI/3_letnik/ai/uz/Scripts/python.exe"

from UZ_utils_a2 import *
from a2_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs

# b
def simple_convolution(I: np.ndarray, k: np.ndarray) -> np.ndarray:
    if not len(I.shape) == 1 and not len(k.shape) == 1:
        print("Input arrays are not 1D")
        return I
    mod = np.mod(len(k), 2)
    if not mod:
        print("k is not of size 2N + 1")
        return I
    N = len(k) // 2
    # k = k[::-1]     # flip kernel
    k = np.flip(k)
    C = np.copy(I)
    for i in range(N, len(C) - N):
        C[i] = np.sum(I[i - N:i + N + 1] * k)
    return C

def test_b():
    signal = read_data("signal.txt")
    kernel = read_data("kernel.txt")
    print(f"Sum of kernel elements: {np.sum(kernel)}")
    c = simple_convolution(signal, kernel)
    kernel = np.flip(kernel)
    c_cv2 = cv2.filter2D(signal, cv2.CV_64F, kernel)
    data = [signal, kernel, c, c_cv2]
    labels = ["Original", "Kernel", "Result", "cv2"]
    fig = plt.figure(figsize=(5, 5))
    for i in range(len(data)):
        plt.plot(range(len(data[i])), data[i], label=labels[i])
    plt.legend()
    plt.show()

    # Q: Can you recognize the shape of the kernel? 
    # A: Given that it smoothenes the image, I can say that it is a Gaussian.
    # Q: What is the sum of the elements in the kernel? 
    # A: Effectively 1 -> It is normalized
    # Q: How does the kernel affect the signal?
    # A: It smoothenes it (if it is Gaussian). To figure out what it really does,
    # we need to look into the frequency spectrum of the kernel.

# d gaussian kernel
def gauss(std: float) -> np.ndarray:
    std3 = np.ceil(3 * std)
    G = np.arange(-std3, std3 + 1)
    G = (1.0 / (np.sqrt(2 * np.pi) * std)) * np.exp(-(G ** 2) / (2 * (std ** 2)))
    G = G / np.sum(G)
    return G

def test_d():
    stds = [0.5, 1, 2, 3, 4]
    fig = plt.figure(figsize=(5, 5))
    for std in stds:
        start = - int(np.ceil(3 * std))
        stop = int(np.ceil(3 * std)) + 1
        plt.plot(np.arange(start, stop), gauss(std), label=f"sigma = {std}")
    plt.legend()
    plt.show()

# e
def e():
    signal = read_data("signal.txt")
    k1 = gauss(2.0)
    k2 = np.array([0.1, 0.6, 0.4])
    k3 = simple_convolution(k1, k2)
    res1 = simple_convolution(signal, k1)
    res1 = simple_convolution(res1, k2)
    res2 = simple_convolution(signal, k2)
    res2 = simple_convolution(res2, k1)
    res3 = simple_convolution(signal, k3)
    results = [signal, res1, res2, res3]
    names = ["s", "(s * k1) * k2", "(s * k2) * k1", "s * (k1 * k2)"]
    fig = plt.figure(figsize=(5, 25))
    grid = gs.GridSpec(nrows=1, ncols=4)
    for i in range(4):
        sp = fig.add_subplot(grid[i])
        sp.plot(range(len(results[i])), results[i])
        sp.set_title(names[i])
    plt.show()

    # As per the results, order of operations doesn't matter,
    # bacause of the convolution's properties, more specifically,
    # it is associative. (f * g) * h = f * (g * h)

if __name__ == "__main__":
    # test_b()
    # test_d()
    e()