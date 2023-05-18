import numpy as np


def read_data(filename: str) -> np.ndarray:
    # reads a numpy array from a text file
    with open(filename) as f:
        s = f.read()

    return np.fromstring(s, sep=' ')

# default magnitude: .1
def gauss_noise(I: np.ndarray, magnitude=25) -> np.ndarray:
    # input: image, magnitude of noise
    # output: modified image

    return I + np.random.normal(size=I.shape) * magnitude


def sp_noise(I: np.ndarray, percent=.1, max=1) -> np.ndarray:
    # input: image, percent of corrupted pixels
    # output: modified image

    res = I.copy()

    res[np.random.rand(I.shape[0], I.shape[1]) < percent / 2] = max     # default was 1
    res[np.random.rand(I.shape[0], I.shape[1]) < percent / 2] = 0

    return res

def sp_noise_1d(I: np.ndarray, percent=.1, max=1) -> np.ndarray:
    res = I.copy()
    res[np.random.rand(I.shape[0]) < percent / 2] = max
    res[np.random.rand(I.shape[0]) < percent / 2] = 0
    return res