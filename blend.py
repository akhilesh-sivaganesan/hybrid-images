#!/usr/bin/python3

from typing import Tuple

import numpy as np


def create_Gaussian_kernel_1D(ksize: int, sigma: int) -> np.ndarray:
    term1 = np.arange(ksize) - np.floor(ksize / 2)
    kernel = np.exp(-(term1) **2 / (2* sigma** 2))
    kernel /= np.sum(kernel)
    kernel = kernel.reshape(-1, 1)
    

    return kernel


def create_Gaussian_kernel_2D(cutoff_frequency: int) -> np.ndarray:
    mult = cutoff_frequency * 4 + 1
    kernel_1d = create_Gaussian_kernel_1D(mult, cutoff_frequency)
    kernel = np.outer(kernel_1d, kernel_1d)

    return kernel


def separate_Gaussian_kernel_2D(kernel: np.ndarray) -> (np.ndarray, np.ndarray):
    u, s, vh = np.linalg.svd(kernel)
    sqrt = np.sqrt(s[:1])
    v = u[:, :1] * sqrt
    h = vh[:1, :].T * sqrt

    return v, h


def my_conv2d_numpy(image: np.ndarray, filter: np.ndarray) -> np.ndarray:
    # get these shapes
    m, n, c = image.shape
    k, j = filter.shape

    pad_m, pad_n = k // 2, j // 2

    pad_d = ((pad_m, pad_m), (pad_n, pad_n), (0, 0))
    padded_image = np.pad(image, pad_d, mode='constant')
    filtered_image = np.zeros_like(image)

    # iterate trhough each pizel

    for i in range(m):
        for j_ in range(n):
            for k_ in range(c):
                filtered_image[i, j_, k_] = np.sum(
                    padded_image[i:i + k, j_:j_ + j, k_] * filter
                )

    return filtered_image


def create_hybrid_image(
    image1: np.ndarray, image2: np.ndarray, filter: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    low_frequencies = my_conv2d_numpy(image1, filter)
    high_frequencies = image2 - my_conv2d_numpy(image2, filter)

    hybrid_image = high_frequencies + low_frequencies
    hybrid_image = np.clip(hybrid_image, 0, 1)

    return low_frequencies, high_frequencies, hybrid_image