#!/usr/bin/python3

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image

import copy
import os
from typing import Any, List, Tuple

def resize_with_PIL(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    img = to_PIL_image(img, scale_to_255=True)
    img = img.resize(size)
    img = to_np_array(img)
    return img


def to_np_array(img: Image, downscale_by_255: bool = True) -> np.ndarray:
    img = np.asarray(img)
    img = img.astype(np.float32)
    if downscale_by_255:
        img /= 255
    return img


def vis_image_scales_numpy(image: np.ndarray) -> np.ndarray:
    original_height = image.shape[0]
    original_width = image.shape[1]
    num_colors = 1 if image.ndim == 2 else 3
    img_scales = np.copy(image)
    cur_image = np.copy(image)

    scales = 5
    scale_factor = 0.5
    padding = 5

    new_h = original_height
    new_w = original_width

    for scale in range(2, scales + 1):
        # add padding
        img_scales = np.hstack(
            (
                img_scales,
                np.ones((original_height, padding, num_colors), dtype=np.float32),
            )
        )

        new_h = int(scale_factor * new_h)
        new_w = int(scale_factor * new_w)
        # downsample image iteratively
        cur_image = resize_with_PIL(cur_image, size=(new_w, new_h))

        # pad the top to append to the output
        h_pad = original_height - cur_image.shape[0]
        pad = np.ones((h_pad, cur_image.shape[1], num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        img_scales = np.hstack((img_scales, tmp))

    return img_scales


def im2single(im: np.ndarray) -> np.ndarray:
    im = im.astype(np.float32) / 255
    return im


def single2im(im: np.ndarray) -> np.ndarray:
    im *= 255
    im = im.astype(np.uint8)
    return im


def to_PIL_image(img: np.ndarray, scale_to_255: False) -> PIL.Image:
    if scale_to_255:
        img *= 255
    return PIL.Image.fromarray(np.uint8(img))


def load_image(path: str) -> np.ndarray:
    img = PIL.Image.open(path)
    img = np.asarray(img, dtype=float)
    float_img_rgb = im2single(img)
    return float_img_rgb


def save_image(path: str, im: np.ndarray) -> None:
    folder_path = os.path.split(path)[0]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    img = copy.deepcopy(im)
    img = single2im(img)
    pil_img = to_PIL_image(img, scale_to_255=False)
    pil_img.save(path)
