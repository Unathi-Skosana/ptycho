"""
File: formatting.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""

import numpy as np


def crop_center(img, crop_x, crop_y):
    """
    Crops image from center to new size

    Args:
        img: input image
        width: width of cropped image
        height: height of cropped image
    Returns:
        Cropped image
    """

    height, width = img.shape
    startx = width // 2 - (crop_y // 2)
    starty = height // 2 - (crop_x // 2)

    return img[starty:starty + crop_x, startx:startx + crop_y]


def pad_with(img, pad_width, iaxis, kwargs):
    """
    Pads image with pad_value

    Args:
        img: input image
        pad_width: pad width
        iaxis: padding axis
        kwargs: arbitrary keyword arguments
    Returns:
        Padded image
    """

    pad_value = kwargs.get('padder', 0)

    img_cpy = np.copy(img)
    img_cpy[:pad_width[0]] = pad_value
    img_cpy[-pad_width[1]:] = pad_value

    return img_cpy


def flatten(arr):
    """
    Flattens array by one dimension

    Returns:
        Flatten array
    """

    return np.array([item for sl in arr for item in sl])


def vectorize_image(img):
    """
    Reshapes an array representing an image into a column
    vector

    Args:
        img: input image

    Returns:
        Column vector representing image
    """

    height, width = img.shape

    return flatten(img.reshape((height*width, 1)))


def bmatrix(arr):
    """
    Converts a numpy array (matrix) to a LaTeX bmatrix

    Args:
        arr: Array
    Returns:
        LaTeX bmatrix as a string
    Raises:
        ValueError: If the array has more than two dimensions
    """

    if len(arr.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')

    lines = str(arr).replace('[', '').replace(']', '').splitlines()
    latex_str = [r'\begin{bmatrix}']
    latex_str += ['  ' + ' & '.join(line.split()) + r'\\' for line in lines]
    latex_str += [r'\end{bmatrix}']

    return '\n'.join(latex_str)
