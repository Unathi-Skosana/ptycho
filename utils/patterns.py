"""
File: patterns.py
Author: Unathi Skosana
Email: ukskosana@gmail.com
Github: https://github.com/Unathi Skosana
Description: 
"""

import numpy as np


def get_gradation_2d(width=256, height=256, start=0,
                     stop=255, is_horizontal=False):
    """
    Creates an image that gradually changes color (single channel)
    horizontally/vertically

    Source: https://note.nkmk.me/en/python-numpy-generate-gradation-image

    Args:
        width: width of image
        height: height of image
        start: 8-bit color value at the start of the gradient
        stop: 8-bit color value at the end of the gradient
        is_horizontal: direction of the gradient
    Returns:
        Gradient image
    """

    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))

    return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradation_3d(width, height, start_list, stop_list, is_horizontal_list):
    """
    Creates an image that gradually changes color (three channels)
    horizontally/vertically

    Source: https://note.nkmk.me/en/python-numpy-generate-gradation-image

    Args:
        width: width of image
        height: height of image
        start: 8-bit color values (RGB) at the start of the gradient
        stop: 8-bit color values (RGB) at the end of the gradient
        is_horizontal: direction of the gradient in each channel
    Returns:
        Gradient image
    """

    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    for i, (start, stop, is_horizontal) \
            in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = get_gradation_2d(width, height, start, stop,
                                           is_horizontal)

    return result


def stripes(gap, stripe_width, width, height, horizontal=True):
    """
    Creates black & white stripped patterns horizontally/vertically

    Args:
        gap: gap between stripes
        stripe_width: stripe width
        width: width of image
        height: height of image
        is_horizontal: direction of stripes
    Returns:
        Stripped image
    """

    canvas = np.full((width, height), 0.5, dtype=np.float64)
    current_col = 0

    while current_col < width:
        if current_col + stripe_width + gap <= width-1:
            if horizontal:
                canvas[current_col:current_col+stripe_width, :] = 1.0
            else:
                canvas[:, current_col:current_col+stripe_width] = 1.0
            current_col += stripe_width + gap
        elif current_col + stripe_width <= width-1:
            if horizontal:
                canvas[current_col:current_col+stripe_width, :] = 1.0
            else:
                canvas[:, current_col:current_col+stripe_width] = 1.0
            current_col = width
        else:
            if horizontal:
                canvas[current_col:, :] = 1
            else:
                canvas[:, current_col:] = 1
            current_col = width
    return canvas


def checkerboard(board_sz, square_sz):
    """
    Creates black & white checkerboard

    Source:
        https://stackoverflow.com/questions/32704485/drawing-a-checkerboard-in-python

    Args:
        board_sz: size of board
        square_sz: size of each square
    Returns:
        Checkerboard image
    Raises:
        ValueError: If board_sz % (2*board_sz) is not an integer
    """

    if board_sz % (2*square_sz):
        raise ValueError('board_sz % (2*square_sz) must be an integer')

    zeros_ones = np.concatenate((np.zeros(square_sz), np.ones(square_sz)))
    board = np.pad(zeros_ones, int((board_sz**2)/2 - square_sz), 'wrap') \
        .reshape((board_sz, board_sz))

    return (board + board.T == 1).astype(int)
