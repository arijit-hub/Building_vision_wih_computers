"""Building Box and Gaussian filters."""

## For the purpose of vis we are going to use the picture of ...#
# Lenna, stored as "lenna.png" in our storage. ##

import cv2
import numpy as np


def box_filter(shape=(3, 3)):
    """Builds a box filter given shape.

    Parameters
    ----------
    shape : tuple [Default:(3,3)]
        Shape of the returned box filter.
        A point to remember is that since,
        opencv coordinates work as [row,column]
        but the shape is given as [width,height],
        they must be reversed when building.

    Returns
    -------
    np.ndarray
        Box filter.

    """

    box_filter_matrix = (1 / (shape[0] * shape[1])) * np.ones(shape=shape)

    return box_filter_matrix


def gaussian_filter(shape=(3, 3), std=1):
    """Builds a Gaussian Kernel given shape and std.

    Parameters
    ----------
    shape : tuple [Default:(3,3)]
        Shape of the returned gaussian filter.

    std : num (float or int)
        Standard deviation for the gaussian filter.

    Returns
    -------
    np.ndarray
        Gaussian filter.

    """
    center_x, center_y = shape[0] // 2, shape[1] // 2
    x, y = np.mgrid[-center_x : center_x + 1, -center_y : center_y + 1]

    gauss_filter_matrix = (1 / (2 * np.pi * std**2)) * np.exp(
        -(x**2 + y**2) / (2 * std**2)
    )
    return gauss_filter_matrix


def blur_image(img_path, filter_type="BOX", filter_shape=(3, 3), std=None):
    """Blurs and displays an image given a type of filter.

    Parameters
    ----------
    img_path : str
        Path to the image to blur.

    filter_type : str (Default : 'BOX')
        Type of filter to use to blur the image.
        Two types of filter are available.
        1. BOX and 2. GAUSSIAN.

    """

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if filter_type == "BOX":
        filter_to_use = box_filter(shape=filter_shape)
    elif filter_type == "GAUSSIAN":
        filter_to_use = gaussian_filter(shape=filter_shape, std=std)

    filtered_img = cv2.filter2D(img, ddepth=-1, kernel=filter_to_use)

    out_filename = f"""filters/output/filtered_img_{filter_type}_filter_{filter_shape[0]}{filter_shape[1]}.png"""
    cv2.imwrite(
        out_filename,
        filtered_img,
    )


## Test ##

if __name__ == "__main__":
    blur_image(
        "D:/Building_vision_wih_computers/filters/lenna.png",
        filter_type="BOX",
        filter_shape=(7, 7),
    )
    blur_image(
        "D:/Building_vision_wih_computers/filters/lenna.png",
        filter_type="GAUSSIAN",
        filter_shape=(7, 7),
        std=2,
    )
