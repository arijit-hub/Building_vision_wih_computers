"""Implements the Harris corner and edge detection algorithm"""

import cv2
import numpy as np


class Harris:
    def __init__(self, img_path):
        """Constructor.

        Parameters
        ----------
        img_path : str
            Path to the image to be processed.

        """

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        self.img = img.astype(np.float32)

    def _blur(self, img, kernel_size):
        """Blurs an image using low pass Gaussian Filter.

        Parameters
        ----------
        img : np.ndarray
            Image to be processed.

        kernel_size : tuple
            Size of the Gaussian Kernel.

        Returns
        -------
        np.ndarray
            Gaussian blurred image.

        """

        return cv2.GaussianBlur(img, kernel_size, 1)

    def _calculate_gradient_squared(self, img):
        """Calculates the gradient in x direction,
        y direction and both of x and y direction and
        also squares them.

        Parameters
        ----------
        img : np.ndarray
            Image.

        Returns
        -------
        np.ndarray
            Gradient of the image in x-direction squared.

        np.ndarray
            Gradient of the image in y-direction squared.

        np.ndarray
            Gradient of the image in x multiplied by Gradient
            of the image y direction.

        """

        I_x = cv2.Sobel(img, -1, dx=1, dy=0)
        I_x_squared = I_x * I_x

        I_y = cv2.Sobel(img, -1, dx=0, dy=1)
        I_y_squared = I_y * I_y

        I_xy_squared = I_x * I_y

        return I_x_squared, I_y_squared, I_xy_squared

    def _make_cornerness_matrix(
        self, I_x_squared_blur, I_y_squared_blur, I_xy_squared_blur, alpha
    ):
        """Calculates the cornerness matrix which represents each pixel
        as a value which decides its cornerness.

        Parameters
        ----------
        I_x_squared_blur : np.ndarray
            Gradient of the image in x-direction squared.

        I_y_squared_blur : np.ndarray
            Gradient of the image in y-direction squared.

        I_xy_squared_blur : np.ndarray
            Gradient of the image in x multiplied by Gradient
            of the image y direction.

        alpha : float
            Multiplicative factor.

        Returns
        -------
        np.ndarray
            Cornerness matrix with each pixel representing the
            corner strength of each pixel.

        """
        cornerness_matrix = (
            I_x_squared_blur * I_y_squared_blur - I_xy_squared_blur * I_xy_squared_blur
        ) - alpha * (I_x_squared_blur + I_y_squared_blur) ** 2

        return cornerness_matrix

    def _get_corner_img(self, cornerness_matrix, threshold):
        """Forms a corner image matrix with white values
        at the corner location.

        Parameters
        ----------
        cornerness_matrix : np.ndarray
            Cornerness matrix with each pixel representing the
            corner strength of each pixel.

        threshold : float
            Threshold value to select if a pixel intensity
            should be considered a corner.

        Returns
        -------
        np.ndarray
            Corner image matrix with white values
            at the corner location
        """
        h, w = cornerness_matrix.shape
        corner_img = np.zeros((h, w))

        for j in range(h):
            for i in range(w):
                if cornerness_matrix[j, i] > threshold:
                    try:
                        if (
                            (cornerness_matrix[j, i] >= cornerness_matrix[j, i + 1])
                            and (
                                cornerness_matrix[j, i]
                                >= cornerness_matrix[j - 1, i + 1]
                            )
                            and (cornerness_matrix[j, i] >= cornerness_matrix[j - 1, i])
                            and (
                                cornerness_matrix[j, i]
                                >= cornerness_matrix[j - 1, i - 1]
                            )
                            and (cornerness_matrix[j, i] >= cornerness_matrix[j, i - 1])
                            and (
                                cornerness_matrix[j, i]
                                >= cornerness_matrix[j + 1, i - 1]
                            )
                            and (cornerness_matrix[j, i] >= cornerness_matrix[j + 1, i])
                            and (
                                cornerness_matrix[j, i]
                                >= cornerness_matrix[j + 1, i + 1]
                            )
                        ):
                            corner_img[j, i] = 255
                    except IndexError:
                        continue

        return corner_img

    def __call__(self, kernel_size, alpha=0.06, threshold=0.1):
        """Calling method.

        Parameters
        ----------
        kernel_size : tuple
            Kernel shape for the blur functionality.

        alpha : float
            Multiplicative factor.

        threshold : float
            Threshold value to select if a pixel intensity
            should be considered a corner.
        """
        img = self.img.copy()
        I_x_squared, I_y_squared, I_xy_squared = self._calculate_gradient_squared(img)
        I_x_squared_blur, I_y_squared_blur, I_xy_squared_blur = (
            self._blur(I_x_squared, kernel_size),
            self._blur(I_y_squared, kernel_size),
            self._blur(I_xy_squared, kernel_size),
        )

        R = self._make_cornerness_matrix(
            I_x_squared_blur, I_y_squared_blur, I_xy_squared_blur, alpha=alpha
        )
        corner_img = self._get_corner_img(R, threshold)

        cv2.imwrite("harris_corner_edge_detection/output/corner_img.png", corner_img)


if __name__ == "__main__":
    detector = Harris("harris_corner_edge_detection/lenna.png")
    detector((3, 3))
