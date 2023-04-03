"""Implementation of Canny edge detection."""

import cv2
import numpy as np


class Canny:
    """Implements the Canny edge detection algorithm."""

    def __init__(self, img_path):
        """Constructor.

        Paramters
        ---------
        img_path : str
            Path to the img to be processed.
        """
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    def _blur(self, img, kernel_size):
        """Implements Gaussian Blur on an image.

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

        return cv2.GaussianBlur(img, kernel_size, 0)  # cv2.BORDER_DEFAULT

    def _gradient_magnitude_angle(self, img):
        """Calculates gradient magnitude and phase of an
        image in x and y direction.

        Parameters
        ----------
        img : np.ndarray
            Image.

        Returns
        -------
        np.ndarray
            Magnitude of gradients.

        np.ndarray
            Angle of the gradients.

        """

        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
        I_x = cv2.filter2D(img, -1, sobel_x)
        I_y = cv2.filter2D(img, -1, sobel_y)
        magnitude = np.hypot(I_x, I_y)
        magnitude = ((magnitude - np.min(magnitude)) / np.max(magnitude)) * 255
        angle = np.arctan2(I_y, I_x) * 180 / np.pi
        return magnitude, angle

    def _non_max_suppression(self, magnitude, angle):
        """Applies non-max suppression to the magnitude
        of gradients given the respective angle. By checking
        the angle of gradient at each pixel the neighbouring
        pixels in the same angle direction are checked and if
        the current pixel has more value than the neighbours
        then it is set to 255, else it is set to 0.

        Parameters
        ----------
        magnitude : np.ndarray
            Magnitude of gradients.

        angle : np.ndarray
            Angle of the gradients.

        Returns
        -------
        np.ndarray
            Gradients after non-max suppression.

        """

        h, w = magnitude.shape
        finer_gradient_img = np.zeros((h, w))
        angle[angle < 0] += 180
        for j in range(h):
            for i in range(w):
                try:
                    neighbour_1 = 255
                    neighbour_2 = 255
                    current_pixel = magnitude[j, i]
                    current_angle = angle[j, i]
                    if (current_angle < 22.5) or (
                        current_angle >= 157.5 and current_angle < 180
                    ):
                        neighbour_1 = magnitude[j, i - 1]
                        neighbour_2 = magnitude[j, i + 1]
                    elif current_angle >= 22.5 and current_angle < 67.5:
                        neighbour_1 = magnitude[j + 1, i - 1]
                        neighbour_2 = magnitude[j - 1, i + 1]
                    elif current_angle >= 67.5 and current_angle < 112.5:
                        neighbour_1 = magnitude[j - 1, i]
                        neighbour_2 = magnitude[j + 1, i]
                    elif current_angle >= 112.5 and current_angle < 157.5:
                        neighbour_1 = magnitude[j - 1, i - 1]
                        neighbour_2 = magnitude[j + 1, i + 1]

                    if (current_pixel >= neighbour_1) and (
                        current_pixel >= neighbour_2
                    ):
                        finer_gradient_img[j, i] = magnitude[j, i]
                    else:
                        finer_gradient_img[j, i] = 0
                except IndexError:
                    continue
        return finer_gradient_img

    def _thresholding(self, finer_gradient_img, low_threshold=0.05, high_threshold=0.1):
        """Thresholds edges and defines them as strong, weak and noise.

        Parameters
        ----------
        finer_gradient_img : np.ndarray
                Gradient image after non-max suppression.

        low_threshold : float
                Lower threshold ratio.

        high_threshold : float
                Higher threshold ratio.

        Returns
        -------
        np.ndarray
            Thresholded array with each pixel denoted as noise,
            weak or strong.

        """

        h, w = finer_gradient_img.shape
        edge_type = np.zeros(shape=(h, w), dtype="object")
        high_threshold_val = np.max(finer_gradient_img) * high_threshold
        low_threshold_val = high_threshold_val * low_threshold

        for j in range(h):
            for i in range(w):
                if finer_gradient_img[j, i] < low_threshold_val:
                    edge_type[j, i] = "noise"
                elif (finer_gradient_img[j, i] >= low_threshold_val) and (
                    finer_gradient_img[j, i] < high_threshold_val
                ):
                    edge_type[j, i] = "weak"

                elif finer_gradient_img[j, i] >= high_threshold_val:
                    edge_type[j, i] = "strong"

        return edge_type

    def _hysteresis_thresholding(self, finer_gradient_img, edge_type):
        """Hysteresis thresholding utility, which keeps weak edges if and
        only if they are connected to strong edge.

        Parameters
        ----------
        finer_gradient_img : np.ndarray
                Non-max suppression gradient image.

        edge_type : np.ndarray
            Thresholded array with each pixel denoted as noise,
            weak or strong.

        Returns
        -------
        np.ndarray
            Image after removal of all noise and weak edges not
            connected to strong edges.

        """
        h, w = finer_gradient_img.shape
        edge_img = np.zeros((h, w))
        for j in range(h):
            for i in range(w):
                try:
                    if edge_type[j, i] == "weak":
                        if (
                            (edge_type[j, i + 1] == "strong")
                            or (edge_type[j - 1, i + 1] == "strong")
                            or (edge_type[j - 1, i] == "strong")
                            or (edge_type[j - 1, i - 1] == "strong")
                            or (edge_type[j, i - 1] == "strong")
                            or (edge_type[j + 1, i - 1] == "strong")
                            or (edge_type[j + 1, i] == "strong")
                            or (edge_type[j + 1, i + 1] == "strong")
                        ):
                            edge_type[j, i] = "strong"
                            edge_img[j, i] = 255
                        else:
                            edge_img[j, i] = 0
                    elif edge_type[j, i] == "noise":
                        edge_img[j, i] = 0
                    elif edge_type[j, i] == "strong":
                        edge_img[j, i] = 255

                except IndexError:
                    continue

        return edge_img

    def __call__(self, kernel_size):
        """Calling method.

        Parameters
        ----------
        kernel_size : tuple
                Kernel size of the gaussian blur filter.

        """

        img = self.img.copy()
        img = self._blur(img, kernel_size)
        grad_magnitude, grad_angle = self._gradient_magnitude_angle(img)
        finer_gradient_img = self._non_max_suppression(grad_magnitude, grad_angle)
        edge_type = self._thresholding(finer_gradient_img)
        edge_img = self._hysteresis_thresholding(finer_gradient_img, edge_type)
        finer_gradient_img = finer_gradient_img.astype("int32")
        cv2.imwrite("canny_edge_detection/output/edge_img.png", edge_img)


if __name__ == "__main__":
    canny = Canny("canny_edge_detection/lenna.png")
    canny(kernel_size=(7, 7))
