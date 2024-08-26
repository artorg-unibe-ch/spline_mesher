import unittest
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from pyhexspline.spline_volume import OCC_volume


class TestOCCVolume(unittest.TestCase):
    def setUp(self):
        # Initialize an instance of OCC_volume with dummy parameters
        self.volume = OCC_volume(
            sitk_image=sitk.Image(),
            img_path=None,
            filepath="",
            filename="",
            ASPECT=1,
            SLICE=1,
            UNDERSAMPLING=1,
            SLICING_COEFFICIENT=1,
            INSIDE_VAL=1.0,
            OUTSIDE_VAL=1.0,
            LOWER_THRESH=1.0,
            UPPER_THRESH=1.0,
            S=1,
            K=1,
            INTERP_POINTS=1,
            debug_orientation=1,
            show_plots=False,
            thickness_tol=1.0,
            phases=1,
        )

    def test_plot_mhd_slice(self):
        self.volume.plot_mhd_slice()

    def test_plot_slice(self):
        self.volume.plot_slice(sitk.Image(), 1, "title", 1)

    def test_exec_thresholding(self):
        # Create a test image
        image = sitk.Image([10, 10], sitk.sitkUInt8)
        image.Fill(5)
        # Execute thresholding
        result = self.volume.exec_thresholding(image, [0, 255, 0, 10])
        # Check that the result is as expected
        self.assertEqual(sitk.GetArrayFromImage(result).all(), np.ones((10, 10)).all())

    def test_draw_contours(self):
        # Create a test image
        img = np.zeros((10, 10), dtype=np.uint8)
        img[3:7, 3:7] = 1
        # Draw contours
        result = self.volume.draw_contours(img, "outer", True)
        # Check that the result is as expected
        self.assertEqual(result[3:7, 3:7].all(), np.ones((4, 4)).all())

    def test_get_binary_contour(self):
        image = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        result = self.instance.get_binary_contour(image)
        self.assertIsInstance(result, sitk.Image)

    def test_get_draw_contour(self):
        image = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        result = self.instance.get_draw_contour(image, "outer")
        self.assertIsInstance(result, np.ndarray)

    def test_pad_image(self):
        image = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        result = self.instance.pad_image(image, 5)
        self.assertIsInstance(result, sitk.Image)

    def test_binary_threshold(self):
        img_path = "test_image.nii"  # replace with your actual test image path
        result = self.instance.binary_threshold(img_path)
        self.assertIsInstance(result, Tuple)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_sort_xy(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = self.instance.sort_xy(x, y)
        self.assertIsInstance(result, Tuple)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_check_orient(self):
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])
        x_o, y_o = self.spline_volume.check_orient(x, y, direction=1)
        self.assertEqual(x_o.tolist(), [3, 2, 1])
        self.assertEqual(y_o.tolist(), [1, 2, 3])

    def test_sort_mahalanobis(self):
        data = np.array([[1, 2, 3], [3, 2, 1]])
        x_sorted, y_sorted = self.spline_volume.sort_mahalanobis(data)
        self.assertEqual(x_sorted.tolist(), [1, 2, 3])
        self.assertEqual(y_sorted.tolist(), [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
