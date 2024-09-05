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
            DP_SIMPLIFICATION_OUTER=1,
            DP_SIMPLIFICATION_INNER=1,
            debug_orientation=1,
            show_plots=False,
            thickness_tol=1.0,
            phases=1,
        )

    def test_exec_thresholding(self):
        # Create a test image
        image = sitk.Image([10, 10], sitk.sitkUInt8)
        image = sitk.Cast(image, sitk.sitkUInt8)
        image = sitk.Add(image, 5)
        # Execute thresholding
        result = self.volume.exec_thresholding(image, [0, 255, 0, 10])
        # Check that the result is as expected
        expected_result = sitk.BinaryThreshold(
            image, lowerThreshold=0, upperThreshold=0.5, insideValue=1, outsideValue=0
        )
        self.assertTrue(
            np.array_equal(
                sitk.GetArrayFromImage(result), sitk.GetArrayFromImage(expected_result)
            )
        )

    def test_draw_contours(self):
        # Create a test image
        img = np.zeros((10, 10), dtype=np.uint8)
        img[3:7, 3:7] = 1
        # Draw contours
        result = self.volume.draw_contours(img, "outer", True)
        # Check that the result is as expected
        expected_result = np.zeros((10, 10), dtype=np.uint8)
        expected_result[3:7, 3:7] = 1
        expected_result[4, 4] = 0
        expected_result[5, 4] = 0
        expected_result[4, 5] = 0
        expected_result[5, 5] = 0
        print("Result:\n", result)
        print("Expected:\n", expected_result)
        self.assertTrue(np.array_equal(result, expected_result))

    def test_get_draw_contour(self):
        # Ensure the input image is not empty
        image = sitk.Image([10, 10, 10], sitk.sitkUInt8)
        # Fill the image with some data to avoid empty sequence error
        image = sitk.Add(image, 1)
        result = self.volume.get_draw_contour(image, "outer")
        self.assertIsInstance(result, np.ndarray)

    def test_binary_threshold(self):
        np_array = np.random.randint(0, 256, (10, 10, 10))
        sitk_image = sitk.GetImageFromArray(np_array)
        result = self.volume.pad_and_plot(sitk_image)
        result = (sitk.GetArrayFromImage(result[0]), sitk.GetArrayFromImage(result[1]))
        self.assertIsInstance(result, Tuple)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_sort_xy(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        result = self.volume.sort_xy(x, y)
        self.assertIsInstance(result, Tuple)
        self.assertIsInstance(result[0], np.ndarray)
        self.assertIsInstance(result[1], np.ndarray)

    def test_check_orient(self):
        x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])
        x_o, y_o = self.volume.check_orient(x, y, direction=1)
        self.assertEqual(x_o.tolist(), [3, 2, 1])
        self.assertEqual(y_o.tolist(), [1, 2, 3])

    def test_sort_mahalanobis(self):
        data = np.array([[1, 2, 3], [3, 2, 1]])
        x_sorted, y_sorted = self.volume.sort_mahalanobis(data)
        self.assertEqual(x_sorted.tolist(), [1, 2, 3])
        self.assertEqual(y_sorted.tolist(), [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
