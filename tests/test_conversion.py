"""Tests for pyhexspline.futils.conversion module."""
import unittest

import numpy as np
import SimpleITK as sitk

from pyhexspline.futils.conversion import (
    ImageProperties,
    NumpySimpleITKImageBridge,
    SimpleITKNumpyImageBridge,
)


class TestImageProperties(unittest.TestCase):
    def _make_image(self, size, pixel_type=sitk.sitkUInt8):
        return sitk.Image(size, pixel_type)

    def test_properties_3d(self):
        img = self._make_image([10, 20, 30])
        props = ImageProperties(img)
        self.assertEqual(props.size, (10, 20, 30))
        self.assertEqual(props.dimensions, 3)
        self.assertEqual(props.origin, (0.0, 0.0, 0.0))
        self.assertEqual(props.spacing, (1.0, 1.0, 1.0))

    def test_properties_2d(self):
        img = self._make_image([10, 20])
        props = ImageProperties(img)
        self.assertEqual(props.size, (10, 20))
        self.assertEqual(props.dimensions, 2)

    def test_is_two_dimensional(self):
        img2d = self._make_image([5, 5])
        img3d = self._make_image([5, 5, 5])
        self.assertTrue(ImageProperties(img2d).is_two_dimensional())
        self.assertFalse(ImageProperties(img3d).is_two_dimensional())

    def test_is_three_dimensional(self):
        img2d = self._make_image([5, 5])
        img3d = self._make_image([5, 5, 5])
        self.assertTrue(ImageProperties(img3d).is_three_dimensional())
        self.assertFalse(ImageProperties(img2d).is_three_dimensional())

    def test_is_vector_image_scalar(self):
        img = self._make_image([5, 5, 5])
        props = ImageProperties(img)
        self.assertFalse(props.is_vector_image())

    def test_is_vector_image_vector(self):
        arr = np.zeros((5, 5, 5, 3), dtype=np.float32)
        img = sitk.GetImageFromArray(arr, isVector=True)
        props = ImageProperties(img)
        self.assertTrue(props.is_vector_image())

    def test_str(self):
        img = self._make_image([4, 4, 4])
        props = ImageProperties(img)
        result = str(props)
        self.assertIn("ImageProperties", result)
        self.assertIn("size", result)
        self.assertIn("spacing", result)

    def test_equality_same_image(self):
        img = self._make_image([8, 8, 8])
        p1 = ImageProperties(img)
        p2 = ImageProperties(img)
        self.assertEqual(p1, p2)

    def test_inequality_different_size(self):
        img1 = self._make_image([4, 4, 4])
        img2 = self._make_image([8, 8, 8])
        p1 = ImageProperties(img1)
        p2 = ImageProperties(img2)
        self.assertNotEqual(p1, p2)

    def test_equality_non_image_properties(self):
        img = self._make_image([4, 4, 4])
        props = ImageProperties(img)
        result = props.__eq__("not_an_image_properties")
        self.assertIs(result, NotImplemented)

    def test_inequality_non_image_properties(self):
        img = self._make_image([4, 4, 4])
        props = ImageProperties(img)
        result = props.__ne__("not_an_image_properties")
        self.assertIs(result, NotImplemented)

    def test_hash(self):
        img = self._make_image([4, 4, 4])
        props = ImageProperties(img)
        h = hash(props)
        self.assertIsInstance(h, int)


class TestNumpySimpleITKImageBridge(unittest.TestCase):
    def _make_props(self, size):
        img = sitk.Image(size, sitk.sitkUInt8)
        return ImageProperties(img)

    def test_convert_1d_array(self):
        props = self._make_props([4, 4, 4])
        arr = np.zeros(64, dtype=np.uint8)
        result = NumpySimpleITKImageBridge.convert(arr, props)
        self.assertIsInstance(result, sitk.Image)
        self.assertEqual(result.GetSize(), (4, 4, 4))

    def test_convert_already_correct_shape(self):
        img = sitk.Image([3, 4, 5], sitk.sitkUInt8)
        props = ImageProperties(img)
        arr = sitk.GetArrayFromImage(img)  # shape (5, 4, 3)
        result = NumpySimpleITKImageBridge.convert(arr, props)
        self.assertIsInstance(result, sitk.Image)
        self.assertEqual(result.GetSize(), (3, 4, 5))

    def test_convert_preserves_origin_and_spacing(self):
        img = sitk.Image([4, 4, 4], sitk.sitkFloat32)
        img.SetOrigin([1.0, 2.0, 3.0])
        img.SetSpacing([0.5, 0.5, 0.5])
        props = ImageProperties(img)
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        result = NumpySimpleITKImageBridge.convert(arr, props)
        self.assertEqual(result.GetOrigin(), (1.0, 2.0, 3.0))
        self.assertEqual(result.GetSpacing(), (0.5, 0.5, 0.5))

    def test_convert_2d_array_as_vector(self):
        props = self._make_props([4, 4, 4])
        arr = np.zeros((64, 3), dtype=np.float32)
        result = NumpySimpleITKImageBridge.convert(arr, props)
        self.assertIsInstance(result, sitk.Image)

    def test_convert_4d_array_as_vector(self):
        img = sitk.Image([3, 4, 5], sitk.sitkFloat32)
        props = ImageProperties(img)
        # shape (5, 4, 3, 3) -> ndim == len(size)+1 == 4, treated as vector
        arr = np.zeros((5, 4, 3, 3), dtype=np.float32)
        result = NumpySimpleITKImageBridge.convert(arr, props)
        self.assertIsInstance(result, sitk.Image)

    def test_convert_unsupported_shape_raises(self):
        props = self._make_props([4, 4, 4])
        arr = np.zeros((4, 4), dtype=np.uint8)
        with self.assertRaises(ValueError):
            NumpySimpleITKImageBridge.convert(arr, props)


class TestSimpleITKNumpyImageBridge(unittest.TestCase):
    def test_convert_returns_array_and_properties(self):
        img = sitk.Image([5, 6, 7], sitk.sitkUInt8)
        arr, props = SimpleITKNumpyImageBridge.convert(img)
        self.assertIsInstance(arr, np.ndarray)
        self.assertIsInstance(props, ImageProperties)
        self.assertEqual(arr.shape, (7, 6, 5))

    def test_convert_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            SimpleITKNumpyImageBridge.convert(None)

    def test_convert_2d_image(self):
        img = sitk.Image([8, 9], sitk.sitkUInt16)
        arr, props = SimpleITKNumpyImageBridge.convert(img)
        self.assertEqual(arr.shape, (9, 8))
        self.assertTrue(props.is_two_dimensional())


if __name__ == "__main__":
    unittest.main()
