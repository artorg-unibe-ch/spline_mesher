"""Tests for pyhexspline.futils.setup_utils and hfe_input_transformer modules."""
import logging
import sys
import unittest
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk

from pyhexspline.futils.setup_utils import logging_setup

# hfe_input_transformer calls matplotlib.use("TkAgg") at module level, which
# requires a display. Patch it out so the module can be imported headlessly.
with patch("matplotlib.use"):
    from pyhexspline.futils.hfe_input_transformer import pad_image


class TestLoggingSetup(unittest.TestCase):
    def test_returns_logger(self):
        logger = logging_setup()
        self.assertIsInstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = logging_setup()
        self.assertEqual(logger.name, "MESHING")

    def test_logger_level(self):
        logger = logging_setup()
        self.assertEqual(logger.level, logging.INFO)

    def test_logger_has_handler(self):
        logger = logging_setup()
        self.assertGreater(len(logger.handlers), 0)


class TestPadImage(unittest.TestCase):
    def _make_image(self, size, pixel_type=sitk.sitkInt16):
        return sitk.Image(size, pixel_type)

    def test_pad_increases_xy_size(self):
        img = self._make_image([10, 10, 5])
        padded = pad_image(img, iso_pad_size=3)
        size = padded.GetSize()
        self.assertEqual(size[0], 10 + 2 * 3)
        self.assertEqual(size[1], 10 + 2 * 3)

    def test_pad_preserves_z_size(self):
        img = self._make_image([10, 10, 5])
        padded = pad_image(img, iso_pad_size=3)
        size = padded.GetSize()
        self.assertEqual(size[2], 5)

    def test_pad_zero_padding(self):
        img = self._make_image([8, 8, 4])
        padded = pad_image(img, iso_pad_size=0)
        self.assertEqual(padded.GetSize(), img.GetSize())

    def test_pad_returns_sitk_image(self):
        img = self._make_image([6, 6, 3])
        padded = pad_image(img, iso_pad_size=2)
        self.assertIsInstance(padded, sitk.Image)

    def test_pad_large_padding(self):
        img = self._make_image([4, 4, 2])
        padded = pad_image(img, iso_pad_size=10)
        size = padded.GetSize()
        self.assertEqual(size[0], 4 + 2 * 10)
        self.assertEqual(size[1], 4 + 2 * 10)
        self.assertEqual(size[2], 2)

    def test_pad_with_nonzero_image_values(self):
        arr = np.ones((3, 6, 6), dtype=np.int16) * 5
        img = sitk.GetImageFromArray(arr)
        padded = pad_image(img, iso_pad_size=1)
        padded_arr = sitk.GetArrayFromImage(padded)
        # Interior voxels should still be 5
        self.assertEqual(padded_arr[1, 2, 2], 5)


if __name__ == "__main__":
    unittest.main()
