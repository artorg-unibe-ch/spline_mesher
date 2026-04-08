"""Tests for pyhexspline.futils.geo_utils module."""
import os
import tempfile
import unittest

from pyhexspline.futils.geo_utils import GeoSort


def _write_temp_geo(content: str) -> str:
    """Write content to a temporary .geo_unrolled file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".geo_unrolled", delete=False
    )
    f.write(content)
    f.close()
    return f.name


_GEO_CONTENT_1 = """\
SetFactory("OpenCASCADE");
cl__1 = 0.5;
Point(1) = {0, 0, 0, cl__1};
Point(2) = {1, 0, 0, cl__1};
Point(3) = {1, 1, 0, cl__1};
Point(4) = {0, 1, 0, cl__1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Surface Loop(1) = {1};
Volume(1) = {1};
"""

_GEO_CONTENT_2 = """\
cl__2 = 0.3;
Point(5) = {0.2, 0.2, 0, cl__2};
Point(6) = {0.8, 0.2, 0, cl__2};
Point(7) = {0.8, 0.8, 0, cl__2};
Point(8) = {0.2, 0.8, 0, cl__2};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
Surface Loop(2) = {2};
Volume(2) = {2};
"""


class TestGeoSort(unittest.TestCase):
    def setUp(self):
        self.file1 = _write_temp_geo(_GEO_CONTENT_1)
        self.file2 = _write_temp_geo(_GEO_CONTENT_2)
        self._tmpdir = tempfile.mkdtemp()
        self.tmpbase = os.path.join(self._tmpdir, "test_geo")
        self.sorted_file = self.tmpbase + "_sorted.geo_unrolled"
        self.geo_sort = GeoSort(self.file1, self.file2, self.tmpbase, "Delete")

    def tearDown(self):
        for path in [self.file1, self.file2, self.sorted_file]:
            if os.path.exists(path):
                os.unlink(path)
        if os.path.isdir(self._tmpdir):
            os.rmdir(self._tmpdir)

    def test_init_attributes(self):
        self.assertEqual(self.geo_sort.file1, self.file1)
        self.assertEqual(self.geo_sort.file2, self.file2)
        self.assertEqual(self.geo_sort.filename, self.tmpbase)
        self.assertEqual(self.geo_sort.filename_sorted, self.sorted_file)
        self.assertEqual(self.geo_sort.boolean, "Delete")

    def test_sort_lines_returns_list(self):
        lines = self.geo_sort.sort_lines(self.file1)
        self.assertIsInstance(lines, list)
        self.assertGreater(len(lines), 0)

    def test_sort_lines_content(self):
        lines = self.geo_sort.sort_lines(self.file1)
        joined = "".join(lines)
        self.assertIn("cl__1", joined)
        self.assertIn("Point", joined)
        self.assertIn("Volume", joined)

    def test_append_file2_to_file1_creates_sorted_file(self):
        self.geo_sort.append_file2_to_file1(self.file1, self.file2)
        self.assertTrue(os.path.exists(self.sorted_file))

    def test_append_file2_to_file1_contains_both(self):
        self.geo_sort.append_file2_to_file1(self.file1, self.file2)
        with open(self.sorted_file) as f:
            content = f.read()
        self.assertIn("cl__1", content)
        self.assertIn("cl__2", content)

    def test_write_geo_creates_file(self):
        self.geo_sort.append_file2_to_file1(self.file1, self.file2)
        result = self.geo_sort.write_geo()
        self.assertTrue(os.path.exists(result))

    def test_write_geo_returns_sorted_path(self):
        self.geo_sort.append_file2_to_file1(self.file1, self.file2)
        result = self.geo_sort.write_geo()
        self.assertEqual(result, self.sorted_file)

    def test_write_geo_content_has_factory_header(self):
        self.geo_sort.append_file2_to_file1(self.file1, self.file2)
        self.geo_sort.write_geo()
        with open(self.sorted_file) as f:
            content = f.read()
        self.assertIn('SetFactory("OpenCASCADE")', content)

    def test_read_lines_structure(self):
        self.geo_sort.append_file2_to_file1(self.file1, self.file2)
        parts = self.geo_sort.read_lines()
        # read_lines returns a list of 10 items (header + 9 categories)
        self.assertEqual(len(parts), 10)

    def test_read_lines_has_volumes(self):
        self.geo_sort.append_file2_to_file1(self.file1, self.file2)
        parts = self.geo_sort.read_lines()
        volumes = parts[8]
        self.assertTrue(len(volumes) >= 2)


if __name__ == "__main__":
    unittest.main()
