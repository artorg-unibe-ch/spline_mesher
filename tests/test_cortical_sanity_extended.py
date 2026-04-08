"""Additional tests for pyhexspline.cortical_sanity module."""
import logging
import math
import unittest

import numpy as np

from pyhexspline import cortical_sanity as cs


def _make_circle(radius: float, n_points: int) -> np.ndarray:
    """Return a circle contour with n_points as an (n, 2) array."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return np.column_stack([np.cos(t) * radius, np.sin(t) * radius])


def _make_checker(min_thickness: float = 0.5) -> cs.CorticalSanityCheck:
    logger = logging.getLogger("test")
    ext = _make_circle(10.0, 50)
    inner = _make_circle(7.0, 50)
    return cs.CorticalSanityCheck(
        MIN_THICKNESS=min_thickness,
        ext_contour=ext,
        int_contour=inner,
        model="test_model",
        save_plot=False,
        logger=logger,
    )


class TestCorticalSanityCheckInit(unittest.TestCase):
    def test_attributes_stored(self):
        checker = _make_checker(1.0)
        self.assertEqual(checker.min_thickness, 1.0)
        self.assertEqual(checker.model, "test_model")
        self.assertFalse(checker.save_plot)


class TestUnitVector(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_unit_vector_3_4(self):
        uv = self.checker.unit_vector(np.array([3, 4]))
        self.assertAlmostEqual(uv[0], 0.6)
        self.assertAlmostEqual(uv[1], 0.8)

    def test_unit_vector_1_0(self):
        uv = self.checker.unit_vector(np.array([1, 0]))
        self.assertAlmostEqual(uv[0], 1.0)
        self.assertAlmostEqual(uv[1], 0.0)

    def test_unit_vector_0_1(self):
        uv = self.checker.unit_vector(np.array([0, 1]))
        self.assertAlmostEqual(uv[0], 0.0)
        self.assertAlmostEqual(uv[1], 1.0)


class TestCenterOfMass(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_uniform_array(self):
        arr = np.ones((4, 4))
        cx, cy = self.checker.center_of_mass(arr)
        self.assertAlmostEqual(cx, 1.5)
        self.assertAlmostEqual(cy, 1.5)

    def test_single_element(self):
        arr = np.array([[5]])
        cx, cy = self.checker.center_of_mass(arr)
        self.assertAlmostEqual(cx, 0.0)
        self.assertAlmostEqual(cy, 0.0)


class TestResetAndRollIndex(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()
        self.arr = np.arange(10).reshape(5, 2).astype(float)

    def test_reset_numpy_index_shifts(self):
        result = self.checker.reset_numpy_index(self.arr, idx=2)
        np.testing.assert_array_equal(result[0], self.arr[2])

    def test_roll_index_shifts(self):
        result = self.checker.roll_index(self.arr, idx=2)
        np.testing.assert_array_equal(result[0], self.arr[2])

    def test_reset_roll_equivalence(self):
        reset = self.checker.reset_numpy_index(self.arr, idx=3)
        rolled = self.checker.roll_index(self.arr, idx=3)
        np.testing.assert_array_equal(reset[0], rolled[0])


class TestConvertRadiansToDegrees(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_pi_is_180(self):
        self.assertAlmostEqual(
            self.checker.convertRadiansToDegrees(np.pi), 180.0, places=7
        )

    def test_zero_is_zero(self):
        self.assertAlmostEqual(
            self.checker.convertRadiansToDegrees(0.0), 0.0
        )

    def test_two_pi_is_360(self):
        self.assertAlmostEqual(
            self.checker.convertRadiansToDegrees(2 * np.pi), 360.0, places=7
        )


class TestCcwAngle(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_angle_range(self):
        a1 = np.random.rand(5, 2)
        a2 = np.random.rand(5, 2)
        idx1 = np.arange(len(a1))
        idx2 = np.arange(len(a2))
        angles = self.checker.ccw_angle(a1, a2, idx1, idx2)
        self.assertTrue(np.all(angles >= 0))
        self.assertTrue(np.all(angles < 2 * np.pi))

    def test_same_direction_gives_zero(self):
        a = np.array([[1.0, 0.0], [1.0, 0.0]])
        idx = np.array([0, 1])
        angles = self.checker.ccw_angle(a, a, idx, idx)
        np.testing.assert_array_almost_equal(angles, 0.0)


class TestIsAngleBiggerBool(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_int_bigger_returns_true(self):
        self.assertTrue(self.checker.is_angle_bigger_bool(2.0, 1.0))

    def test_ext_bigger_returns_false(self):
        self.assertFalse(self.checker.is_angle_bigger_bool(1.0, 2.0))

    def test_equal_returns_true(self):
        self.assertTrue(self.checker.is_angle_bigger_bool(1.5, 1.5))


class TestNearestPoint(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()
        self.circle = _make_circle(10.0, 100)

    def test_nearest_point_returns_correct_index(self):
        loc, idx = self.checker.nearest_point(self.circle, [10, 0])
        self.assertAlmostEqual(loc[0], self.circle[idx][0])
        self.assertAlmostEqual(loc[1], self.circle[idx][1])

    def test_nearest_point_close_to_query(self):
        query = [5.0, 5.0]
        loc, _ = self.checker.nearest_point(self.circle, query)
        dist = math.hypot(loc[0] - query[0], loc[1] - query[1])
        # All points on the circle are at radius 10; distance to closest should be finite
        self.assertGreater(dist, 0.0)
        self.assertLess(dist, 20.0)


class TestKDTnn(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()
        self.arr = _make_circle(10.0, 50)

    def test_returns_integer(self):
        idx = self.checker.KDT_nn(np.array([10.0, 0.0]), self.arr)
        self.assertIsInstance(int(idx), int)

    def test_returns_expected_index(self):
        idx = self.checker.KDT_nn(self.arr[5], self.arr)
        self.assertEqual(idx, 5)


class TestProjectPointOnLine(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_projection_on_horizontal_line(self):
        p = np.array([1.0, 1.0])
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 0.0])
        proj = self.checker.project_point_on_line(p, a, b)
        np.testing.assert_array_almost_equal(proj, [1.0, 0.0])

    def test_projection_on_diagonal_line(self):
        p = np.array([0.0, 2.0])
        a = np.array([0.0, 0.0])
        b = np.array([2.0, 2.0])
        proj = self.checker.project_point_on_line(p, a, b)
        np.testing.assert_array_almost_equal(proj, [1.0, 1.0])

    def test_point_on_line_projects_to_itself(self):
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 0.0])
        p = np.array([2.0, 0.0])
        proj = self.checker.project_point_on_line(p, a, b)
        np.testing.assert_array_almost_equal(proj, p)


class TestResampleContour(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()
        self.circle = _make_circle(10.0, 50)

    def test_output_shape(self):
        resampled = self.checker.resample_contour(self.circle, n_points=25)
        self.assertEqual(resampled.shape, (25, 2))

    def test_output_stays_near_circle(self):
        resampled = self.checker.resample_contour(self.circle, n_points=100)
        radii = np.sqrt(resampled[:, 0] ** 2 + resampled[:, 1] ** 2)
        np.testing.assert_array_almost_equal(radii, 10.0, decimal=1)

    def test_same_n_points(self):
        resampled = self.checker.resample_contour(self.circle, n_points=50)
        self.assertEqual(resampled.shape, (50, 2))


class TestLinesegDists(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_point_above_segment(self):
        p = np.array([1.0, 1.0])
        a = np.array([[0.0, 0.0], [2.0, 0.0]])
        b = np.array([[2.0, 0.0], [4.0, 0.0]])
        dists = self.checker.lineseg_dists(p, a, b)
        self.assertEqual(dists.shape[0], 2)
        self.assertGreater(dists[0], 0.0)

    def test_point_on_segment_has_zero_distance(self):
        p = np.array([1.0, 0.0])
        a = np.array([[0.0, 0.0]])
        b = np.array([[2.0, 0.0]])
        dists = self.checker.lineseg_dists(p, a, b)
        self.assertAlmostEqual(float(dists[0]), 0.0, places=6)


class TestCheckMinThickness(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_close_points_below_threshold(self):
        arr1 = np.full((5, 2), 1.00)
        arr2 = np.full((5, 2), 1.01)
        idx1 = np.arange(len(arr1))
        idx2 = np.arange(len(arr2))
        result = self.checker.check_min_thickness(arr1, arr2, idx1, idx2)
        self.assertTrue(np.asarray(result).all())

    def test_far_points_above_threshold(self):
        arr1 = np.full((5, 2), 0.00)
        arr2 = np.full((5, 2), 10.00)
        idx1 = np.arange(len(arr1))
        idx2 = np.arange(len(arr2))
        result = self.checker.check_min_thickness(arr1, arr2, idx1, idx2)
        self.assertFalse(np.asarray(result).all())


class TestNearestPairsArrs(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_output_shapes(self):
        ext = _make_circle(10.0, 10)
        inner = _make_circle(7.0, 10)
        dn, bi, ci = self.checker.nearest_pairs_arrs(ext, inner)
        self.assertEqual(dn.shape, (10,))
        self.assertEqual(bi.shape, (10, 2))
        self.assertEqual(ci.shape, (10, 2))

    def test_distances_are_nonnegative(self):
        ext = _make_circle(10.0, 10)
        inner = _make_circle(7.0, 10)
        dn, _, _ = self.checker.nearest_pairs_arrs(ext, inner)
        self.assertTrue(np.all(dn >= 0))


class TestIsInternalRadiusBiggerThanExternal(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_inner_smaller_than_ext(self):
        ext = _make_circle(10.0, 20)
        inner = _make_circle(5.0, 20)
        idx = np.arange(len(ext))
        result = self.checker.is_internal_radius_bigger_than_external(
            ext, inner, idx, idx
        )
        # inner radius is smaller, so bool_radius should be all False
        self.assertFalse(any(result))

    def test_inner_bigger_than_ext(self):
        ext = _make_circle(5.0, 20)
        inner = _make_circle(10.0, 20)
        idx = np.arange(len(ext))
        result = self.checker.is_internal_radius_bigger_than_external(
            ext, inner, idx, idx
        )
        # inner radius is larger than ext, so bool_radius should be all True
        self.assertTrue(all(result))


class TestOffsetSurface(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker()

    def test_negative_offset_shrinks_polygon(self):
        circle = _make_circle(10.0, 100)
        offset = self.checker.offset_surface(circle, -2.0)
        # The resulting polygon should have a mean radius near 8
        radii = np.sqrt(offset[:, 0] ** 2 + offset[:, 1] ** 2)
        self.assertAlmostEqual(radii.mean(), 8.0, delta=0.5)

    def test_positive_offset_grows_polygon(self):
        circle = _make_circle(10.0, 100)
        offset = self.checker.offset_surface(circle, 2.0)
        radii = np.sqrt(offset[:, 0] ** 2 + offset[:, 1] ** 2)
        self.assertAlmostEqual(radii.mean(), 12.0, delta=0.5)

    def test_returns_2d_array(self):
        circle = _make_circle(10.0, 50)
        offset = self.checker.offset_surface(circle, -1.0)
        self.assertEqual(offset.ndim, 2)
        self.assertEqual(offset.shape[1], 2)


class TestPushContour(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker(min_thickness=1.0)

    def test_returns_tuple_of_arrays(self):
        ext = _make_circle(10.0, 50)
        inner = _make_circle(7.0, 50)
        new_int, ext_offset = self.checker.push_contour(ext, inner, -1.0)
        self.assertIsInstance(new_int, np.ndarray)
        self.assertIsInstance(ext_offset, np.ndarray)

    def test_new_int_shape_preserved(self):
        n = 40
        ext = _make_circle(10.0, n)
        inner = _make_circle(7.0, n)
        new_int, _ = self.checker.push_contour(ext, inner, -1.0)
        self.assertEqual(new_int.shape[1], 2)

    def test_inner_outside_gets_pushed_inside(self):
        # inner contour that is completely outside the shrunk external
        ext = _make_circle(10.0, 100)
        inner = _make_circle(15.0, 100)  # outside ext
        new_int, ext_offset = self.checker.push_contour(ext, inner, -1.0)
        # After push, all inner points should be within the ext_offset polygon
        import shapely.geometry as shpg

        poly = shpg.Polygon(ext_offset)
        for pt in new_int:
            is_inside = shpg.Point(pt).within(poly)
            is_on_boundary = poly.distance(shpg.Point(pt)) < 1e-6
            self.assertTrue(is_inside or is_on_boundary)


class TestCorticalSanityCheck(unittest.TestCase):
    def setUp(self):
        self.checker = _make_checker(min_thickness=1.0)

    def test_cortical_sanity_check_no_plots(self):
        ext = _make_circle(10.0, 50)
        inner = _make_circle(7.0, 50)
        result = self.checker.cortical_sanity_check(
            ext, inner, iterator=0, show_plots=False
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
