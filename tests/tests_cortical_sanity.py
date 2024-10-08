# from spline_mesher import cortical_sanity as cs
import math
import numpy as np

from pyhexspline import cortical_sanity as cs

# Path: 02_CODE/tests/tests_cortical_sanity.py


def test_ccw_angle():
    """CCW angle should be between 0 and 2*pi"""
    a = cs.CorticalSanityCheck
    a1 = np.random.rand(5, 2)
    a2 = np.random.rand(5, 2)
    a = a.ccw_angle(
        a,
        array1=a1,
        array2=a2,
        idx1=np.arange(len(a1)),
        idx2=np.arange(len(a2)),
    )
    assert 0 <= a.all() <= 2 * np.pi, "CCW angle is not between 0 and 2*pi"


def test_convertRadiansToDegrees():
    """Convert to degrees should convert radians to degrees"""
    a = cs.CorticalSanityCheck
    a = a.convertRadiansToDegrees(a, radians=np.pi)
    assert math.isclose(
        a, 180, rel_tol=1e-7
    ), "Radians are not correctly converted to degrees"


def test_reset_numpy_index(arr=np.random.rand(100, 2)):
    """Reset numpy index should reset the index to 0"""
    b = cs.CorticalSanityCheck
    b = b.reset_numpy_index(b, arr, (len(arr[:, 0]) - 1))
    assert b[0, 0] == arr[-1, 0]


def test_roll_numpy_index(arr=np.random.rand(100, 2)):
    c = cs.CorticalSanityCheck
    c = c.reset_numpy_index(c, arr, (len(arr[:, 0]) - 1))

    d = cs.CorticalSanityCheck
    d = d.roll_index(d, arr, (len(arr[:, 0]) - 1))
    assert c[0, 0] == d[0, 0]


def test_is_angle_bigger_bool():
    alpha_int = np.array([15, 15, 15, 15])
    alpha_ext = np.array([1.5, 1.5, 1.5, 1.5])

    bool_angle = cs.CorticalSanityCheck
    bool_angle = [
        bool_angle.is_angle_bigger_bool(bool_angle, alpha_int, alpha_ext)
        for alpha_int, alpha_ext in zip(alpha_int, alpha_ext)
    ]
    print(bool_angle)
    assert np.asarray(bool_angle).all()


def test_check_min_thickness():
    """Check intersections should return True if there are intersections"""
    a = cs.CorticalSanityCheck
    arr1 = np.full((5, 10), 1.00)
    arr2 = np.full((5, 10), 1.01)
    idx_arr1 = np.arange(len(arr1[:, 0]))
    idx_arr2 = np.arange(len(arr2[:, 0]))
    a = a.check_min_thickness(a, arr1, arr2, idx_arr1, idx_arr2)
    assert np.asarray(a).all()


def test_check_bigger_thickness():
    b = cs.CorticalSanityCheck
    arr1 = np.full((5, 10), 1.00)
    arr2 = np.full((5, 10), 10.00)
    idx_arr1 = np.arange(len(arr1[:, 0]))
    idx_arr2 = np.arange(len(arr2[:, 0]))
    b = b.check_min_thickness(b, arr1, arr2, idx_arr1, idx_arr2)
    assert not np.asarray(b).all()
