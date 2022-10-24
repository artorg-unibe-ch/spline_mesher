
from code_pack import cortical_sanity as cs
import nose.tools as nt
import numpy as np
# Path: 02_CODE/tests/tests_cortical_sanity.py

def test_ccw_angle():
    """CCW angle should be between 0 and 2*pi"""
    a = cs.CorticalSanityCheck
    a1 = np.random.rand(5, 2)
    a2 = np.random.rand(5, 2)
    a = a.ccw_angle(a, array1=a1, array2=a2)
    nt.assert_true(0 <= a.all() <= 2 * np.pi)

def test_convertRadiansToDegrees():
    """Convert to degrees should convert radians to degrees"""
    a = cs.CorticalSanityCheck
    a = a.convertRadiansToDegrees(a, radians=np.pi)
    nt.assert_almost_equal(a, 180, places=7)

def test_reset_numpy_index(arr=np.random.rand(100,2)):
    """Reset numpy index should reset the index to 0"""
    b = cs.CorticalSanityCheck
    b = b.reset_numpy_index(b, arr, (len(arr[:,0])-1))
    nt.assert_equal(b[0, 0], arr[-1, 0])

def test_is_angle_bigger_bool():
    alpha_int = np.array([15, 15, 15, 15])
    alpha_ext = np.array([1.5, 1.5, 1.5, 1.5])

    bool_angle = cs.CorticalSanityCheck
    bool_angle = [bool_angle.is_angle_bigger_bool(bool_angle, alpha_int, alpha_ext) for alpha_int, alpha_ext in zip(alpha_int, alpha_ext)]
    print(bool_angle)
    nt.assert_true(np.asarray(bool_angle).all())

def test_check_min_thickness():
    """Check intersections should return True if there are intersections"""
    a = cs.CorticalSanityCheck
    a = a.check_min_thickness(a, np.full((5, 10), 1.00), np.full((5, 10), 1.01))
    nt.assert_true(np.asarray(a).all())

def test_check_bigger_thickness():
    b = cs.CorticalSanityCheck
    b = b.check_min_thickness(b, np.full((5, 10), 1.00), np.full((5, 10), 10.00))
    nt.assert_false(np.asarray(b).all())
