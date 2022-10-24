
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
