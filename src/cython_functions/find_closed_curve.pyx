import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
cimport cython


cpdef list check_closed_curve_loop(int l1, int l3, dict lines_lower_dict, dict lines_upper_dict, dict lines_intersurf_dict):
    """
    Check for a closed curve loop.

    This function checks if a closed curve loop can be formed using the provided line tags from the lower surface,
    upper surface, and intersurface dictionaries.

    Args:
        l1 (int): Line tag from the lower surface.
        l3 (int): Line tag from the upper surface.
        lines_lower_dict (dict): Dictionary of lower surface lines and their points.
        lines_upper_dict (dict): Dictionary of upper surface lines and their points.
        lines_intersurf_dict (dict): Dictionary of intersurface lines and their points.

    Returns:
        list: List of line tags forming a closed curve loop, or None if no loop is found.
    """
    cdef tuple l1_tuple, l3_tuple, l2_tuple, l4_tuple
    cdef int l4, l2
    l1_tuple = tuple(lines_lower_dict[l1])
    l3_tuple = tuple(lines_upper_dict[l3])
    for l4, l2 in product(lines_intersurf_dict.keys(), repeat=2):
        if l4 == l2:
            continue
        l2_tuple = tuple(lines_intersurf_dict[l2])
        l4_tuple = tuple(lines_intersurf_dict[l4])
        if (l1_tuple[0] == l2_tuple[1] or l1_tuple[1] == l2_tuple[0]) and (l1_tuple[0] == l4_tuple[0] or l1_tuple[1] == l4_tuple[1]):
            if (l3_tuple[0] == l2_tuple[0] or l3_tuple[1] == l2_tuple[1]) and (l3_tuple[0] == l4_tuple[1] or l3_tuple[1] == l4_tuple[0]):
                cl_s = [l1, l2, -l3, -l4] # ! -l3 and -l4
                return cl_s
    return None

cpdef find_closed_curve_loops(dict lines_lower_dict, dict lines_upper_dict, dict lines_intersurf_dict):
    """
    Find closed curve loops.

    This function finds all possible closed curve loops using the provided dictionaries of lines from the lower surface,
    upper surface, and intersurface.

    Args:
        lines_lower_dict (dict): Dictionary of lower surface lines and their points.
        lines_upper_dict (dict): Dictionary of upper surface lines and their points.
        lines_intersurf_dict (dict): Dictionary of intersurface lines and their points.

    Returns:
        list: List of closed curve loops, where each loop is represented as a list of line tags.
    """
    closed_curve_loops = []
    cdef list args_list = [(l1, l3, lines_lower_dict, lines_upper_dict, lines_intersurf_dict) for l1, l3 in product(list(lines_lower_dict.keys()), list(lines_upper_dict.keys()))]
    cdef int i
    with Pool(cpu_count()-2) as p:
        for result in p.starmap(check_closed_curve_loop, args_list):
            if result is not None:
                closed_curve_loops.append(result)
    return closed_curve_loops
