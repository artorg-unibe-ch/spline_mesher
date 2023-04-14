import numpy as np
from itertools import product

cpdef find_closed_curve_loops(dict lines_lower_dict, dict lines_upper_dict, dict lines_intersurf_dict):
    cdef list closed_curve_loops = []
    cdef int l1, l3, l4, l2
    cdef tuple l1_tuple, l3_tuple, l2_tuple, l4_tuple

    for l1, l3 in product(lines_lower_dict.keys(), lines_upper_dict.keys()):
        l1_tuple = tuple(lines_lower_dict[l1])
        l3_tuple = tuple(lines_upper_dict[l3])
        for l4, l2 in product(lines_intersurf_dict.keys(), repeat=2):
            if l4 == l2:
                continue
            l2_tuple = tuple(lines_intersurf_dict[l2])
            l4_tuple = tuple(lines_intersurf_dict[l4])
            if (l1_tuple[0] == l2_tuple[1] or l1_tuple[1] == l2_tuple[0]) and (l1_tuple[0] == l4_tuple[0] or l1_tuple[1] == l4_tuple[1]):
                if (l3_tuple[0] == l2_tuple[0] or l3_tuple[1] == l2_tuple[1]) and (l3_tuple[0] == l4_tuple[1] or l3_tuple[1] == l4_tuple[0]):
                    cl_s = [l1, l2, l3, l4]
                    closed_curve_loops.append(cl_s)
    return closed_curve_loops
