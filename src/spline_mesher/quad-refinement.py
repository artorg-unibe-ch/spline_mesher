import matplotlib.pyplot as plt
import numpy as np
import gmsh

# * SETTINGS
cl = 3
cg = (0, 0)


# * MULTIPLE INLINE SQUARES FUNCTION
def sort_trab_ccw(squares_unique_coords):
    squares_center = np.mean(squares_unique_coords, axis=0)
    squares_ccw_idx = np.argsort(
        np.arctan2(
            squares_unique_coords[:, 1] - squares_center[1],
            squares_unique_coords[:, 0] - squares_center[0],
        )
    )
    return squares_unique_coords[squares_ccw_idx]


def gmsh_add_points(squares_ccw_sort):
    points = np.empty(len(squares_ccw_sort))
    for i, point in enumerate(squares_ccw_sort):
        point_tags = gmsh.model.occ.addPoint(point[0], point[1], 0)
        points[i] = point_tags
    points = np.array(points, dtype=int)
    return points


def gmsh_create_lines(points):
    lines = []
    for point_arr in points:
        point_arr = np.append(point_arr, point_arr[0])
    for i in range(len(point_arr) - 1):
        line_tags = gmsh.model.occ.addLine(point_arr[i], point_arr[i + 1])
        lines.append(line_tags)
    return lines


def create_squares_m(cg, cl, n_squares, _dir):
    squares = []
    if _dir == "vertical":
        for i in range(n_squares):
            center_x = cg[0] + (i - (n_squares - 1) / 2) * cl
            square = np.array(
                [
                    [center_x - cl / 2, cg[1] - cl / 2],
                    [center_x - cl / 2, cg[1] + cl / 2],
                    [center_x + cl / 2, cg[1] + cl / 2],
                    [center_x + cl / 2, cg[1] - cl / 2],
                ]
            )
            squares.append(square)
            squares_reshaped = np.array(squares).reshape(-1, 2)
            squares_ccw_sort = sort_trab_ccw(squares_reshaped)

    elif _dir == "horizontal":
        for i in range(n_squares):
            center_y = cg[1] + (i - (n_squares - 1) / 2) * cl
            square = np.array(
                [
                    [cg[0] - cl / 2, center_y - cl / 2],
                    [cg[0] + cl / 2, center_y - cl / 2],
                    [cg[0] + cl / 2, center_y + cl / 2],
                    [cg[0] - cl / 2, center_y + cl / 2],
                ]
            )
            squares.append(square)
            squares_reshaped = np.array(squares).reshape(-1, 2)
            squares_ccw_sort = sort_trab_ccw(squares_reshaped)
    return squares_ccw_sort


def create_square_directional(cg, cl_i, cl_j):
    cl_t = cl_i + cl_j
    m = 3
    if cl_i == 0:
        _dir_s = "vertical"
        # fmt: off
        square_bottom = create_squares_m((cg[0] - cl_i, cg[1] - cl_j), cl_t, 1, _dir=_dir_s)
        square_n1 = create_squares_m((cg[0] - cl_i, cg[1] - cl_j), cl_t / m, 2 * m + 1, _dir=_dir_s)
        square_n11 = create_squares_m((cg[0] - cl_i, cg[1] - (cl_j + cl_j / m)), cl_t / (3 * m), 27, _dir=_dir_s)
        square_n11_bottom = create_squares_m((cg[0] - cl_i, cg[1] - (cl_j + cl_j / m + cl_j / (3 * m))), cl_t / (3 * m), 27, _dir=_dir_s)
        # fmt: on
    elif cl_j == 0:
        # fmt: off
        _dir_s = "horizontal"
        square_bottom = create_squares_m((cg[0] - cl_i, cg[1] - cl_j), cl_t, 1, _dir=_dir_s)
        square_n1 = create_squares_m((cg[0] - cl_i, cg[1] - cl_j), cl_t / m, 2 * m + 1, _dir=_dir_s)
        square_n11 = create_squares_m((cg[0] - (cl_i + cl_i / m), cg[1] - cl_j), cl_t / (3 * m), 27, _dir=_dir_s)
        square_n11_bottom = create_squares_m((cg[0] - (cl_i + cl_i / m + cl_i / (3 * m)), cg[1] - cl_j), cl_t / (3 * m), 27, _dir=_dir_s)
        # fmt: on
    else:
        raise ValueError("cl_i and cl_j cannot be both different from 0")
    # fmt: off
    point_coords = np.concatenate((square_bottom, square_n1, square_n11, square_n11_bottom)).tolist()
    # fmt: on
    point_unique = np.unique(point_coords, axis=0)
    # points = point_coords[~np.isin(point_coords, point_unique)]
    point_tags = gmsh_add_points(point_unique)

    point_coords = np.array(point_coords)
    return point_coords, point_tags


def remove_duplicate_points():
    """I'm ashamed of this function but it works"""
    gmsh.model.occ.synchronize()
    points = gmsh.model.occ.getEntities(0)
    coords = []
    for point in points:
        values_s = gmsh.model.getValue(0, point[1], [])
        coords.append(values_s)
    coords = np.array(coords)
    u, indices = np.unique(coords.round(decimals=1), axis=0, return_index=True)
    coords_unique = np.array(coords)[indices.astype(int)]
    # delete previous points
    for point in points:
        gmsh.model.occ.remove([point])
    # add new points
    for point in coords_unique:
        gmsh.model.occ.addPoint(point[0], point[1], point[2], tag=-1)


def main():
    gmsh.initialize()
    square_center = create_squares_m(cg, cl, 1, _dir="horizontal")
    square_inf = create_square_directional(cg=cg, cl_i=0, cl_j=cl)
    square_l = create_square_directional(cg=cg, cl_i=cl, cl_j=0)
    square_sup = create_square_directional(cg=cg, cl_i=0, cl_j=-cl)
    square_r = create_square_directional(cg=cg, cl_i=-cl, cl_j=0)
    remove_duplicate_points()

    gmsh.model.occ.synchronize()
    gmsh.write("99_testing_prototyping/square_2.geo_unrolled")
    # gmsh.fltk.run()
    gmsh.finalize()

    a = 10
    fig = plt.figure(figsize=(a, a))
    ax = fig.add_subplot(111)
    ax.plot(square_inf[0][:, 0], square_inf[0][:, 1], "o", label="square_inf")
    ax.plot(square_r[0][:, 0], square_r[0][:, 1], "o", label="square_r")
    ax.plot(square_sup[0][:, 0], square_sup[0][:, 1], "o", label="square_sup")
    ax.plot(square_l[0][:, 0], square_l[0][:, 1], "o", label="square_l")
    ax.plot(square_center[:, 0], square_center[:, 1], "o", label="square_center")

    plt.legend()
    ax.set_aspect("equal")
    plt.show()


if "__main__" == __name__:
    main()
