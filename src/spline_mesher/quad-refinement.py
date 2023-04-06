"""
Quad refinement scheme for the inner trabecular region
Author: Simone Poncioni, MSB
Date: 04.2023

https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array
https://blender.stackexchange.com/questions/230534/fastest-way-to-skin-a-grid

"""
import logging
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.util import view_as_blocks, view_as_windows

LOGGING_NAME = "SIMONE"
# flake8: noqa: E402


class QuadRefinement:
    def __init__(self, vertices_tags: list[int], nb_layers: int, DIM: int = 2):
        self.vertices_tags = np.array(vertices_tags)
        self.nb_layers = int(nb_layers)
        self.DIM = int(DIM)
        self.model = gmsh.model
        self.factory = gmsh.model.occ
        self.EDGE_LENGTH = float(1)
        self.SHOW_PLOT = bool(False)

    def center_of_mass(self, vertices_coords):
        return np.mean(vertices_coords, axis=0)

    def get_vertices_coords(self):
        self.factory.synchronize()
        vertices_coords = []
        for vertex in self.vertices_tags:
            vertices_coords.append(self.model.getValue(0, vertex, []))
        return np.array(vertices_coords, dtype=np.float32)

    def create_square_grid(self, big_square_size, squares_per_side, origin):
        # Calculate the size of each small square
        small_square_size = big_square_size / squares_per_side

        vertices = []
        for i in range(squares_per_side + 1):
            for j in range(squares_per_side + 1):
                # Calculate the coordinates of the square's top-left corner
                x_s_1 = float(i * small_square_size)
                y_s_1 = float(j * small_square_size)
                vertices.append([x_s_1, y_s_1, origin[2]])

        # center vertices to origing
        vert_max_x = max(vertices, key=lambda x: x[0])[0]
        vert_max_y = max(vertices, key=lambda x: x[1])[1]
        vert_max_z = max(vertices, key=lambda x: x[2])[2]
        vertices_s = [
            np.array(v) - np.array([vert_max_x / 2, vert_max_y / 2, vert_max_z])
            for v in vertices
        ]
        # shift vertices to origin
        vertices = [np.array(v) + np.array(origin) for v in vertices_s]
        return vertices

    def create_subvertices(self, center_of_mass, hierarchy, square_size_0):
        squares_per_side_1 = 3
        if hierarchy < 1:
            raise ValueError("Hierarchy must be >= 1")
        elif hierarchy >= 1:
            squares_per_side = squares_per_side_1 ** (hierarchy)
        vertices = self.create_square_grid(
            square_size_0, squares_per_side, center_of_mass
        )
        return vertices

    def plot_vertices(self, vertices):
        plt.plot(figsize=(5, 5))
        plt.plot(
            [v[0] for v in vertices],
            [v[1] for v in vertices],
            ".",
            color="black",
        )
        # annotate index of each vertex
        for i, v in enumerate(vertices):
            plt.text(v[0] + 0.01, v[1] + 0.01, str(i), color="red", fontsize=10)
        plt.show()

    def gmsh_add_points(self, vertices):
        point_tags = []
        for v in vertices:
            v_tag = self.factory.addPoint(v[0], v[1], v[2], -1)
            point_tags.append(v_tag)
        return point_tags

    def gmsh_add_surfaces(self, line_tags):
        cloops = []
        for subline_tags in line_tags[30:-30]:
            print(subline_tags)
            curve = self.factory.addCurveLoop(subline_tags, -1)
            cloops.append(curve)

        surf_tags = []
        for cloop in cloops:
            surf = self.factory.addPlaneSurface([cloop], -1)
            surf_tags.append(surf)
        return surf_tags

    def get_vertices_minmax(self, point_coords):
        max_x = max(point_coords, key=lambda x: x[0])[0]
        min_x = min(point_coords, key=lambda x: x[0])[0]
        max_y = max(point_coords, key=lambda x: x[1])[1]
        min_y = min(point_coords, key=lambda x: x[1])[1]
        # compose verts_2 with the 4 outermost vertices of point_coords
        verts = np.array(
            [
                [max_x, max_y, 0],
                [min_x, max_y, 0],
                [min_x, min_y, 0],
                [max_x, min_y, 0],
            ],
            dtype=np.float32,
        )
        return verts

    def get_affine_transformation_matrix(self, verts_1, point_coords):
        # find 4 outermost vertices of point_coords
        verts_2 = self.get_vertices_minmax(point_coords)

        # ensure c-continuity of verts_1 and verts_2 after slicing (as checked by cv::Mat::checkVector())
        # https://stackoverflow.com/questions/54552289/assertion-error-from-opencv-checkvector-in-python
        verts_1_affine = np.copy(verts_1[:3, :2], order="C")
        verts_2_affine = np.copy(verts_2[:3, :2], order="C")
        M = cv2.getAffineTransform(verts_2_affine, verts_1_affine)

        if self.SHOW_PLOT:
            plt.figure(figsize=(5, 5))
            plt.scatter(verts_1[:, 0], verts_1[:, 1], color="red")
            plt.scatter(verts_2[:, 0], verts_2[:, 1], color="blue")
            plt.show()
        return M

    def set_transformation_matrix(self, M, point_coords):
        coords = np.array(point_coords[:, :2], dtype=np.float32).reshape(-1, 1, 2)
        coords_transformed = cv2.transform(coords, M)
        new_coords = coords_transformed.reshape(-1, 2)
        new_coords_3d = np.hstack((new_coords, point_coords[:, 2].reshape(-1, 1)))

        if self.SHOW_PLOT:
            plt.figure(figsize=(5, 5))
            plt.scatter(
                point_coords[:, 0], point_coords[:, 1], color="red", label="original"
            )
            plt.scatter(
                new_coords[:, 0], new_coords[:, 1], color="blue", label="transformed"
            )
            plt.legend()
            plt.show()
        return new_coords_3d

    def vertices_grid_cleanup(self, vertices):
        vertices_unique = np.unique(
            vertices.round(decimals=4),
            axis=0,
        )
        vertices_sorted = np.array(
            sorted(vertices_unique, key=lambda k: [k[1], k[0]]), dtype=np.float32
        )
        return vertices_sorted

    def gmsh_add_line(self, point_skins):
        line_tags = []
        _iter = int(1)  # slicing iterator
        for i, skin in enumerate(point_skins[::_iter]):
            for j, _ in enumerate(skin[::_iter]):
                p0 = skin[j * _iter][0][0]
                p1 = skin[j * _iter][0][-1]
                p2 = skin[j * _iter][-1][0]
                p3 = skin[j * _iter][-1][-1]
                line_s1 = self.factory.addLine(p0, p1)
                line_s2 = self.factory.addLine(p0, p2)
                line_s3 = self.factory.addLine(p2, p3)
                line_s4 = self.factory.addLine(p1, p3)
                line_tags.append((line_s1, line_s2, line_s3, line_s4))
        return line_tags

    def gmsh_add_custom_lines(self, point_tags):
        # insert 5 tags at the beginning of the list (4 of the vertices and 1 for starting the count at 1 and not 0)
        pt = [None] * 5 + point_tags
        # add center square
        l1 = self.factory.addLine(pt[38], pt[41])
        l2 = self.factory.addLine(pt[41], pt[71])
        l3 = self.factory.addLine(pt[71], pt[68])
        l4 = self.factory.addLine(pt[68], pt[38])
        center_square = (l1, l2, l3, l4)

        # add diagonal lines
        ld1 = self.factory.addLine(pt[38], pt[27])
        ld2 = self.factory.addLine(pt[41], pt[32])
        ld3 = self.factory.addLine(pt[71], pt[82])
        ld4 = self.factory.addLine(pt[68], pt[77])

        # add 4 squares connecting to the diagonal lines
        l10 = self.factory.addLine(pt[27], pt[26])
        l11 = self.factory.addLine(pt[26], pt[16])
        l12 = self.factory.addLine(pt[16], pt[17])
        l13 = self.factory.addLine(pt[17], pt[27])

        l21 = self.factory.addLine(pt[32], pt[33])
        l22 = self.factory.addLine(pt[33], pt[23])
        l23 = self.factory.addLine(pt[23], pt[22])
        l24 = self.factory.addLine(pt[22], pt[32])

        l31 = self.factory.addLine(pt[82], pt[83])
        l32 = self.factory.addLine(pt[83], pt[93])
        l33 = self.factory.addLine(pt[93], pt[92])
        l34 = self.factory.addLine(pt[92], pt[82])

        l41 = self.factory.addLine(pt[77], pt[76])
        l42 = self.factory.addLine(pt[76], pt[86])
        l43 = self.factory.addLine(pt[86], pt[87])
        l44 = self.factory.addLine(pt[87], pt[77])
        border_squares = (
            l10,
            l11,
            l12,
            l13,
            l21,
            l22,
            l23,
            l24,
            l31,
            l32,
            l33,
            l34,
            l41,
            l42,
            l43,
            l44,
        )

        # center trapezoids
        l51 = self.factory.addLine(pt[38], pt[29])
        l52 = self.factory.addLine(pt[29], pt[30])
        l53 = self.factory.addLine(pt[30], pt[41])

        l61 = self.factory.addLine(pt[41], pt[52])
        l62 = self.factory.addLine(pt[52], pt[62])
        l63 = self.factory.addLine(pt[62], pt[71])

        l71 = self.factory.addLine(pt[71], pt[80])
        l72 = self.factory.addLine(pt[80], pt[79])
        l73 = self.factory.addLine(pt[79], pt[68])

        l81 = self.factory.addLine(pt[68], pt[57])
        l82 = self.factory.addLine(pt[57], pt[47])
        l83 = self.factory.addLine(pt[47], pt[38])
        center_trapezoids = (l51, l52, l53, l61, l62, l63, l71, l72, l73, l81, l82, l83)

        # diagonal trapezoids
        l911 = self.factory.addLine(pt[38], pt[18])
        l912 = self.factory.addLine(pt[18], pt[17])
        l913 = self.factory.addLine(pt[18], pt[19])
        l914 = self.factory.addLine(pt[19], pt[29])

        l921 = self.factory.addLine(pt[41], pt[21])
        l922 = self.factory.addLine(pt[21], pt[22])
        l923 = self.factory.addLine(pt[21], pt[20])
        l924 = self.factory.addLine(pt[20], pt[30])

        l931 = self.factory.addLine(pt[41], pt[43])
        l932 = self.factory.addLine(pt[43], pt[53])
        l933 = self.factory.addLine(pt[53], pt[52])
        l934 = self.factory.addLine(pt[43], pt[33])

        l941 = self.factory.addLine(pt[71], pt[73])
        l942 = self.factory.addLine(pt[73], pt[63])
        l943 = self.factory.addLine(pt[63], pt[62])
        l944 = self.factory.addLine(pt[73], pt[83])

        l951 = self.factory.addLine(pt[71], pt[91])
        l952 = self.factory.addLine(pt[91], pt[92])
        l953 = self.factory.addLine(pt[91], pt[90])
        l954 = self.factory.addLine(pt[90], pt[80])

        l951 = self.factory.addLine(pt[68], pt[88])
        l952 = self.factory.addLine(pt[88], pt[87])
        l953 = self.factory.addLine(pt[88], pt[89])
        l954 = self.factory.addLine(pt[89], pt[79])

        l961 = self.factory.addLine(pt[68], pt[66])
        l962 = self.factory.addLine(pt[66], pt[56])
        l963 = self.factory.addLine(pt[56], pt[57])
        l964 = self.factory.addLine(pt[66], pt[76])

        l971 = self.factory.addLine(pt[38], pt[36])
        l972 = self.factory.addLine(pt[36], pt[46])
        l973 = self.factory.addLine(pt[46], pt[47])
        l974 = self.factory.addLine(pt[36], pt[26])
        trapezoids = (
            l911,
            l912,
            l913,
            l914,
            l921,
            l922,
            l923,
            l924,
            l931,
            l932,
            l933,
            l934,
            l941,
            l942,
            l943,
            l944,
            l951,
            l952,
            l953,
            l954,
            l961,
            l962,
            l963,
            l964,
            l971,
            l972,
            l973,
            l974,
        )

        # close squares of 1st level
        l975 = self.factory.addLine(pt[19], pt[20])
        l976 = self.factory.addLine(pt[53], pt[63])
        l977 = self.factory.addLine(pt[90], pt[89])
        l978 = self.factory.addLine(pt[56], pt[46])
        center_squares_first_lvl = (l975, l976, l977, l978)

        # square of 2nd level
        line_tags_2nd_lvl = []
        for tag in range(6, len(pt[:15])):
            line_tag_s = self.factory.addLine(pt[tag - 1], pt[tag])
            line_tags_2nd_lvl.append(line_tag_s)
        for tag in pt[16:-10:10]:
            line_tag_s = self.factory.addLine(pt[tag - 1], pt[tag])
            line_tags_2nd_lvl.append(line_tag_s)
        for tag in pt[24:-10:10]:
            line_tag_s = self.factory.addLine(pt[tag - 1], pt[tag])
            line_tags_2nd_lvl.append(line_tag_s)
        for tag in range(pt[-9], len(pt)):
            line_tag_s = self.factory.addLine(tag - 1, tag)
            line_tags_2nd_lvl.append(line_tag_s)
        for tag in pt[16:24]:
            line_tag_s = self.factory.addLine(tag - 10, tag)
            line_tags_2nd_lvl.append(line_tag_s)
        for tag in pt[96:104]:
            line_tag_s = self.factory.addLine(tag - 10, tag)
            line_tags_2nd_lvl.append(line_tag_s)
        for tag in pt[15::10]:
            line_tag_s = self.factory.addLine(tag - 10, tag)
            line_tags_2nd_lvl.append(line_tag_s)
        for tag in pt[24::10]:
            line_tag_s = self.factory.addLine(tag - 10, tag)
            line_tags_2nd_lvl.append(line_tag_s)

        return center_square

    def skin_gmsh_points(self, point_tags):
        sqrt_shape = int(np.sqrt(len(point_tags)))
        windows = view_as_windows(
            np.array(point_tags, dtype=int).reshape((sqrt_shape, sqrt_shape)),
            window_shape=(2, 2),
            step=1,
        )
        return windows

    def main(self):
        # TODO: remove after testing
        ###############################
        gmsh.initialize()
        a = 10
        self.factory.addPoint(a, 0, 0, -1)
        self.factory.addPoint(0, a, 0, -1)
        self.factory.addPoint(-a, 0, 0, -1)
        self.factory.addPoint(0, -a, 0, -1)
        self.factory.synchronize()
        ###############################

        SQUARE_SIZE_0 = int(1)
        MAX_SUBDIVISIONS = int(3)

        # * 1. get the vertices coordinates from self.vertices_tags
        vertices_coords = self.get_vertices_coords()
        # * 2. get the center of mass of the vertices
        center_of_mass = self.center_of_mass(vertices_coords)
        # * 3. create the vertices of the squares
        vertices = np.empty((0, 3))
        for subs in range(1, MAX_SUBDIVISIONS):
            square = self.create_subvertices(center_of_mass, subs, SQUARE_SIZE_0)
            vertices = np.concatenate((vertices, square), axis=0)
        if self.SHOW_PLOT:
            self.plot_vertices(vertices)
        # * 4. sort the vertices in a grid
        vertices_sorted = self.vertices_grid_cleanup(vertices)
        if self.SHOW_PLOT:
            self.plot_vertices(vertices_sorted)
        # * 5. Calculate transformation matrix
        M = self.get_affine_transformation_matrix(vertices_coords, vertices_sorted)
        transformed_coords = self.set_transformation_matrix(M, vertices_sorted)
        # * 6. Create GMSH points
        point_tags = self.gmsh_add_points(transformed_coords)
        # skin point_tags
        windows_1 = self.skin_gmsh_points(point_tags)  # TODO: change to blocks
        # # * 7. Create GMSH lines
        line_tags = self.gmsh_add_custom_lines(point_tags)
        # line_tags = self.gmsh_add_line(windows_1)
        # # * 8. Create GMSH surfaces
        # surf_tags = self.gmsh_add_surfaces(line_tags)
        self.factory.synchronize()
        gmsh.fltk.run()
        gmsh.finalize()


'''
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
'''

if "__main__" == __name__:
    trab_refinement = QuadRefinement(vertices_tags=[1, 2, 3, 4], nb_layers=1, DIM=2)
    trab_refinement.main()
