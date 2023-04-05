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
from skimage.util import view_as_blocks

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
        self.SHOW_PLOT = bool(True)

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

    def add_gmsh_points(self, vertices):
        point_tags = []
        for v in vertices:
            v_tag = self.factory.addPoint(v[0], v[1], v[2], -1)
            point_tags.append(v_tag)
        return point_tags

    def get_affine_transformation_matrix(self, verts_1, point_coords):
        # find 4 outermost vertices of point_coords
        max_x = max(point_coords, key=lambda x: x[0])[0]
        min_x = min(point_coords, key=lambda x: x[0])[0]
        max_y = max(point_coords, key=lambda x: x[1])[1]
        min_y = min(point_coords, key=lambda x: x[1])[1]
        # compose verts_2 with the 4 outermost vertices of point_coords
        verts_2 = np.array(
            [
                [max_x, max_y, 0],
                [min_x, max_y, 0],
                [min_x, min_y, 0],
                [max_x, min_y, 0],
            ],
            dtype=np.float32,
        )

        # ensure c-continuity of verts_1 and verts_2 after slicing (as checked by cv::Mat::checkVector())
        # https://stackoverflow.com/questions/54552289/assertion-error-from-opencv-checkvector-in-python
        verts_1_aff = np.copy(verts_1[:3, :2], order="C")
        verts_2_aff = np.copy(verts_2[:3, :2], order="C")

        if self.SHOW_PLOT:
            plt.figure(figsize=(5, 5))
            plt.scatter(verts_1[:, 0], verts_1[:, 1], color="red")
            plt.scatter(verts_2[:, 0], verts_2[:, 1], color="blue")
            plt.show()
        M = cv2.getAffineTransform(verts_2_aff, verts_1_aff)
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
        vertices_unique = np.unique(vertices, axis=0)  # ! remove after testing
        if self.SHOW_PLOT:
            self.plot_vertices(vertices_unique)
        # * 4. Calculate transformation matrix
        M = self.get_affine_transformation_matrix(vertices_coords, vertices_unique)
        transformed_coords = self.set_transformation_matrix(M, vertices_unique)
        # * 5. Create GMSH points
        transformed_coords_unique = np.unique(transformed_coords, axis=0)
        point_tags = self.add_gmsh_points(transformed_coords_unique)

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
