"""
Quad refinement scheme for the inner trabecular region
Author: Simone Poncioni, MSB
Date: 04.2023

https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array
https://blender.stackexchange.com/questions/230534/fastest-way-to-skin-a-grid
"""
import time
from itertools import chain
import logging

import cv2
import gmsh
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import view_as_windows
from cython_functions import find_closed_curve as fcc

LOGGING_NAME = "SIMONE"
# flake8: noqa: E402


class QuadRefinement:
    def __init__(
        self,
        vertices_tags: list[int],
        nb_layers: int,
        DIM: int = 2,
        SQUARE_SIZE_0_MM: int = 1,
        MAX_SUBDIVISIONS: int = 3,
    ):
        self.vertices_tags = np.array(vertices_tags)
        self.nb_layers = int(nb_layers)
        self.DIM = int(DIM)
        self.model = gmsh.model
        self.factory = gmsh.model.occ
        self.EDGE_LENGTH = float(1)
        self.SHOW_PLOT = bool(False)
        self.logger = logging.getLogger(LOGGING_NAME)
        self.SQUARE_SIZE_0_MM = SQUARE_SIZE_0_MM
        self.MAX_SUBDIVISIONS = MAX_SUBDIVISIONS
        self.logger.setLevel(logging.DEBUG)

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
            self.logger.debug(subline_tags)
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

    def gmsh_add_plane_surface(self, line_tags):
        # create surface from the center square
        center_cloop = self.factory.addCurveLoop(line_tags)
        center_surf = self.factory.addPlaneSurface([center_cloop])
        return center_surf

    def gmsh_add_custom_lines(self, point_tags):
        # insert 5 tags at the beginning of the list (4 of the vertices and 1 for starting the count at 1 and not 0)
        pt = [None] * 5 + point_tags
        # add center square
        l1 = self.factory.addLine(pt[38], pt[41])
        l2 = self.factory.addLine(pt[41], pt[71])
        l3 = self.factory.addLine(pt[71], pt[68])
        l4 = self.factory.addLine(pt[68], pt[38])
        center_square = [l1, l2, l3, l4]

        # add diagonal lines
        ld1 = self.factory.addLine(pt[38], pt[27])
        ld2 = self.factory.addLine(pt[41], pt[32])
        ld3 = self.factory.addLine(pt[71], pt[82])
        ld4 = self.factory.addLine(pt[68], pt[77])
        diagonal_lines = [ld1, ld2, ld3, ld4]

        # add 4 squares connecting to the diagonal lines
        l10 = self.factory.addLine(pt[27], pt[26])
        l11 = self.factory.addLine(pt[26], pt[16])
        l12 = self.factory.addLine(pt[16], pt[17])
        l13 = self.factory.addLine(pt[17], pt[27])

        l20 = self.factory.addLine(pt[32], pt[33])
        l21 = self.factory.addLine(pt[33], pt[23])
        l22 = self.factory.addLine(pt[23], pt[22])
        l23 = self.factory.addLine(pt[22], pt[32])

        l30 = self.factory.addLine(pt[82], pt[83])
        l31 = self.factory.addLine(pt[83], pt[93])
        l32 = self.factory.addLine(pt[93], pt[92])
        l33 = self.factory.addLine(pt[92], pt[82])

        l40 = self.factory.addLine(pt[77], pt[76])
        l41 = self.factory.addLine(pt[76], pt[86])
        l42 = self.factory.addLine(pt[86], pt[87])
        l43 = self.factory.addLine(pt[87], pt[77])
        border_squares = [
            l10,
            l11,
            l12,
            l13,
            l20,
            l21,
            l22,
            l23,
            l30,
            l31,
            l32,
            l33,
            l40,
            l41,
            l42,
            l43,
        ]

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
        center_trapezoids = [l51, l52, l53, l61, l62, l63, l71, l72, l73, l81, l82, l83]

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

        l961 = self.factory.addLine(pt[68], pt[88])
        l962 = self.factory.addLine(pt[88], pt[87])
        l963 = self.factory.addLine(pt[88], pt[89])
        l964 = self.factory.addLine(pt[89], pt[79])

        l971 = self.factory.addLine(pt[68], pt[66])
        l972 = self.factory.addLine(pt[66], pt[56])
        l973 = self.factory.addLine(pt[56], pt[57])
        l974 = self.factory.addLine(pt[66], pt[76])

        l981 = self.factory.addLine(pt[38], pt[36])
        l982 = self.factory.addLine(pt[36], pt[46])
        l983 = self.factory.addLine(pt[46], pt[47])
        l984 = self.factory.addLine(pt[36], pt[26])
        trapezoids = [
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
            l981,
            l982,
            l983,
            l984,
        ]

        # close squares of 1st level
        l975 = self.factory.addLine(pt[19], pt[20])
        l976 = self.factory.addLine(pt[53], pt[63])
        l977 = self.factory.addLine(pt[90], pt[89])
        l978 = self.factory.addLine(pt[56], pt[46])
        center_squares_first_lvl = [l975, l976, l977, l978]

        line_tags = (
            center_square,
            diagonal_lines,
            border_squares,
            center_trapezoids,
            trapezoids,
            center_squares_first_lvl,
        )
        return line_tags

    def gmsh_skin_points(self, point_tags):
        sqrt_shape = int(np.sqrt(len(point_tags)))
        windows = view_as_windows(
            np.array(point_tags, dtype=int).reshape((sqrt_shape, sqrt_shape)),
            window_shape=(2, 2),
            step=1,
        )
        return windows

    def gmsh_add_surfs(self, l_tags):
        """l_tags: list of line tags"""

        surf_tags = []
        center_surf = self.gmsh_add_plane_surface(l_tags[0])
        surf_tags.append(center_surf)

        border_squares_split = np.array_split(l_tags[2], 4)
        for border_tags in border_squares_split:
            _surf = self.gmsh_add_plane_surface(border_tags)
            surf_tags.append(_surf)

        center_trapezoids_split = np.array_split(l_tags[3], 4)
        # add center_square to center_trapezoids_split
        for i, center_line_tag in enumerate(l_tags[0]):
            center_trapezoids_split[i] = np.append(
                center_trapezoids_split[i], center_line_tag
            )

        for trapezoid in center_trapezoids_split:
            _surf = self.gmsh_add_plane_surface(trapezoid)
            surf_tags.append(_surf)

        cl_1 = np.array([l_tags[1][0], l_tags[2][3], l_tags[4][0], l_tags[4][1]])
        cl_2 = np.array([l_tags[1][1], l_tags[2][4], l_tags[4][11], l_tags[4][8]])
        cl_3 = np.array([l_tags[1][2], l_tags[2][8], l_tags[4][12], l_tags[4][15]])
        cl_4 = np.array([l_tags[1][3], l_tags[2][12], l_tags[4][24], l_tags[4][27]])

        cl_5 = np.array([l_tags[1][0], l_tags[2][0], l_tags[4][31], l_tags[4][28]])
        cl_6 = np.array([l_tags[1][1], l_tags[2][7], l_tags[4][5], l_tags[4][4]])
        cl_7 = np.array([l_tags[1][2], l_tags[2][11], l_tags[4][17], l_tags[4][16]])
        cl_8 = np.array([l_tags[1][3], l_tags[2][15], l_tags[4][21], l_tags[4][20]])

        cl_9 = np.array([l_tags[3][0], l_tags[4][0], l_tags[4][2], l_tags[4][3]])
        cl_10 = np.array([l_tags[3][2], l_tags[4][7], l_tags[4][6], l_tags[4][4]])
        cl_11 = np.array([l_tags[3][3], l_tags[4][8], l_tags[4][9], l_tags[4][10]])
        cl_12 = np.array([l_tags[3][5], l_tags[4][14], l_tags[4][13], l_tags[4][12]])
        cl_13 = np.array([l_tags[3][6], l_tags[4][16], l_tags[4][18], l_tags[4][19]])
        cl_14 = np.array([l_tags[3][8], l_tags[4][23], l_tags[4][22], l_tags[4][20]])
        cl_15 = np.array([l_tags[3][9], l_tags[4][24], l_tags[4][25], l_tags[4][26]])
        cl_16 = np.array([l_tags[3][11], l_tags[4][30], l_tags[4][29], l_tags[4][28]])

        cl_17 = np.array([l_tags[3][1], l_tags[4][3], l_tags[5][0], l_tags[4][7]])
        cl_18 = np.array([l_tags[3][4], l_tags[4][10], l_tags[5][1], l_tags[4][14]])
        cl_19 = np.array([l_tags[3][7], l_tags[4][19], l_tags[5][2], l_tags[4][23]])
        cl_20 = np.array([l_tags[3][10], l_tags[4][26], l_tags[5][3], l_tags[4][30]])

        loops = [
            cl_1,
            cl_2,
            cl_3,
            cl_4,
            cl_5,
            cl_6,
            cl_7,
            cl_8,
            cl_9,
            cl_10,
            cl_11,
            cl_12,
            cl_13,
            cl_14,
            cl_15,
            cl_16,
            cl_17,
            cl_18,
            cl_19,
            cl_20,
        ]
        for curve_loop in loops:
            _surf = self.gmsh_add_plane_surface(curve_loop)
            surf_tags.append(_surf)
        return surf_tags

    def gmsh_make_transfinite(self, line_tags, surf_tags, n_trans):
        self.factory.synchronize()
        for _, subset in enumerate(line_tags):
            subset = list(map(int, subset))
            for line in subset:
                self.model.mesh.setTransfiniteCurve(line, n_trans, "Progression", 1.0)

        for surf in surf_tags:
            self.model.mesh.setTransfiniteSurface(surf)
        return None

    def gmsh_get_unique_surfaces(self, curve_loop):
        all_surfaces = []
        for curve in curve_loop:
            adjacent_surfaces, _ = self.model.getAdjacencies(1, curve)
            all_surfaces.append(adjacent_surfaces)

        # combine all surfaces into one list with no duplicates
        all_surfaces_flat = [item for sublist in all_surfaces for item in sublist]
        unique_surfaces = np.unique(all_surfaces_flat)
        return unique_surfaces

    def gmsh_add_outer_connectivity(self, line_tags, initial_point_tags, point_tags):
        # insert 5 tags at the beginning of the list (4 of the vertices and 1 for starting the count at 1 and not 0)
        pt = [None] * 5 + point_tags

        # external lines
        ext_lines = []
        for i in range(0, 4):
            _line = self.factory.addLine(
                initial_point_tags[i], initial_point_tags[i - 1]
            )
            ext_lines.append(_line)

        # connecting lines
        connecting_lines = [
            [initial_point_tags[0], pt[93]],
            [initial_point_tags[1], pt[86]],
            [initial_point_tags[2], pt[16]],
            [initial_point_tags[3], pt[23]],
        ]

        # + 4
        # line_01 = self.factory.addLine(pt[16], pt[15])
        # line_02 = self.factory.addLine(pt[16], pt[6])
        line_01 = self.factory.addLine(connecting_lines[0][0], connecting_lines[0][1])
        line_02 = self.factory.addLine(connecting_lines[1][0], connecting_lines[1][1])
        line_03 = self.factory.addLine(connecting_lines[2][0], connecting_lines[2][1])
        line_04 = self.factory.addLine(connecting_lines[3][0], connecting_lines[3][1])
        conn_lines = [line_01, line_02, line_03, line_04]

        line_tags_new = line_tags + (
            ext_lines,
            conn_lines,
        )

        ext_lines = list(map(int, ext_lines))
        conn_lines = list(map(int, conn_lines))

        # make the lines transfinite
        self.factory.synchronize()
        for line in ext_lines:
            self.model.mesh.setTransfiniteCurve(line, 8, "Progression", 1.0)
        for line in conn_lines:
            self.model.mesh.setTransfiniteCurve(line, 1, "Progression", 1.0)

        cl_s_01 = [
            ext_lines[0],
            conn_lines[0],
            line_tags[2][9],
            line_tags[4][15],
            line_tags[4][13],
            line_tags[5][1],
            line_tags[4][9],
            line_tags[4][11],
            line_tags[2][5],
            conn_lines[3],
        ]

        cl_s_02 = [
            ext_lines[1],
            conn_lines[1],
            line_tags[2][14],
            line_tags[4][21],
            line_tags[4][22],
            line_tags[5][2],
            line_tags[4][18],
            line_tags[4][17],
            line_tags[2][10],
            conn_lines[0],
        ]

        cl_s_03 = [
            ext_lines[2],
            conn_lines[2],
            line_tags[2][1],
            line_tags[4][31],
            line_tags[4][29],
            line_tags[5][3],
            line_tags[4][25],
            line_tags[4][27],
            line_tags[2][13],
            conn_lines[1],
        ]

        cl_s_04 = [
            ext_lines[3],
            conn_lines[3],
            line_tags[2][6],
            line_tags[4][5],
            line_tags[4][6],
            line_tags[5][0],
            line_tags[4][2],
            line_tags[4][1],
            line_tags[2][2],
            conn_lines[2],
        ]

        curve_loops_line_tags = [cl_s_01, cl_s_02, cl_s_03, cl_s_04]
        curve_loops = []
        for cl in curve_loops_line_tags:
            _cl = self.factory.addCurveLoop(cl)
            curve_loops.append(_cl)

        ext_surf_tags = []
        for cl in curve_loops:
            _surf = self.factory.addPlaneSurface([cl])
            ext_surf_tags.append(_surf)

        # insert connecting line to connecting_lines_app
        connecting_lines.insert(0, connecting_lines[-1])
        self.factory.synchronize()
        ext_surf_tags = list(map(int, ext_surf_tags))
        corners = []
        for i in range(1, len(ext_surf_tags) + 1):
            corners_s = (
                connecting_lines[i][0],
                connecting_lines[i][1],
                connecting_lines[i - 1][1],
                connecting_lines[i - 1][0],
            )
            self.logger.debug(ext_surf_tags[i - 1])
            self.logger.debug(corners_s)
            self.model.mesh.setTransfiniteSurface(
                ext_surf_tags[i - 1],
                cornerTags=corners_s,
            )
            corners.append(corners_s)
        return line_tags_new, ext_surf_tags, curve_loops_line_tags, corners

    def create_connecting_surfaces(self, curve_loops_line_tags):
        self.factory.synchronize()
        surface_loops = []
        surface_loops_t = []
        for i in range(1, len(curve_loops_line_tags)):
            for j, _ in enumerate(curve_loops_line_tags[i - 1]):
                unique_surfaces_s = self.gmsh_get_unique_surfaces(
                    curve_loops_line_tags[i - 1][j]
                )
                unique_surface_next = self.gmsh_get_unique_surfaces(
                    curve_loops_line_tags[i][j]
                )
                surfaces_app = np.append(unique_surfaces_s, unique_surface_next)
                m = np.zeros_like(surfaces_app, dtype=bool)
                m[np.unique(surfaces_app, return_index=True)[1]] = True
                surface_loops_t.append(surfaces_app[~m])
            surface_loops.append(surface_loops_t)
        return surface_loops

    def quad_refinement(self):
        SQUARE_SIZE_0_MM = self.SQUARE_SIZE_0_MM
        MAX_SUBDIVISIONS = self.MAX_SUBDIVISIONS

        # * 1. get the vertices coordinates from self.vertices_tags
        vertices_coords = self.get_vertices_coords()
        # * 2. get the center of mass of the vertices
        center_of_mass = self.center_of_mass(vertices_coords)
        # * 3. create the vertices of the squares
        vertices = np.empty((0, 3))
        for subs in range(1, MAX_SUBDIVISIONS):
            square = self.create_subvertices(center_of_mass, subs, SQUARE_SIZE_0_MM)
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
        # * 7. Create GMSH lines
        line_tags = self.gmsh_add_custom_lines(point_tags)
        # * 8. Create GMSH surfaces
        surf_tags = self.gmsh_add_surfs(line_tags)
        self.gmsh_make_transfinite(line_tags, surf_tags, 1)
        # * 9. Add outer connectivity
        (
            line_tags,
            ext_surf_tags,
            curve_loops_line_tags,
            corners_external,
        ) = self.gmsh_add_outer_connectivity(line_tags, self.vertices_tags, point_tags)
        [surf_tags.append(ext_surf_tag) for ext_surf_tag in ext_surf_tags]
        return point_tags, line_tags, surf_tags, curve_loops_line_tags, corners_external

    def get_adjacent_points(self, surf_tag: str) -> list:
        _, adjacecies_down = gmsh.model.getAdjacencies(self.DIM, surf_tag)
        adj_points = []
        for line in adjacecies_down:
            _, pt_tags_adj_down = gmsh.model.getAdjacencies(self.DIM - 1, line)
            adj_points.append(pt_tags_adj_down[0])
        return adj_points

    def flatten(self, arr):
        return [item for sublist in arr for item in sublist]

    def common(self, arr):
        u, c = np.unique(arr, return_counts=True)
        return u[c > 1]

    def get_common_lines(self, arr):
        lines = []
        for _, point in enumerate(arr):
            adj_up, _ = gmsh.model.getAdjacencies(0, point)
            lines.append(adj_up.tolist())
            self.logger.debug(f"point {point} has adj lines {adj_up}")
        lines_flatten = self.flatten(lines)
        lines_common = self.common(lines_flatten)
        return lines_common

    def create_intersurface_connection(self, point_tags: list[int]) -> list[int]:
        line_tags_intersurf = []
        for i in range(1, len(point_tags)):
            start_stop = []
            for j in range(len(point_tags[i])):
                for m in range(1, len(point_tags[i][j]) + 1):
                    start_point_0 = point_tags[i - 1][j][m - 1]
                    end_point_0 = point_tags[i][j][m - 1]
                    if m == len(point_tags[i][j]):
                        start_point_1 = point_tags[i - 1][j][0]
                        end_point_1 = point_tags[i][j][0]
                    else:
                        start_point_1 = point_tags[i - 1][j][m]
                        end_point_1 = point_tags[i][j][m]
                    start_stop.append(
                        [
                            start_point_0,
                            end_point_0,
                            start_point_1,
                            end_point_1,
                        ]
                    )
            st = np.array(start_stop).reshape((-1, 4))
            points_0 = st[:, :2]
            _, idx_0 = np.unique(points_0, return_index=True, axis=0)
            unique_st = st[idx_0]

            line_tags_intersurf_a = []
            for _, point in enumerate(unique_st):
                _line = self.factory.addLine(point[0], point[1], tag=-1)
                line_tags_intersurf_a.append(_line)
            line_tags_intersurf.append(line_tags_intersurf_a)
            self.factory.synchronize()
        return line_tags_intersurf

    def create_line_dict(self, lines_lower_surf, lines_upper_surf, lines_intersurf):
        # Step 1: Create a dictionary of all lines and their corresponding points
        lines_lower_dict = {}
        lines_upper_dict = {}
        lines_intersurf_dict = {}
        for subset in lines_lower_surf:
            for line in subset:
                _, line_points = gmsh.model.getAdjacencies(1, line)
                lines_lower_dict[line] = line_points
        for subset in lines_upper_surf:
            for line in subset:
                _, line_points = gmsh.model.getAdjacencies(1, line)
                lines_upper_dict[line] = line_points
        for line in lines_intersurf:
            _, line_points = gmsh.model.getAdjacencies(1, line)
            lines_intersurf_dict[line] = line_points
        return lines_lower_dict, lines_upper_dict, lines_intersurf_dict

    def create_surf_dict(self, surf_lower, surf_upper, surf_inter):
        surf_lower_dict = {}
        surf_upper_dict = {}
        surf_inter_dict = {}
        for surf in surf_lower:
            _, surf_lines = gmsh.model.getAdjacencies(2, surf)
            surf_lower_dict[surf] = surf_lines
        for surf in surf_upper:
            _, surf_lines = gmsh.model.getAdjacencies(2, surf)
            surf_upper_dict[surf] = surf_lines
        for surf in surf_inter:
            _, surf_lines = gmsh.model.getAdjacencies(2, surf)
            surf_inter_dict[surf] = surf_lines
        return surf_lower_dict, surf_upper_dict, surf_inter_dict

    def check_closed_surface_loop(
        self, lines_lower_dict: dict, lines_upper_dict: dict, lines_intersurf_dict: dict
    ) -> dict:
        surf_loop_dict = {}
        index = 1
        for key1, value1 in lines_lower_dict.items():
            key_pairs = []
            for key2, value2 in lines_intersurf_dict.items():
                if any(item in value2 for item in value1):
                    self.logger.debug(f"Found a matching value in {key1} and {key2}")
                    key_pairs.append(key2)
            surf_loop = [key1] + key_pairs

            surf_loop_dict[index] = surf_loop
            index += 1

        # every line in surf_loop_dict should be contained in 4 surfaces made from lines_intersurf_dict --> check the combinations
        for key3, value3 in lines_upper_dict.items():
            key_pairs_2 = []
            for key2, value2 in lines_intersurf_dict.items():
                if any(item in value2 for item in value3):
                    self.logger.debug(f"Found a matching value in {key3} and {key2}")
                    key_pairs_2.append(key2)

            # check if all values in key_pairs_two are in item, if yes, append key3 to surf_loop_dict
            for _, item in surf_loop_dict.items():
                if all(key_pair in item for key_pair in key_pairs_2):
                    item.append(key3)
                    self.logger.debug(f"Found a matching value: {item}")
        return surf_loop_dict

    def add_curve_loop(self, curve_loop_tags):
        return self.factory.addCurveLoop(curve_loop_tags, tag=-1)

    def gmsh_add_surface(self, curve_loop):
        return self.factory.addPlaneSurface(curve_loop, tag=-1)

    def add_surface_loop(self, surface_loop_tags):
        return self.factory.addSurfaceLoop(surface_loop_tags, sewing=False, tag=-1)

    def gmsh_add_volume(self, surface_loop):
        return self.factory.addVolume(surface_loop, tag=-1)


if "__main__" == __name__:
    logger = logging.getLogger(LOGGING_NAME)
    gmsh.initialize()
    a = 20
    c1 = 1
    c2 = 1
    # ? make sure of the point order (has to be clockwise)

    outer_point_tags = []
    for i in range(0, 10, 9):
        d1 = c1 - (0 * i * c1)
        d2 = c2 - (0 * i * c2)
        point_01 = gmsh.model.occ.addPoint(d1 * a, 0, i, -1)
        point_02 = gmsh.model.occ.addPoint(0, -d2 * a, i, -1)
        point_03 = gmsh.model.occ.addPoint(-d1 * a, 0, i, -1)
        point_04 = gmsh.model.occ.addPoint(0, d2 * a, i, -1)
        initial_point_tags = [point_01, point_02, point_03, point_04]
        outer_point_tags.append(initial_point_tags)
    ###############################

    point_tags = []
    line_tags = []
    line_tags_external = []
    surf_tags = []
    surf_tags_internal = []
    surf_loops = []
    corners_external = []
    for initial_point_tags in outer_point_tags:
        trab_refinement = QuadRefinement(
            vertices_tags=initial_point_tags,
            nb_layers=1,
            DIM=2,
            SQUARE_SIZE_0_MM=1,
            MAX_SUBDIVISIONS=3,
        )
        (
            trab_refinement_point_tags,
            trab_refinement_line_tags,
            trab_refinement_surf_tags,
            trab_refinement_external_curve_loops,
            trab_refinement_corners_external,
        ) = trab_refinement.quad_refinement()

        point_tags.append(trab_refinement_point_tags)
        line_tags.append(trab_refinement_line_tags)
        surf_tags.append(trab_refinement_surf_tags)
        surf_tags_internal.append(trab_refinement_surf_tags)
        line_tags_external.append(trab_refinement_external_curve_loops)
        corners_external.append(trab_refinement_corners_external)

    points_in_surf = []
    plane_surfs = []
    for _slice in surf_tags_internal:
        connected_points = []
        for surf in _slice:
            tag = trab_refinement.get_adjacent_points(surf)
            tag_unique = np.unique(tag)
            connected_points.append(tag_unique)
        points_in_surf.append(connected_points)
        plane_surfs.append(_slice)

    line_tags_intersurf = trab_refinement.create_intersurface_connection(points_in_surf)
    intersurface_surfaces_tags = []
    volume_tags = []
    for _iter in range(1, len(points_in_surf)):
        (
            lines_lower_dict,
            lines_upper_dict,
            lines_intersurf_dict,
        ) = trab_refinement.create_line_dict(
            lines_lower_surf=line_tags[_iter - 1],
            lines_upper_surf=line_tags[_iter],
            lines_intersurf=line_tags_intersurf[_iter - 1],
        )

        start_time = time.time()
        curve_loops = fcc.find_closed_curve_loops(
            lines_lower_dict, lines_upper_dict, lines_intersurf_dict
        )
        end_time = time.time()
        exec_time = end_time - start_time
        trab_refinement.logger.debug(
            f"Time to find closed curve loops: {exec_time:.2f} seconds"
        )
        trab_refinement.factory.synchronize()
        intersurface_surfaces_slice = []
        for cl in curve_loops:
            curve_loop_tag = trab_refinement.add_curve_loop(cl)
            surf = trab_refinement.gmsh_add_surface([curve_loop_tag])
            intersurface_surfaces_slice.append(surf)
        intersurface_surfaces_tags.append(intersurface_surfaces_slice)

        # * ADDING SURFACE LOOPS AND INTERSURFACE VOLUMES
        trab_refinement.factory.synchronize()
        (
            surf_lower_dict,
            surf_upper_dict,
            surf_inter_dict,
        ) = trab_refinement.create_surf_dict(
            plane_surfs[_iter - 1],
            plane_surfs[_iter],
            intersurface_surfaces_slice,
        )
        logger.debug(surf_lower_dict)
        logger.debug(surf_upper_dict)
        logger.debug(surf_inter_dict)
        logger.debug("-----------------")

        start_time = time.time()
        surf_loops_slice = trab_refinement.check_closed_surface_loop(
            surf_lower_dict, surf_upper_dict, surf_inter_dict
        )
        surf_loops.append(surf_loops_slice)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(f"Time to check closed surface loops: {elapsed_time:.2f} seconds")

    external_surface_loops = trab_refinement.create_connecting_surfaces(
        line_tags_external
    )

    gmsh.model.occ.synchronize()
    intersurface_volume_tags = []
    for surf_slice in surf_loops:
        for _, _surf_values in surf_slice.items():
            surf_loop_tag = trab_refinement.add_surface_loop(_surf_values)
            vol_tag = trab_refinement.gmsh_add_volume([surf_loop_tag])
            intersurface_volume_tags.append(vol_tag)
        volume_tags.append(intersurface_volume_tags)

    gmsh.model.occ.synchronize()
    # add external surface loops, because it need 4 faces to work
    # for subsubset in external_surface_loops:
    #     for subset in subsubset:
    #         for surf in subset:
    #             gmsh.model.mesh.setTransfiniteSurface(surf)

    # vol_ext_tag = []
    # for i in range(len(corners_external) - 1):
    #     corners_curr = corners_external[i]
    #     corners_next = corners_external[i + 1]
    #     for j, _ in enumerate(corners_curr):
    #         print(f"corner_curr: {corners_curr[j]}")
    #         print(f"corner_next: {corners_next[j]}")
    #         print("------------")

    #         corners_ll = corners_curr[j] + corners_next[j]
    #         corners_list = [corner for corner in corners_ll]
    #         surf_loop = trab_refinement.add_surface_loop(external_surface_loops[i][j])
    #         vol_ext_s = trab_refinement.gmsh_add_volume([surf_loop])
    #         vol_ext_tag.append(vol_ext_s)

    # TODO: move to upper scope when testing is done
    surfs = list(chain(surf_tags_internal, intersurface_surfaces_tags))
    trab_refinement.factory.synchronize()
    for line_subset in line_tags_intersurf:
        for line in line_subset:
            gmsh.model.mesh.setTransfiniteCurve(line, 2, "Progression", 1.0)
    for surf_subset in surfs:
        for surf in surf_subset:
            gmsh.model.mesh.setTransfiniteSurface(surf)
            gmsh.model.mesh.setRecombine(surf, 2)
    # for intersurf_vols in volume_tags[:-4]:
    #     for vol in intersurf_vols:
    #         gmsh.model.mesh.setTransfiniteVolume(vol)

    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
    # gmsh.option.setNumber("Mesh.Recombine3DLevel", 2)
    # gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # * 10. Create 2D mesh
    trab_refinement.factory.synchronize()
    gmsh.write(
        "99_testing_prototyping/trabecular-refinement/transfinite-volume-tests.geo_unrolled"
    )
    # https://gitlab.onelab.info/gmsh/gmsh/-/issues/1710
    # gmsh.model.mesh.generate(3)
    gmsh.fltk.run()
    gmsh.finalize()
