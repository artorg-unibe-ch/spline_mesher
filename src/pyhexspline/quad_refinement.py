"""
Quad refinement scheme for the inner trabecular region
Author: Simone Poncioni, MSB
Date: 04.2023

https://stackoverflow.com/questions/31527755/extract-blocks-or-patches-from-numpy-array
https://blender.stackexchange.com/questions/230534/fastest-way-to-skin-a-grid
https://gitlab.onelab.info/gmsh/gmsh/-/issues/1710
https://gitlab.onelab.info/gmsh/gmsh/-/issues/2439

"""

import logging
import time
from itertools import chain

import cv2
import gmsh
import matplotlib.pyplot as plt
import numpy as np
from cython_functions import find_closed_curve as fcc
from skimage.util import view_as_windows

LOGGING_NAME = "MESHING"
# flake8: noqa: E402


class QuadRefinement:
    def __init__(
        self,
        nb_layers: int,
        DIM: int = 2,
        SQUARE_SIZE_0_MM: int = 1,
        MAX_SUBDIVISIONS: int = 3,
        ELMS_THICKNESS: int = 2,
    ):
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
        self.elms_thickness = ELMS_THICKNESS

    def center_of_mass(self, vertices_coords):
        """
        Calculate the center of mass of the given vertices.

        This function computes the center of mass for the provided vertices coordinates.

        Args:
            vertices_coords (ndarray): Array of vertices coordinates.

        Returns:
            ndarray: The center of mass coordinates.
        """
        return np.mean(vertices_coords, axis=0)

    def get_vertices_coords(self, vertices_tags):
        """
        Get the coordinates of vertices.

        This function retrieves the coordinates of the specified vertices.

        Args:
            vertices_tags (List[int]): List of vertex tags.

        Returns:
            ndarray: Array of vertex coordinates.
        """
        self.factory.synchronize()
        vertices_coords = []
        for vertex in vertices_tags:
            vertices_coords.append(self.model.getValue(0, vertex, []))
        return np.array(vertices_coords, dtype=np.float32)

    def create_square_grid(self, big_square_size, squares_per_side, origin):
        """
        Create a grid of squares.

        This function generates a grid of squares based on the specified size and number of squares per side.

        Args:
            big_square_size (float): The size of the big square.
            squares_per_side (int): The number of squares per side.
            origin (List[float]): The origin coordinates.

        Returns:
            List[ndarray]: List of vertices coordinates for the grid.
        """

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
        """
        Create subvertices for the grid.

        This function generates subvertices for the grid based on the specified hierarchy and square size.

        Args:
            center_of_mass (ndarray): The center of mass coordinates.
            hierarchy (int): The hierarchy level.
            square_size_0 (float): The initial square size.

        Returns:
            List[ndarray]: List of subvertices coordinates.
        """
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
        """
        Plot the vertices.

        This function plots the provided vertices and annotates their indices.

        Args:
            vertices (List[ndarray]): List of vertices coordinates.

        Returns:
            None
        """
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
        """
        Add points to the Gmsh model.

        This function adds the provided vertices as points to the Gmsh model.

        Args:
            vertices (List[ndarray]): List of vertices coordinates.

        Returns:
            List[int]: List of point tags.
        """
        point_tags = []
        for v in vertices:
            v_tag = self.factory.addPoint(v[0], v[1], v[2], tag=-1)
            point_tags.append(v_tag)
        return point_tags

    def gmsh_add_surfaces(self, line_tags):
        """
        Add surfaces to the Gmsh model.

        This function adds surfaces to the Gmsh model based on the provided line tags.

        Args:
            line_tags (List[int]): List of line tags.

        Returns:
            List[int]: List of surface tags.
        """
        cloops = []
        for subline_tags in line_tags[30:-30]:
            self.logger.debug(subline_tags)
            curve = self.factory.addCurveLoop(subline_tags, tag=-1)
            cloops.append(curve)

        surf_tags = []
        for cloop in cloops:
            surf = self.factory.addPlaneSurface([cloop], tag=-1)
            surf_tags.append(surf)
        return surf_tags

    def get_vertices_minmax(self, point_coords):
        """
        Get the minimum and maximum vertices.

        This function retrieves the minimum and maximum vertices from the provided point coordinates.

        Args:
            point_coords (ndarray): Array of point coordinates.

        Returns:
            ndarray: Array of the four outermost vertices.
        """
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
        """
        Get the affine transformation matrix.

        This function calculates the affine transformation matrix to map the provided vertices to the point coordinates.

        Args:
            verts_1 (ndarray): Array of vertices coordinates.
            point_coords (ndarray): Array of point coordinates.

        Returns:
            ndarray: The affine transformation matrix.
        """
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
        """
        Apply the affine transformation matrix.

        This function applies the affine transformation matrix to the provided point coordinates.

        Args:
            M (ndarray): The affine transformation matrix.
            point_coords (ndarray): Array of point coordinates.

        Returns:
            ndarray: Array of transformed coordinates.
        """
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
        """
        Clean up the vertices grid.

        This function removes duplicate vertices and sorts the remaining vertices.

        Args:
            vertices (List[ndarray]): List of vertices coordinates.

        Returns:
            ndarray: Array of cleaned and sorted vertices.
        """
        vertices_unique = np.unique(
            vertices.round(decimals=4),
            axis=0,
        )
        vertices_sorted = np.array(
            sorted(vertices_unique, key=lambda k: [k[1], k[0]]), dtype=np.float32
        )
        return vertices_sorted

    def gmsh_add_line(self, point_skins):
        """
        Add lines to the Gmsh model.

        This function adds lines to the Gmsh model based on the provided point skins.

        Args:
            point_skins (List[ndarray]): List of point skins.

        Returns:
            List[Tuple[int, int, int, int]]: List of line tags.
        """
        line_tags = []
        _iter = int(1)  # slicing iterator
        for i, skin in enumerate(point_skins[::_iter]):
            for j, _ in enumerate(skin[::_iter]):
                p0 = skin[j * _iter][0][0]
                p1 = skin[j * _iter][0][-1]
                p2 = skin[j * _iter][-1][0]
                p3 = skin[j * _iter][-1][-1]
                line_s1 = self.factory.addLine(p0, p1, tag=-1)
                line_s2 = self.factory.addLine(p0, p2, tag=-1)
                line_s3 = self.factory.addLine(p2, p3, tag=-1)
                line_s4 = self.factory.addLine(p1, p3, tag=-1)
                line_tags.append((line_s1, line_s2, line_s3, line_s4))
        return line_tags

    def gmsh_add_plane_surface(self, line_tags):
        """
        Add a plane surface to the Gmsh model.

        This function creates a plane surface from the provided line tags and adds it to the Gmsh model.

        Args:
            line_tags (List[int]): List of line tags.

        Returns:
            int: The tag of the created plane surface.
        """
        center_cloop = self.factory.addCurveLoop(line_tags, tag=-1)
        center_surf = self.factory.addPlaneSurface([center_cloop], tag=-1)
        return center_surf

    def gmsh_add_custom_lines(self, point_tags):
        """
        Add lines to the Gmsh model of the internal refined structure.

        This function adds custom lines to the Gmsh model based on the provided point tags. It includes
        center squares, diagonal lines, border squares, center trapezoids, diagonal trapezoids, and outer squares.

        Args:
            point_tags (List[int]): List of point tags.

        Returns:
            Tuple[List[int], List[int]]: Tuple containing the line tags and outer square line tags.
        """
        # insert 5 tags at the beginning of the list (4 of the vertices and 1 for starting the count at 1 and not 0)
        pt = [None] * 5 + point_tags
        # add center square
        l1 = self.factory.addLine(pt[38], pt[41], tag=-1)
        l2 = self.factory.addLine(pt[41], pt[71], tag=-1)
        l3 = self.factory.addLine(pt[71], pt[68], tag=-1)
        l4 = self.factory.addLine(pt[68], pt[38], tag=-1)
        center_square = [l1, l2, l3, l4]

        # add diagonal lines
        ld1 = self.factory.addLine(pt[38], pt[27], tag=-1)
        ld2 = self.factory.addLine(pt[41], pt[32], tag=-1)
        ld3 = self.factory.addLine(pt[71], pt[82], tag=-1)
        ld4 = self.factory.addLine(pt[68], pt[77], tag=-1)
        diagonal_lines = [ld1, ld2, ld3, ld4]

        # add 4 squares connecting to the diagonal lines
        l10 = self.factory.addLine(pt[27], pt[26], tag=-1)
        l11 = self.factory.addLine(pt[26], pt[16], tag=-1)
        l12 = self.factory.addLine(pt[16], pt[17], tag=-1)
        l13 = self.factory.addLine(pt[17], pt[27], tag=-1)

        l20 = self.factory.addLine(pt[32], pt[33], tag=-1)
        l21 = self.factory.addLine(pt[33], pt[23], tag=-1)
        l22 = self.factory.addLine(pt[23], pt[22], tag=-1)
        l23 = self.factory.addLine(pt[22], pt[32], tag=-1)

        l30 = self.factory.addLine(pt[82], pt[83], tag=-1)
        l31 = self.factory.addLine(pt[83], pt[93], tag=-1)
        l32 = self.factory.addLine(pt[93], pt[92], tag=-1)
        l33 = self.factory.addLine(pt[92], pt[82], tag=-1)

        l40 = self.factory.addLine(pt[77], pt[76], tag=-1)
        l41 = self.factory.addLine(pt[76], pt[86], tag=-1)
        l42 = self.factory.addLine(pt[86], pt[87], tag=-1)
        l43 = self.factory.addLine(pt[87], pt[77], tag=-1)
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
        l51 = self.factory.addLine(pt[38], pt[29], tag=-1)
        l52 = self.factory.addLine(pt[29], pt[30], tag=-1)
        l53 = self.factory.addLine(pt[30], pt[41], tag=-1)

        l61 = self.factory.addLine(pt[41], pt[52], tag=-1)
        l62 = self.factory.addLine(pt[52], pt[62], tag=-1)
        l63 = self.factory.addLine(pt[62], pt[71], tag=-1)

        l71 = self.factory.addLine(pt[71], pt[80], tag=-1)
        l72 = self.factory.addLine(pt[80], pt[79], tag=-1)
        l73 = self.factory.addLine(pt[79], pt[68], tag=-1)

        l81 = self.factory.addLine(pt[68], pt[57], tag=-1)
        l82 = self.factory.addLine(pt[57], pt[47], tag=-1)
        l83 = self.factory.addLine(pt[47], pt[38], tag=-1)
        center_trapezoids = [l51, l52, l53, l61, l62, l63, l71, l72, l73, l81, l82, l83]

        # diagonal trapezoids
        l911 = self.factory.addLine(pt[38], pt[18], tag=-1)
        l912 = self.factory.addLine(pt[18], pt[17], tag=-1)
        l913 = self.factory.addLine(pt[18], pt[19], tag=-1)
        l914 = self.factory.addLine(pt[19], pt[29], tag=-1)

        l921 = self.factory.addLine(pt[41], pt[21], tag=-1)
        l922 = self.factory.addLine(pt[21], pt[22], tag=-1)
        l923 = self.factory.addLine(pt[21], pt[20], tag=-1)
        l924 = self.factory.addLine(pt[20], pt[30], tag=-1)

        l931 = self.factory.addLine(pt[41], pt[43], tag=-1)
        l932 = self.factory.addLine(pt[43], pt[53], tag=-1)
        l933 = self.factory.addLine(pt[53], pt[52], tag=-1)
        l934 = self.factory.addLine(pt[43], pt[33], tag=-1)

        l941 = self.factory.addLine(pt[71], pt[73], tag=-1)
        l942 = self.factory.addLine(pt[73], pt[63], tag=-1)
        l943 = self.factory.addLine(pt[63], pt[62], tag=-1)
        l944 = self.factory.addLine(pt[73], pt[83], tag=-1)

        l951 = self.factory.addLine(pt[71], pt[91], tag=-1)
        l952 = self.factory.addLine(pt[91], pt[92], tag=-1)
        l953 = self.factory.addLine(pt[91], pt[90], tag=-1)
        l954 = self.factory.addLine(pt[90], pt[80], tag=-1)

        l961 = self.factory.addLine(pt[68], pt[88], tag=-1)
        l962 = self.factory.addLine(pt[88], pt[87], tag=-1)
        l963 = self.factory.addLine(pt[88], pt[89], tag=-1)
        l964 = self.factory.addLine(pt[89], pt[79], tag=-1)

        l971 = self.factory.addLine(pt[68], pt[66], tag=-1)
        l972 = self.factory.addLine(pt[66], pt[56], tag=-1)
        l973 = self.factory.addLine(pt[56], pt[57], tag=-1)
        l974 = self.factory.addLine(pt[66], pt[76], tag=-1)

        l981 = self.factory.addLine(pt[38], pt[36], tag=-1)
        l982 = self.factory.addLine(pt[36], pt[46], tag=-1)
        l983 = self.factory.addLine(pt[46], pt[47], tag=-1)
        l984 = self.factory.addLine(pt[36], pt[26], tag=-1)
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
        l975 = self.factory.addLine(pt[19], pt[20], tag=-1)
        l976 = self.factory.addLine(pt[53], pt[63], tag=-1)
        l977 = self.factory.addLine(pt[90], pt[89], tag=-1)
        l978 = self.factory.addLine(pt[56], pt[46], tag=-1)
        center_squares_first_lvl = [l975, l976, l977, l978]

        # TODO: add 'if' depending on number of subdivisions, now works for only 2 subdivisions
        # assignee: @simoneponcioni
        # add outer squares
        pt = [None] * 4 + pt
        l100 = self.factory.addLine(pt[9], pt[19], tag=-1)
        l101 = self.factory.addLine(pt[19], pt[29], tag=-1)
        l102 = self.factory.addLine(pt[29], pt[39], tag=-1)
        l103 = self.factory.addLine(pt[39], pt[49], tag=-1)
        l104 = self.factory.addLine(pt[49], pt[59], tag=-1)
        l105 = self.factory.addLine(pt[59], pt[69], tag=-1)
        l106 = self.factory.addLine(pt[69], pt[79], tag=-1)
        l107 = self.factory.addLine(pt[79], pt[89], tag=-1)
        l108 = self.factory.addLine(pt[89], pt[99], tag=-1)

        l109 = self.factory.addLine(pt[9], pt[10], tag=-1)
        l110 = self.factory.addLine(pt[19], pt[20], tag=-1)
        l111 = self.factory.addLine(pt[29], pt[30], tag=-1)
        l112 = self.factory.addLine(pt[39], pt[40], tag=-1)
        l113 = self.factory.addLine(pt[49], pt[50], tag=-1)
        l114 = self.factory.addLine(pt[59], pt[60], tag=-1)
        l115 = self.factory.addLine(pt[69], pt[70], tag=-1)
        l116 = self.factory.addLine(pt[79], pt[80], tag=-1)
        l117 = self.factory.addLine(pt[89], pt[90], tag=-1)
        l118 = self.factory.addLine(pt[99], pt[100], tag=-1)

        l120 = self.factory.addLine(pt[10], pt[20], tag=-1)
        l121 = self.factory.addLine(pt[90], pt[100], tag=-1)

        l130 = self.factory.addLine(pt[10], pt[11], tag=-1)
        l131 = self.factory.addLine(pt[11], pt[12], tag=-1)
        l132 = self.factory.addLine(pt[12], pt[13], tag=-1)
        l133 = self.factory.addLine(pt[13], pt[14], tag=-1)
        l134 = self.factory.addLine(pt[14], pt[15], tag=-1)
        l135 = self.factory.addLine(pt[15], pt[16], tag=-1)
        l136 = self.factory.addLine(pt[16], pt[17], tag=-1)
        l137 = self.factory.addLine(pt[17], pt[18], tag=-1)

        l140 = self.factory.addLine(pt[11], pt[21], tag=-1)
        l141 = self.factory.addLine(pt[12], pt[22], tag=-1)
        l142 = self.factory.addLine(pt[13], pt[23], tag=-1)
        l143 = self.factory.addLine(pt[14], pt[24], tag=-1)
        l144 = self.factory.addLine(pt[15], pt[25], tag=-1)
        l145 = self.factory.addLine(pt[16], pt[26], tag=-1)
        l146 = self.factory.addLine(pt[17], pt[27], tag=-1)
        l147 = self.factory.addLine(pt[18], pt[28], tag=-1)

        l148 = self.factory.addLine(pt[27], pt[28], tag=-1)

        l150 = self.factory.addLine(pt[28], pt[38], tag=-1)
        l151 = self.factory.addLine(pt[38], pt[48], tag=-1)
        l152 = self.factory.addLine(pt[48], pt[58], tag=-1)
        l153 = self.factory.addLine(pt[58], pt[68], tag=-1)
        l154 = self.factory.addLine(pt[68], pt[78], tag=-1)
        l155 = self.factory.addLine(pt[78], pt[88], tag=-1)
        l156 = self.factory.addLine(pt[88], pt[98], tag=-1)
        l157 = self.factory.addLine(pt[98], pt[108], tag=-1)

        l160 = self.factory.addLine(pt[37], pt[38], tag=-1)
        l161 = self.factory.addLine(pt[47], pt[48], tag=-1)
        l162 = self.factory.addLine(pt[57], pt[58], tag=-1)
        l163 = self.factory.addLine(pt[67], pt[68], tag=-1)
        l164 = self.factory.addLine(pt[77], pt[78], tag=-1)
        l165 = self.factory.addLine(pt[87], pt[88], tag=-1)
        l166 = self.factory.addLine(pt[97], pt[98], tag=-1)
        l167 = self.factory.addLine(pt[107], pt[108], tag=-1)

        l170 = self.factory.addLine(pt[100], pt[101], tag=-1)
        l171 = self.factory.addLine(pt[101], pt[102], tag=-1)
        l172 = self.factory.addLine(pt[102], pt[103], tag=-1)
        l173 = self.factory.addLine(pt[103], pt[104], tag=-1)
        l174 = self.factory.addLine(pt[104], pt[105], tag=-1)
        l175 = self.factory.addLine(pt[105], pt[106], tag=-1)
        l176 = self.factory.addLine(pt[106], pt[107], tag=-1)

        l180 = self.factory.addLine(pt[91], pt[101], tag=-1)
        l181 = self.factory.addLine(pt[92], pt[102], tag=-1)
        l182 = self.factory.addLine(pt[93], pt[103], tag=-1)
        l183 = self.factory.addLine(pt[94], pt[104], tag=-1)
        l184 = self.factory.addLine(pt[95], pt[105], tag=-1)
        l185 = self.factory.addLine(pt[96], pt[106], tag=-1)
        l186 = self.factory.addLine(pt[97], pt[107], tag=-1)

        outer_square = [
            l100,
            l101,
            l102,
            l103,
            l104,
            l105,
            l106,
            l107,
            l108,
            l109,
            l110,
            l111,
            l112,
            l113,
            l114,
            l115,
            l116,
            l117,
            l118,
            l120,
            l121,
            l130,
            l131,
            l132,
            l133,
            l134,
            l135,
            l136,
            l137,
            l140,
            l141,
            l142,
            l143,
            l144,
            l145,
            l146,
            l147,
            l148,
            l150,
            l151,
            l152,
            l153,
            l154,
            l155,
            l156,
            l157,
            l160,
            l161,
            l162,
            l163,
            l164,
            l165,
            l166,
            l167,
            l170,
            l171,
            l172,
            l173,
            l174,
            l175,
            l176,
            l180,
            l181,
            l182,
            l183,
            l184,
            l185,
            l186,
        ]

        line_tags = (
            center_square,
            diagonal_lines,
            border_squares,
            center_trapezoids,
            trapezoids,
            center_squares_first_lvl,
        )
        return line_tags, outer_square

    def gmsh_skin_points(self, point_tags):
        """
        Generate a grid of point skins.

        This function generates a grid of point skins from the provided point tags.

        Args:
            point_tags (List[int]): List of point tags.

        Returns:
            ndarray: Array of point skins.
        """
        sqrt_shape = int(np.sqrt(len(point_tags)))
        windows = view_as_windows(
            np.array(point_tags, dtype=int).reshape((sqrt_shape, sqrt_shape)),
            window_shape=(2, 2),
            step=1,
        )
        return windows

    def gmsh_add_surfs(self, l_tags):
        """
        Add surfaces to the Gmsh model.

        This function adds surfaces to the Gmsh model based on the provided line tags. It includes
        center surfaces, border squares, center trapezoids, and additional surfaces.

        Args:
            l_tags (List[int]): List of line tags.

        Returns:
            List[int]: List of surface tags.
        """

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
        """
        Set transfinite meshing for lines and surfaces.

        This function sets transfinite meshing for the provided lines and surfaces.

        Args:
            line_tags (List[int]): List of line tags.
            surf_tags (List[int]): List of surface tags.
            n_trans (int): Number of transfinite divisions.

        Returns:
            None
        """
        self.factory.synchronize()
        for _, subset in enumerate(line_tags):
            subset = list(map(int, subset))
            for line in subset:
                self.model.mesh.setTransfiniteCurve(line, n_trans, "Progression", 1.0)

        for surf in surf_tags:
            self.model.mesh.setTransfiniteSurface(surf)
        return None

    def gmsh_get_unique_surfaces(self, curve_loop):
        """
        Get unique surfaces from curve loops.

        This function retrieves unique surfaces from the provided curve loops.

        Args:
            curve_loop (List[int]): List of curve loop tags.

        Returns:
            ndarray: Array of unique surface tags.
        """
        all_surfaces = []
        for curve in curve_loop:
            adjacent_surfaces, _ = self.model.getAdjacencies(1, curve)
            all_surfaces.append(adjacent_surfaces)

        # combine all surfaces into one list with no duplicates
        all_surfaces_flat = [item for sublist in all_surfaces for item in sublist]
        unique_surfaces = np.unique(all_surfaces_flat)
        return unique_surfaces

    def gmsh_add_outer_connectivity(self, line_tags, initial_point_tags, point_tags):
        """
        Add outer connectivity to the Gmsh model.

        This function adds outer connectivity lines and surfaces to the Gmsh model based on the provided
        line tags and point tags.

        Args:
            line_tags (List[int]): List of line tags.
            initial_point_tags (List[int]): List of initial point tags.
            point_tags (List[int]): List of point tags.

        Returns:
            Tuple[List[int], List[int], List[int], List[Tuple[int, int, int, int]]]:
                - Updated line tags.
                - List of external surface tags.
                - List of curve loop line tags.
                - List of corner points for transfinite surfaces.
        """
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
        line_01 = self.factory.addLine(
            connecting_lines[0][0], connecting_lines[0][1], tag=-1
        )
        line_02 = self.factory.addLine(
            connecting_lines[1][0], connecting_lines[1][1], tag=-1
        )
        line_03 = self.factory.addLine(
            connecting_lines[2][0], connecting_lines[2][1], tag=-1
        )
        line_04 = self.factory.addLine(
            connecting_lines[3][0], connecting_lines[3][1], tag=-1
        )
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
        # for line in conn_lines:
        #     self.model.mesh.setTransfiniteCurve(line, 1, "Progression", 1.0)

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
            _cl = self.factory.addCurveLoop(cl, tag=-1)
            curve_loops.append(_cl)

        ext_surf_tags = []
        for cl in curve_loops:
            _surf = self.factory.addPlaneSurface([cl], tag=-1)
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
            # self.model.mesh.setTransfiniteSurface(
            #     ext_surf_tags[i - 1],
            #     cornerTags=corners_s,
            # )
            corners.append(corners_s)
        return line_tags_new, ext_surf_tags, curve_loops_line_tags, corners

    def create_connecting_surfaces(self, curve_loops_line_tags):
        """
        Create connecting surfaces between curve loops.

        This function creates connecting surfaces between the provided curve loops.

        Args:
            curve_loops_line_tags (List[int]): List of curve loop line tags.

        Returns:
            List[ndarray]: List of surface loops.
        """
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

    def close_outer_loop(self, line_tags, outer_square_lines):
        """
        Close the outer loop with surfaces.

        This function closes the outer loop by creating surfaces from the provided line tags and outer square lines.

        Args:
            line_tags (List[int]): List of line tags.
            outer_square_lines (List[int]): List of outer square line tags.

        Returns:
            List[int]: List of surface tags.
        """
        # read all the lines
        self.factory.synchronize()
        line_tags_f = [item for sublist in line_tags for item in sublist]
        line_tags = line_tags_f + outer_square_lines

        # start line indices at 1
        line = [None] + line_tags

        curve_loop_30 = [line[73], line[83], line[92], line[82]]
        curve_loop_31 = [line[74], line[84], line[10], line[83]]
        curve_loop_32 = [line[75], line[85], line[68], line[84]]
        curve_loop_33 = [line[76], line[86], line[66], line[85]]
        curve_loop_34 = [line[77], line[87], line[72], line[86]]
        curve_loop_35 = [line[78], line[88], line[62], line[87]]
        curve_loop_36 = [line[79], line[89], line[64], line[88]]
        curve_loop_37 = [line[80], line[90], line[22], line[89]]
        curve_loop_38 = [line[81], line[91], line[93], line[90]]
        curve_loop_39 = [line[93], line[127], line[134], line[23]]
        curve_loop_40 = [line[134], line[128], line[135], line[58]]
        curve_loop_41 = [line[135], line[129], line[136], line[59]]
        curve_loop_42 = [line[136], line[130], line[137], line[71]]
        curve_loop_43 = [line[137], line[131], line[138], line[55]]
        curve_loop_44 = [line[138], line[132], line[139], line[54]]
        curve_loop_45 = [line[139], line[133], line[140], line[19]]
        curve_loop_46 = [line[140], line[126], line[118], line[125]]
        curve_loop_47 = [line[18], line[125], line[117], line[124]]
        curve_loop_48 = [line[124], line[116], line[123], line[52]]
        curve_loop_49 = [line[123], line[115], line[122], line[50]]
        curve_loop_50 = [line[122], line[114], line[121], line[70]]
        curve_loop_51 = [line[121], line[113], line[120], line[46]]
        curve_loop_52 = [line[120], line[112], line[119], line[48]]
        curve_loop_53 = [line[119], line[111], line[110], line[14]]
        curve_loop_54 = [line[110], line[109], line[101], line[108]]
        curve_loop_55 = [line[108], line[15], line[107], line[100]]
        curve_loop_56 = [line[107], line[42], line[106], line[99]]
        curve_loop_57 = [line[106], line[43], line[105], line[98]]
        curve_loop_58 = [line[105], line[69], line[104], line[97]]
        curve_loop_59 = [line[104], line[39], line[103], line[96]]
        curve_loop_60 = [line[103], line[38], line[102], line[95]]
        curve_loop_61 = [line[102], line[11], line[92], line[94]]

        # concatenate in list all curve loops
        curve_loops_outer = [
            curve_loop_30,
            curve_loop_31,
            curve_loop_32,
            curve_loop_33,
            curve_loop_34,
            curve_loop_35,
            curve_loop_36,
            curve_loop_37,
            curve_loop_38,
            curve_loop_39,
            curve_loop_40,
            curve_loop_41,
            curve_loop_42,
            curve_loop_43,
            curve_loop_44,
            curve_loop_45,
            curve_loop_46,
            curve_loop_47,
            curve_loop_48,
            curve_loop_49,
            curve_loop_50,
            curve_loop_51,
            curve_loop_52,
            curve_loop_53,
            curve_loop_54,
            curve_loop_55,
            curve_loop_56,
            curve_loop_57,
            curve_loop_58,
            curve_loop_59,
            curve_loop_60,
            curve_loop_61,
        ]

        surf = []
        for cl in curve_loops_outer:
            curve_loop = self.factory.addCurveLoop(cl)
            surf_s = self.factory.addPlaneSurface([curve_loop])
            surf.append(surf_s)

        self.model.occ.synchronize()
        for s in surf:
            self.model.mesh.setTransfiniteSurface(s)

        return surf

    def quad_refinement(self, initial_point_tags):
        """
        Perform quadrilateral refinement on the given initial point tags.

        This function refines the quadrilateral mesh based on the initial point tags. It creates subvertices,
        sorts them, applies an affine transformation, and adds points, lines, and surfaces to the Gmsh model.

        Args:
            initial_point_tags (List[int]): List of initial point tags.

        Returns:
            Tuple[List[int], List[int], List[int]]: Tuple containing the point tags, line tags, and surface tags.
        """
        SQUARE_SIZE_0_MM = self.SQUARE_SIZE_0_MM
        MAX_SUBDIVISIONS = self.MAX_SUBDIVISIONS

        self.logger.debug("* 1. Get the vertices coordinates from self.vertices_tags")
        vertices_coords = self.get_vertices_coords(vertices_tags=initial_point_tags)
        self.logger.debug("* 2. get the center of mass of the vertices")
        center_of_mass = self.center_of_mass(vertices_coords)
        self.logger.debug("* 3. create the vertices of the squares")
        vertices = np.empty((0, 3))
        for subs in range(1, MAX_SUBDIVISIONS):
            square = self.create_subvertices(center_of_mass, subs, SQUARE_SIZE_0_MM)
            vertices = np.concatenate((vertices, square), axis=0)
        if self.SHOW_PLOT:
            self.plot_vertices(vertices)
        self.logger.debug("* 4. sort the vertices in a grid")
        vertices_sorted = self.vertices_grid_cleanup(vertices)
        if self.SHOW_PLOT:
            self.plot_vertices(vertices_sorted)
        self.logger.debug("* 5. Calculate transformation matrix")
        M = self.get_affine_transformation_matrix(vertices_coords, vertices_sorted)
        transformed_coords = self.set_transformation_matrix(M, vertices_sorted)
        self.logger.debug("* 6. Create GMSH points")
        point_tags = self.gmsh_add_points(transformed_coords)
        self.logger.debug("* 7. Create GMSH lines")
        line_tags, outer_square_lines = self.gmsh_add_custom_lines(point_tags)

        # make transfinite lines
        self.model.occ.synchronize()
        lines = self.model.getEntities(1)
        for line in lines:
            self.model.mesh.setTransfiniteCurve(line[1], 2)

        self.logger.debug("* 8. Create GMSH surfaces")
        # create curve loops based on the outer square lines
        surf_tags = self.gmsh_add_surfs(line_tags)
        outer_surfs = self.close_outer_loop(line_tags, outer_square_lines)

        point_tags = initial_point_tags + point_tags
        line_tags = line_tags + tuple([outer_square_lines])
        surf_tags = surf_tags + outer_surfs

        return (
            point_tags,
            line_tags,
            surf_tags,
        )

    def get_adjacent_points(self, surf_tag: str) -> list:
        """
        Get adjacent points for a given surface tag.

        This function retrieves the adjacent points for the specified surface tag.

        Args:
            surf_tag (str): The surface tag.

        Returns:
            List[int]: List of adjacent point tags.
        """
        _, adjacecies_down = gmsh.model.getAdjacencies(self.DIM, surf_tag)
        adj_points = []
        for line in adjacecies_down:
            _, pt_tags_adj_down = gmsh.model.getAdjacencies(self.DIM - 1, line)
            adj_points.append(pt_tags_adj_down[0])
        return adj_points

    def flatten(self, arr):
        """
        Flatten a nested list.

        This function flattens a nested list into a single list.

        Args:
            arr (List[List[Any]]): The nested list.

        Returns:
            List[Any]: The flattened list.
        """
        return [item for sublist in arr for item in sublist]

    def common(self, arr):
        """
        Find common elements in an array.

        This function finds the common elements in the provided array.

        Args:
            arr (ndarray): The input array.

        Returns:
            ndarray: Array of common elements.
        """
        u, c = np.unique(arr, return_counts=True)
        return u[c > 1]

    def get_common_lines(self, arr):
        """
        Get common lines from an array of points.

        This function retrieves the common lines from the provided array of points.

        Args:
            arr (List[int]): List of point tags.

        Returns:
            ndarray: Array of common line tags.
        """
        lines = []
        for _, point in enumerate(arr):
            adj_up, _ = gmsh.model.getAdjacencies(0, point)
            lines.append(adj_up.tolist())
            self.logger.debug(f"point {point} has adj lines {adj_up}")
        lines_flatten = self.flatten(lines)
        lines_common = self.common(lines_flatten)
        return lines_common

    def create_intersurface_connection(self, point_tags):
        """
        Create intersurface connections.

        This function creates intersurface connections based on the provided point tags.

        Args:
            point_tags (List[List[int]]): List of point tags for each surface.

        Returns:
            List[List[int]]: List of line tags for intersurface connections.
        """
        line_tags_intersurf = []
        for i in range(1, len(point_tags)):
            start_stop = []
            for j in range(len(point_tags[i])):
                for m in range(1, len(point_tags[i][j]) + 1):
                    start_point_0 = point_tags[i - 1][j][m - 1]
                    end_point_0 = point_tags[i][j][m - 1]
                    if m == len(point_tags[i][j]):
                        start_point_1 = point_tags[i - 1][j][-1]  # point_tags[i][j][0]
                        end_point_1 = point_tags[i][j][-1]  # point_tags[i][j][0]
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
        """
        Create dictionaries of lines and their corresponding points.

        This function creates dictionaries for the lower surface lines, upper surface lines, and intersurface lines,
        mapping each line to its corresponding points.

        Args:
            lines_lower_surf (List[List[int]]): List of line tags for the lower surface.
            lines_upper_surf (List[List[int]]): List of line tags for the upper surface.
            lines_intersurf (List[int]): List of line tags for the intersurface.

        Returns:
            Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
                - Dictionary of lower surface lines and their points.
                - Dictionary of upper surface lines and their points.
                - Dictionary of intersurface lines and their points.
        """
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
        """
        Create dictionaries of surfaces and their corresponding lines.

        This function creates dictionaries for the lower surfaces, upper surfaces, and intersurfaces,
        mapping each surface to its corresponding lines.

        Args:
            surf_lower (List[int]): List of surface tags for the lower surface.
            surf_upper (List[int]): List of surface tags for the upper surface.
            surf_inter (List[int]): List of surface tags for the intersurface.

        Returns:
            Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
                - Dictionary of lower surfaces and their lines.
                - Dictionary of upper surfaces and their lines.
                - Dictionary of intersurfaces and their lines.
        """

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
        """
        Check for closed surface loops.

        This function checks for closed surface loops by matching lines from the lower surface, upper surface,
        and intersurface dictionaries.

        Args:
            lines_lower_dict (Dict[int, List[int]]): Dictionary of lower surface lines and their points.
            lines_upper_dict (Dict[int, List[int]]): Dictionary of upper surface lines and their points.
            lines_intersurf_dict (Dict[int, List[int]]): Dictionary of intersurface lines and their points.

        Returns:
            Dict[int, List[int]]: Dictionary of closed surface loops.
        """
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
        """
        Add a curve loop to the Gmsh model.

        This function adds a curve loop to the Gmsh model based on the provided curve loop tags.

        Args:
            curve_loop_tags (List[int]): List of curve loop tags.

        Returns:
            int: The tag of the created curve loop.
        """
        return self.factory.addCurveLoop(curve_loop_tags, tag=-1)

    def gmsh_add_surface(self, curve_loop):
        """
        Add a surface to the Gmsh model.

        This function adds a surface to the Gmsh model based on the provided curve loop.

        Args:
            curve_loop (int): The curve loop tag.

        Returns:
            int: The tag of the created surface.
        """
        return self.factory.addSurfaceFilling(curve_loop, tag=-1)

    def add_surface_loop(self, surface_loop_tags):
        """
        Add a surface loop to the Gmsh model.

        This function adds a surface loop to the Gmsh model based on the provided surface loop tags.

        Args:
            surface_loop_tags (List[int]): List of surface loop tags.

        Returns:
            int: The tag of the created surface loop.
        """
        return self.factory.addSurfaceLoop(surface_loop_tags, sewing=False, tag=-1)

    def gmsh_add_volume(self, surface_loop):
        """
        Add a volume to the Gmsh model.

        This function adds a volume to the Gmsh model based on the provided surface loop.

        Args:
            surface_loop (int): The surface loop tag.

        Returns:
            int: The tag of the created volume.
        """
        return self.factory.addVolume(surface_loop, tag=-1)

    def exec_quad_refinement(self, outer_point_tags):
        """
        Execute quadrilateral refinement on the given outer point tags.

        This function performs quadrilateral refinement on the provided outer point tags, creating points, lines,
        surfaces, and volumes in the Gmsh model.

        Args:
            outer_point_tags (List[List[int]]): List of lists of point tags, each representing a corner of the structure.

        Returns:
            Tuple[List[List[int]], List[List[int]], List[List[int]]]:
                - List of line tags for intersurface connections.
                - List of surface tags.
                - List of volume tags.
        """
        point_tags = []
        line_tags = []
        surf_tags = []
        surf_loops = []
        for i, initial_point_tags in enumerate(outer_point_tags):
            (
                trab_refinement_point_tags,
                trab_refinement_line_tags,
                trab_refinement_surf_tags,
            ) = self.quad_refinement(initial_point_tags)

            point_tags.append(trab_refinement_point_tags)
            line_tags.append(trab_refinement_line_tags)
            surf_tags.append(trab_refinement_surf_tags)

        # add missing line at the end of the surface
        intersurf_line_last_line = []
        for i in range(1, len(point_tags)):
            ll = self.factory.addLine(point_tags[i - 1][-1], point_tags[i][-1], tag=-1)
            intersurf_line_last_line.append(ll)

        points_in_surf = []
        plane_surfs = []
        for _slice in surf_tags:
            connected_points = []
            for surf in _slice:
                pt_tag = self.get_adjacent_points(surf)
                tag_unique = np.unique(pt_tag)
                connected_points.append(tag_unique)
            points_in_surf.append(connected_points)
            plane_surfs.append(_slice)

        line_tags_intersurf = self.create_intersurface_connection(points_in_surf)
        for i, l_intersurf_subset in enumerate(line_tags_intersurf):
            l_intersurf_subset.extend([intersurf_line_last_line[i]])

        intersurface_surfaces_tags = []
        volume_tags = []
        for _iter in range(1, len(points_in_surf)):
            (
                lines_lower_dict,
                lines_upper_dict,
                lines_intersurf_dict,
            ) = self.create_line_dict(
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
            self.logger.debug(
                f"Time to find closed curve loops: {exec_time:.2f} seconds"
            )
            self.factory.synchronize()

            intersurface_surfaces_slice = []
            for cl in curve_loops:
                curve_loop_tag = self.add_curve_loop(cl)
                surf = self.gmsh_add_surface(curve_loop_tag)
                intersurface_surfaces_slice.append(surf)
            intersurface_surfaces_tags.append(intersurface_surfaces_slice)

            # * ADDING SURFACE LOOPS AND INTERSURFACE VOLUMES
            self.factory.synchronize()
            (
                surf_lower_dict,
                surf_upper_dict,
                surf_inter_dict,
            ) = self.create_surf_dict(
                plane_surfs[_iter - 1],
                plane_surfs[_iter],
                intersurface_surfaces_slice,
            )
            self.logger.debug(surf_lower_dict)
            self.logger.debug(surf_upper_dict)
            self.logger.debug(surf_inter_dict)
            self.logger.debug("-----------------")

            start_time = time.time()
            surf_loops_slice = self.check_closed_surface_loop(
                surf_lower_dict, surf_upper_dict, surf_inter_dict
            )
            surf_loops.append(surf_loops_slice)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.debug(
                f"Time to check closed surface loops: {elapsed_time:.2f} seconds"
            )

        gmsh.model.occ.synchronize()
        intersurface_volume_tags = []
        for surf_slice in surf_loops:
            for _, _surf_values in surf_slice.items():
                surf_loop_tag = self.add_surface_loop(_surf_values)
                vol_tag = self.gmsh_add_volume([surf_loop_tag])
                intersurface_volume_tags.append(vol_tag)
            volume_tags.append(intersurface_volume_tags)

        # # TODO: move to upper scope when testing is done
        surfs = list(chain(surf_tags, intersurface_surfaces_tags))
        self.factory.synchronize()
        for line_subset in line_tags_intersurf:
            for line in line_subset:
                gmsh.model.mesh.setTransfiniteCurve(
                    line, self.elms_thickness, "Progression", 1.0
                )
        for surf_subset in surfs:
            for surf in surf_subset:
                gmsh.model.mesh.setTransfiniteSurface(surf)
                gmsh.model.mesh.setRecombine(surf, 2)
        for intersurf_vols in volume_tags:
            for vol in intersurf_vols:
                gmsh.model.mesh.setTransfiniteVolume(vol)

        return line_tags_intersurf, surfs, volume_tags


def create_initial_points(coords):
    """
    Add point in the Gmsh model.

    This function adds a point to the Gmsh model based on the provided coordinates.

    Args:
        coords (Tuple[float, float, float]): The coordinates of the point.

    Returns:
        int: The tag of the created point.
    """

    return gmsh.model.occ.addPoint(*coords, tag=-1)
