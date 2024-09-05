import logging
import math
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, List, Tuple, Union

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import shapely.geometry as shpg
from cython_functions import find_closed_curve as fcc
from numpy import float32, float64, int64, ndarray, uint64

# flake8: noqa: E203
LOGGING_NAME = "MESHING"


class Mesher:
    def __init__(
        self,
        geo_file_path: str,
        mesh_file_path: str,
        slicing_coefficient: int,
        n_longitudinal: int,
        n_transverse_cort: int,
        n_transverse_trab: int,
        n_radial: int,
        ellipsoid_fitting: bool = False,
    ):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.option = gmsh.option
        self.plugin = gmsh.plugin
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path
        self.slicing_coefficient = slicing_coefficient
        self.n_longitudinal = n_longitudinal
        self.n_transverse_cort = n_transverse_cort
        self.n_transverse_trab = n_transverse_trab
        self.n_radial = n_radial
        self.ellipsoid_fitting = ellipsoid_fitting
        self.logger = logging.getLogger(LOGGING_NAME)

    def polygon_tensor_of_inertia(
        self, ext_arr: ndarray, int_arr: ndarray, true_coi: bool = True
    ) -> tuple:
        """
        This function calculates the centroid of a polygon, which can be either the true centroid of the cortex
        or the centroid of the internal contour, based on the 'true_coi' parameter.

        If 'true_coi' is True, the function calculates the true centroid of the cortex by subtracting the internal contours
        from the external contours. If the subtraction fails, the function logs an error, saves a plot of the polygons,
        and exits the program.

        If 'true_coi' is False, the function calculates the centroid of the internal contour.

        Args:
            ext_arr (numpy.ndarray): A 2D array representing the external contour of the polygon.

            int_arr (numpy.ndarray): A 2D array representing the internal contour of the polygon.

            true_coi (bool, optional): A flag indicating whether to calculate the true centroid of the cortex.
                                    Defaults to True.

        Returns:
            cortex_centroid (tuple): The coordinates of the calculated centroid.
        """
        extpol = shpg.polygon.Polygon(ext_arr)
        intpol = shpg.polygon.Polygon(int_arr)
        if true_coi:
            try:
                cortex = extpol.difference(intpol)
            except Exception:
                plt.figure(figsize=(5, 5))
                plt.plot(ext_arr[:, 0], ext_arr[:, 1], "r")
                plt.plot(int_arr[:, 0], int_arr[:, 1], "b")
                plt.title("Polygon difference failed")
                # makedir if non-existent
                mesh_dir = Path(self.mesh_file_path).parent
                mesh_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(f"{mesh_dir}/polygon_difference_failed.png")
                self.logger.error("Polygon difference failed")
                sys.exit(99)
            cortex_centroid = (
                cortex.centroid.coords.xy[0][0],
                cortex.centroid.coords.xy[1][0],
            )
        else:
            cortex_centroid = (
                extpol.centroid.coords.xy[0][0],
                extpol.centroid.coords.xy[1][0],
            )
        return cortex_centroid

    def shapely_line_polygon_intersection(
        self, poly: ndarray, line_1: ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Find the intersection point between two lines
        Args:
            line1 (shapely.geometry.LineString): line 1
            line2 (shapely.geometry.LineString): line 2
        Returns:
            shapely.geometry.Point: intersection point
        """

        shapely_poly = shpg.Polygon(poly)
        shapely_line = shpg.LineString(line_1)
        return list(shapely_poly.intersection(shapely_line).coords)

    def partition_lines(
        self, radius: int, centroid: ndarray
    ) -> Tuple[ndarray, ndarray]:
        points = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        points_on_circle = np.array(
            [
                centroid[0] + radius * np.cos(points),
                centroid[1] + radius * np.sin(points),
            ]
        ).transpose()
        line_1 = np.array([points_on_circle[0], points_on_circle[2]])
        line_2 = np.array([points_on_circle[1], points_on_circle[3]])
        return line_1, line_2

    def intersection_point(
        self, arr: ndarray, intersection: ndarray
    ) -> Tuple[ndarray, int64]:
        """
        Insert intersection point between two lines in the array
        Args:
            arr (ndarray): array of points
            intersection_1 (ndarray): intersection point 1
            intersection_2 (ndarray): intersection point 2
        Returns:
            ndarray: new array with intersection points
        """

        dists, closest_idx_2 = spatial.KDTree(arr).query(intersection, k=2)
        if (closest_idx_2 == [0, len(arr) - 1]).all():
            return dists, closest_idx_2[1]
        else:
            self.logger.debug(
                f"closest index where to insert the intersection point: \
            {np.min(closest_idx_2)} (indices: {closest_idx_2})"
            )
            pass
            return dists, np.min(closest_idx_2)

    def shift_point(self, arr, intersection):
        """
        Shift a point in the array to the intersection point.

        This function finds the closest point in the array to the given intersection
        point and replaces it with the intersection point.

        Args:
            arr (ndarray): The array of points.
            intersection (ndarray): The intersection point to be inserted.

        Returns:
            ndarray: The updated array with the intersection point inserted.
        """
        _, closest_idx = spatial.KDTree(arr).query(intersection)
        arr = arr[closest_idx] = intersection
        return arr

    def insert_closest_point(self, arr, closest_idx, values):
        """
        Insert intersection point between two lines in the array
        Args:
            arr (ndarray): array of points
            closest_idx (ndarray): index of the closest point
        Returns:
            ndarray: new array with intersection points
        """

        arr = np.insert(arr, closest_idx, values, axis=0)
        return arr

    def insert_tensor_of_inertia(self, array: ndarray, centroid: ndarray) -> tuple:
        """
        Insert the intersection points of the centroid with the contours.

        This function calculates the intersection points of the centroid with the
        contours, finds the nearest neighbour of the intersection points, and inserts
        the intersection points into the contours.

        Args:
            array (ndarray): The array of contour points.
            centroid (ndarray): The centroid of the cortex.

        Returns:
            tuple: The updated array with intersection points, the list of indices of the intersection points, and the intersection points.
        """
        radius = 150
        intersection_1 = self.shapely_line_polygon_intersection(
            array, self.partition_lines(radius, centroid)[0]
        )
        intersection_2 = self.shapely_line_polygon_intersection(
            array, self.partition_lines(radius, centroid)[1]
        )
        intersections = np.array([intersection_1, intersection_2])

        idx_list = []
        for _, intersection in enumerate(intersections):
            for _, inters in enumerate(intersection):
                _, closest_idx = self.intersection_point(array, inters)
                array[closest_idx] = inters
                idx_list.append(closest_idx)
        return array, idx_list, intersections

    def gmsh_add_points(
        self,
        x: Union[float32, float64],
        y: Union[float32, float64],
        z: Union[float32, float64],
    ) -> int:
        """
        Add a point to the Gmsh model.
        https://gitlab.onelab.info/gmsh/gmsh/-/issues/456
        https://bbanerjee.github.io/ParSim/fem/meshing/gmsh/quadrlateral-meshing-with-gmsh/

        This function adds a point with the given coordinates to the Gmsh model and
        returns the point tag.

        Args:
            x (Union[float32, float64]): The x-coordinate of the point.
            y (Union[float32, float64]): The y-coordinate of the point.
            z (Union[float32, float64]): The z-coordinate of the point.

        Returns:
            int: The tag of the added point.
        """
        gmsh.option.setNumber("General.Terminal", 1)
        point_tag = self.factory.addPoint(x, y, z, tag=-1)
        return point_tag

    def insert_points(self, array: ndarray) -> ndarray:
        """
        Insert points into the Gmsh model.

        This function inserts the points from the given array into the Gmsh model and
        returns an array of point tags.

        Args:
            array (ndarray): The array of points to be inserted.

        Returns:
            ndarray: The array of point tags.
        """
        array_pts_tags = []
        for i, _ in enumerate(array):
            array_tag = self.gmsh_add_points(array[i][0], array[i][1], array[i][2])
            array_pts_tags = np.append(array_pts_tags, array_tag)
        array_pts_tags = np.asarray(array_pts_tags, dtype=int)
        return array_pts_tags

    def insert_lines(self, array: ndarray) -> List[int]:
        """
        Insert lines into the Gmsh model.

        This function inserts lines connecting consecutive points in the given array
        into the Gmsh model and returns a list of line tags.

        Args:
            array (ndarray): The array of points to be connected by lines.

        Returns:
            List[int]: The list of line tags.
        """

        array_lines_tags = []
        for i in range(len(array) - 1):
            array_tag = self.factory.addLine(array[i], array[i + 1], tag=-1)
            array_lines_tags.append(array_tag)
        return array_lines_tags

    def create_centre_splines(
        self,
        i: int,
        point_tags_c: ndarray,
        CENTER_ARC: float,
        signs: List[Tuple[int, int]],
    ) -> List[int]:
        """
        Create center splines in the Gmsh model.

        This function creates center splines based on the given point tags, center arc,
        and signs, and returns a list of spline tags.

        Args:
            i (int): The index of the current center of interest.
            point_tags_c (ndarray): The array of point tags for the center.
            CENTER_ARC (float): The center arc value.
            signs (List[Tuple[int, int]]): The list of sign tuples for the splines.

        Returns:
            List[int]: The list of spline tags.
        """

        coi_r = self.coi_idx[i].reshape(-1, 3)
        coi_r_center = np.mean(coi_r, axis=0)
        vectors = coi_r - coi_r_center
        xx = vectors[0, 0]
        yy = vectors[2, 1]

        midpoints = []
        for sign in signs:
            x = coi_r_center[0] + CENTER_ARC * sign[0] * xx
            y = coi_r_center[1] + CENTER_ARC * sign[1] * yy
            point_arc_tag_i = self.factory.addPoint(x, y, coi_r_center[2])
            midpoints.append(point_arc_tag_i)

        line_tags_s = []
        for j in range(4):
            line_tags_s.append(
                self.factory.addSpline(
                    [point_tags_c[i][j], midpoints[j], point_tags_c[i][(j + 1) % 4]],
                    tag=-1,
                )
            )
        return line_tags_s

    def insert_ellipse_arcs(self, array: np.ndarray, center_tag: int):
        """
        Insert ellipse arcs into the Gmsh model.

        This function inserts ellipse arcs connecting consecutive points in the given
        array into the Gmsh model and returns a list of ellipse arc tags.

        Args:
            array (ndarray): The array of points to be connected by ellipse arcs.
            center_tag (int): The tag of the center point for the ellipse arcs.

        Returns:
            List[int]: The list of ellipse arc tags.
        """
        array_ellipse_tags = []
        for i in range(len(array) - 1):
            try:
                array_tag = self.factory.add_ellipse_arc(
                    array[i], center_tag, array[i], array[i + 1], tag=-1
                )
            except Exception:
                array_tag = self.factory.add_ellipse_arc(
                    array[i], center_tag, array[i + 1], array[i + 1], tag=-1
                )
            array_ellipse_tags.append(array_tag)
        return array_ellipse_tags

    def sort_intersection_points(self, array: ndarray) -> ndarray:
        """
        Sort the intersection points in counterclockwise direction.

        This function sorts the intersection points in the given array in a
        counterclockwise direction and returns the sorted array.

        Args:
            array (ndarray): The array of intersection points to be sorted.

        Returns:
            ndarray: The sorted array of intersection points.
        """
        self.factory.synchronize()
        points = []
        for _, subarray in enumerate(array):
            points_subarr = [self.model.getValue(0, point, []) for point in subarray]
            points.append(points_subarr)
        points_sorted = self.sort_ccw(points[0])
        array_sorted = [subarr[points_sorted] for subarr in array]
        return np.array(array_sorted, dtype=int)

    def insert_intersection_line(
        self, point_tags: ndarray, idx_list: list
    ) -> Tuple[ndarray, ndarray]:
        """
        Insert intersection lines into the Gmsh model.

        This function inserts lines connecting the intersection points in the given
        point tags array and returns the line tags and the sorted intersection points.

        Args:
            point_tags (ndarray): The array of point tags.
            idx_list (list): The list of indices for the intersection points.

        Returns:
            Tuple[ndarray, ndarray]: The array of line tags and the sorted intersection points.
        """
        point_tags = np.array(point_tags).tolist()
        reshaped_point_tags = np.reshape(point_tags, (len(idx_list), -1))
        indexed_points = reshaped_point_tags[
            np.arange(len(idx_list))[:, np.newaxis], idx_list
        ]

        sorted_indexed_points = self.sort_intersection_points(indexed_points)

        line_tags = []
        for j in range(len(sorted_indexed_points[0, :])):
            for i in range(len(sorted_indexed_points[:, j]) - 1):
                line = self.factory.addLine(
                    sorted_indexed_points[i][j], sorted_indexed_points[i + 1][j]
                )
                line_tags = np.append(line_tags, line)
        return line_tags, sorted_indexed_points

    def sort_bspline_cw(self, coords, point_tags):
        """
        Sort B-spline points in clockwise order.

        This function sorts the given B-spline points in a clockwise order based on
        their angles from the centroid and returns the sorted point tags.

        Args:
            coords (ndarray): The coordinates of the B-spline points.
            point_tags (ndarray): The tags of the B-spline points.

        Returns:
            ndarray: The sorted point tags.
        """
        coords = np.asarray(coords)
        center = np.mean(coords, axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_indices = list(sorted_indices)
        point_tags_sorted = [point_tags[i] for i in sorted_indices]
        return point_tags_sorted

    def get_sort_coords(self, point_tags: List[ndarray]) -> ndarray:
        """
        Get and sort coordinates of points.

        This function retrieves the coordinates of the given point tags, slices them
        into sub-arrays, and sorts each sub-array in a counterclockwise order.

        Args:
            point_tags (List[ndarray]): The list of point tags.

        Returns:
            ndarray: The sorted array of point tags.
        """
        self.factory.synchronize()
        start_point = [lst[0] for lst in point_tags]
        coords = [self.model.getValue(0, point, []) for point in start_point]
        # slice it into 4 sub-arrays (division of 4 per each slice)
        coords_sliced = [coords[i : i + 4] for i in range(0, len(coords), 4)]
        point_tags_sliced = [
            point_tags[i : i + 4] for i in range(0, len(point_tags), 4)
        ]

        point_tags_sliced = np.array(point_tags_sliced, dtype=object)
        # sort the sub-arrays in c-clockwise order
        point_tags_sliced_sorted = np.copy(point_tags_sliced)
        for i, coord_slice in enumerate(coords_sliced):
            point_tags_sorted = self.sort_ccw(coord_slice)
            point_tags_sliced_sorted[i] = point_tags_sliced[i][point_tags_sorted]
        return point_tags_sliced_sorted

    def gmsh_insert_bspline(self, points: ndarray) -> int:
        """
        Insert a B-spline into the Gmsh model.

        This function inserts a B-spline with the given points into the Gmsh model and
        returns the B-spline tag.

        Args:
            points (ndarray): The points of the B-spline.

        Returns:
            int: The tag of the added B-spline.
        """
        points = np.array(points).tolist()
        b_spline = self.factory.addBSpline(points, tag=-1)
        return b_spline

    def insert_bspline(
        self,
        array: np.ndarray,
        array_pts_tags: np.ndarray,
        indexed_points_coi: np.ndarray,
    ) -> np.ndarray:
        """
        Insert plane bsplines based on the slices' points cloud
        Returns the list of the bspline tags and creates bspline in gmsh

        Args:
            array (np.ndarray): array containing geometry point cloud
            array_pts_tags (np.ndarray): array containing the tags of the points
            indexed_points_coi (np.ndarray): indices at which the bspline is inserted (start-end points of the bspline)

        Returns:
            array_bspline (np.ndarray): point tags of the bspline
        """
        array_pts_tags_split = np.array_split(
            array_pts_tags, len(np.unique(array[:, 2]))
        )

        idx_list_sorted = np.sort(indexed_points_coi)
        idx_list_sorted = np.insert(idx_list_sorted, 0, idx_list_sorted[:, -1], axis=1)

        array_pts_tags_split = [
            np.append(array_pts_tags_split[i], array_pts_tags_split[i])
            for i in range(len(array_pts_tags_split))
        ]

        array_split_idx_full = []
        array_split_idx = None
        for i, _ in enumerate(array_pts_tags_split):
            for j in range(len(idx_list_sorted[0, :]) - 1):
                # section the array with the idx_list
                if j == 0:
                    idx_min = min(
                        np.where(array_pts_tags_split[i] == idx_list_sorted[i, j])[0]
                    )
                    idx_max = max(
                        np.where(array_pts_tags_split[i] == idx_list_sorted[i, j + 1])[
                            0
                        ]
                    )
                else:
                    idx_min = min(
                        np.where(array_pts_tags_split[i] == idx_list_sorted[i, j])[0]
                    )
                    idx_max = min(
                        np.where(array_pts_tags_split[i] == idx_list_sorted[i, j + 1])[
                            0
                        ]
                    )
                array_split_idx = array_pts_tags_split[i][idx_min : idx_max + 1]
                array_split_idx_full.append(array_split_idx)

        array_split_idx_full_sorted = self.get_sort_coords(array_split_idx_full)

        array_bspline = []
        for array_split_idx_slice in array_split_idx_full_sorted:
            for bspline_indices in array_split_idx_slice:
                array_bspline_s = self.gmsh_insert_bspline(bspline_indices)
                array_bspline = np.append(array_bspline, array_bspline_s)
        return array_bspline, array_split_idx

    def add_bspline_filling(
        self, bsplines: ndarray, intersections: ndarray, slicing_coefficient: int
    ) -> ndarray:
        """
        Add B-spline fillings to the Gmsh model.

        This function adds B-spline fillings between the given B-splines and
        intersections in the Gmsh model and returns an array of B-spline filling tags.

        Args:
            bsplines (ndarray): The array of B-spline tags.
            intersections (ndarray): The array of intersection tags.
            slicing_coefficient (int): The slicing coefficient.

        Returns:
            ndarray: The array of B-spline filling tags.
        """

        bsplines = bsplines.astype(int)

        # reshape and convert to list
        bspline_t = np.array(bsplines, dtype="int").reshape((slicing_coefficient, -1))
        bspline_first_elements = bspline_t[:, 0][:, np.newaxis]
        array_bspline_sliced = np.concatenate(
            (bspline_t, bspline_first_elements), axis=1
        )

        intersections_t = np.array(intersections, dtype=int).reshape(
            (-1, slicing_coefficient - 1)
        )
        intersections_first_elements = intersections_t[:, [0]]
        intersections_s = np.concatenate(
            (intersections_t, intersections_first_elements), axis=1
        )
        intersections_append_s = np.append(
            intersections_s, intersections_s[0, :][np.newaxis, :], axis=0
        )
        intersections_append = (
            np.array(intersections_append_s).reshape((-1, slicing_coefficient)).tolist()
        )

        try:
            BSplineFilling_tags = []
            for i in range(len(array_bspline_sliced) - 1):
                bspline_s = array_bspline_sliced[i]
                for j in range(len(bspline_s) - 1):
                    WIRE = self.factory.addWire(
                        [
                            intersections_append[j][i],
                            array_bspline_sliced[i][j],
                            intersections_append[j + 1][i],
                            array_bspline_sliced[i + 1][j],
                        ],
                        tag=-1,
                        checkClosed=True,
                    )

                    BSplineFilling = self.factory.addBSplineFilling(
                        WIRE, type="Stretch", tag=-1
                    )
                    BSplineFilling_tags = np.append(BSplineFilling_tags, BSplineFilling)
        except Exception as e:
            # I'm truly not proud of this trick
            self.logger.warning(f"Error in BSplineFilling: {e}\nRolling the array")
            bspline_t = np.array(bsplines, dtype="int").reshape(
                (slicing_coefficient, -1)
            )
            bspline_t = np.array([np.roll(row, -1) for row in bspline_t])
            bspline_first_elements = bspline_t[:, 0][:, np.newaxis]
            array_bspline_sliced = np.concatenate(
                (bspline_t, bspline_first_elements), axis=1
            )
            BSplineFilling_tags = []
            for i in range(len(array_bspline_sliced) - 1):
                bspline_s = array_bspline_sliced[i]
                for j in range(len(bspline_s) - 1):
                    WIRE = self.factory.addWire(
                        [
                            intersections_append[j][i],
                            array_bspline_sliced[i][j],
                            intersections_append[j + 1][i],
                            array_bspline_sliced[i + 1][j],
                        ],
                        tag=-1,
                        checkClosed=True,
                    )

                    BSplineFilling = self.factory.addBSplineFilling(
                        WIRE, type="Stretch", tag=-1
                    )
                    BSplineFilling_tags = np.append(BSplineFilling_tags, BSplineFilling)
        return BSplineFilling_tags

    def add_interslice_segments(
        self, point_tags_ext: ndarray, point_tags_int: ndarray
    ) -> ndarray:
        """
        Add interslice segments to the Gmsh model.

        This function adds interslice segments connecting the external and internal
        point tags in the Gmsh model and returns an array of segment tags.

        Args:
            point_tags_ext (ndarray): The array of external point tags.
            point_tags_int (ndarray): The array of internal point tags.

        Returns:
            ndarray: The array of interslice segment tags.
        """

        point_tags_ext = point_tags_ext.flatten().tolist()
        point_tags_int = point_tags_int.flatten().tolist()

        interslice_seg_tag = []
        for i, _ in enumerate(point_tags_ext):
            interslice_seg_tag_s = self.factory.addLine(
                point_tags_ext[i], point_tags_int[i], tag=-1
            )
            interslice_seg_tag = np.append(interslice_seg_tag, interslice_seg_tag_s)

        return interslice_seg_tag

    def add_slice_surfaces(
        self, ext_tags: ndarray, int_tags: ndarray, interslice_seg_tags: ndarray
    ) -> ndarray:
        """
        Add slice surfaces to the Gmsh model.

        This function adds slice surfaces between the external, internal, and interslice
        segment tags in the Gmsh model and returns an array of surface tags.

        Args:
            ext_tags (ndarray): The array of external tags.
            int_tags (ndarray): The array of internal tags.
            interslice_seg_tags (ndarray): The array of interslice segment tags.

        Returns:
            ndarray: The array of slice surface tags.
        """

        ext_r = ext_tags.reshape((self.slicing_coefficient, -1)).astype(int)
        int_r = int_tags.reshape((self.slicing_coefficient, -1)).astype(int)
        inter_r = interslice_seg_tags.reshape((self.slicing_coefficient, -1))
        inter_c = np.concatenate(
            (inter_r[:, 0:], inter_r[:, 0].reshape(self.slicing_coefficient, 1)), axis=1
        )
        inter_a = np.concatenate((inter_c, inter_c[-1].reshape(1, -1)), axis=0).astype(
            int
        )

        try:
            ext_int_tags = []
            self.logger.debug("Surface slices")
            self.logger.debug("j, i")
            for i in range(len(inter_a) - 1):
                interslice_i = inter_a[i]
                for j in range(len(interslice_i) - 1):
                    self.logger.debug(
                        f"{j}, {i}:\t{inter_a[i][j]} {ext_r[i][j]} {inter_a[i][j + 1]} {int_r[i][j]}"
                    )
                    ext_int_tags_s = self.factory.addCurveLoop(
                        [
                            inter_a[i][j],
                            ext_r[i][j],
                            inter_a[i][j + 1],
                            int_r[i][j],
                        ],
                        tag=-1,
                    )
                    ext_int_tags = np.append(ext_int_tags, ext_int_tags_s)
        except Exception as e:
            self.logger.warning(f"Error in add_slice_surfaces: {e}\nRolling the array")
            ext_r = np.array([np.roll(ext_r, -1) for ext_r in ext_r])
            ext_int_tags = []
            for i in range(len(inter_a) - 1):
                interslice_i = inter_a[i]
                for j in range(len(interslice_i) - 1):
                    ext_int_tags_s = self.factory.addCurveLoop(
                        [
                            inter_a[i][j],
                            ext_r[i][j],
                            inter_a[i][j + 1],
                            int_r[i][j],
                        ],
                        tag=-1,
                    )
                    ext_int_tags = np.append(ext_int_tags, ext_int_tags_s)

        ext_int_tags_l = np.array(ext_int_tags, dtype="int").tolist()
        slices_ext_int_tags = []
        for surf_tag in ext_int_tags_l:
            slice_tag = self.factory.addSurfaceFilling(surf_tag, tag=-1)
            slices_ext_int_tags.append(slice_tag)
        return np.array(slices_ext_int_tags)

    def add_intersurface_planes(
        self,
        intersurface_line_tags: ndarray,
        intersection_line_tag_ext: ndarray,
        intersection_line_tag_int: ndarray,
    ) -> List[int]:
        """
        Add intersurface planes to the Gmsh model.

        This function adds intersurface planes between the given line tags and
        intersection line tags in the Gmsh model and returns a list of surface tags.

        Args:
            intersurface_line_tags (ndarray): The array of intersurface line tags.
            intersection_line_tag_ext (ndarray): The array of external intersection line tags.
            intersection_line_tag_int (ndarray): The array of internal intersection line tags.

        Returns:
            List[int]: The list of intersurface plane tags.
        """
        # 4 lines per surface
        intersurface_line_tags_r = np.array(intersurface_line_tags, dtype=int).reshape(
            (-1, 4)
        )
        intersection_line_tag_ext_r = (
            np.array(intersection_line_tag_ext, dtype=int).reshape((4, -1)).T
        )
        intersection_line_tag_int_r = (
            np.array(intersection_line_tag_int, dtype=int).reshape((4, -1)).T
        )

        intersurface_curveloop_tag = []
        self.logger.debug("Intersurface planes:")
        self.logger.debug("j, i")
        for i in range(len(intersurface_line_tags_r) - 1):
            intersurface_line_tags_r_i = intersurface_line_tags_r[i]
            for j in range(len(intersurface_line_tags_r_i)):
                self.logger.debug(
                    f"{j}, {i}:\t{intersurface_line_tags_r[i][j]} {intersection_line_tag_int_r[i][j]} {intersurface_line_tags_r[i + 1][j]} {intersection_line_tag_ext_r[i][j]}"
                )
                intersurf_tag = self.factory.addCurveLoop(
                    [
                        intersurface_line_tags_r[i][j],
                        intersection_line_tag_int_r[i][j],
                        intersurface_line_tags_r[i + 1][j],
                        intersection_line_tag_ext_r[i][j],
                    ],
                    tag=-1,
                )
                intersurface_curveloop_tag.append(intersurf_tag)

        intersurface_surface_tags = []
        for surf_tag in intersurface_curveloop_tag:
            intersurf_tag = self.factory.addSurfaceFilling(surf_tag, tag=-1)
            intersurface_surface_tags.append(intersurf_tag)
        return intersurface_surface_tags

    def gmsh_geometry_formulation(
        self, array: np.ndarray, idx_list: np.ndarray
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Formulate the geometry in the Gmsh model.

        This function formulates the geometry in the Gmsh model by inserting points,
        intersection lines, B-splines, and B-spline fillings, and returns the tags of
        the indexed points, B-splines, intersection lines, and B-spline fillings.

        Args:
            array (ndarray): The array of points.
            idx_list (ndarray): The list of indices for the intersection points.

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]: The tags of the indexed points, B-splines, intersection lines, and B-spline fillings.
        """

        array_pts_tags = self.insert_points(array)

        intersection_line_tag, indexed_points_coi = self.insert_intersection_line(
            array_pts_tags, idx_list
        )

        array_bspline, array_split_idx = self.insert_bspline(
            array, array_pts_tags, indexed_points_coi
        )

        # create curveloop between array_bspline and intersection_line_tag
        bspline_filling_tags = self.add_bspline_filling(
            array_bspline, intersection_line_tag, self.slicing_coefficient
        )
        return (
            indexed_points_coi,
            array_bspline,
            intersection_line_tag,
            bspline_filling_tags,
        )

    def meshing_transfinite(
        self,
        longitudinal_line_tags: Union[List[int], ndarray],
        transverse_line_tags: List[int],
        radial_line_tags: Union[List[int], ndarray],
        surface_tags: Union[List[int], ndarray],
        volume_tags: Union[List[int], ndarray],
        phase: str = "cort",
    ):
        """
        Apply transfinite meshing to the specified geometric entities.

        This function sets transfinite meshing properties for curves, surfaces, and volumes
        based on the provided tags and the specified phase.

        Args:
            longitudinal_line_tags (Union[List[int], ndarray]): Tags of longitudinal lines.
            transverse_line_tags (List[int]): Tags of transverse lines.
            radial_line_tags (Union[List[int], ndarray]): Tags of radial lines.
            surface_tags (Union[List[int], ndarray]): Tags of surfaces.
            volume_tags (Union[List[int], ndarray]): Tags of volumes.
            phase (str, optional): The phase of the meshing process ("cort" or "trab"). Default is "cort".

        Returns:
            None
        """
        self.factory.synchronize()

        longitudinal_line_tags = list(map(int, longitudinal_line_tags))
        radial_line_tags = list(map(int, radial_line_tags))
        surface_tags = list(map(int, surface_tags))
        volume_tags = list(map(int, volume_tags))

        n_transverse = None
        if phase == "cort":
            n_transverse = self.n_transverse_cort
            PROGRESSION_FACTOR = 1.0
        elif phase == "trab":
            n_transverse = self.n_transverse_trab
            PROGRESSION_FACTOR = 1.05

        for ll in longitudinal_line_tags:
            self.model.mesh.setTransfiniteCurve(ll, self.n_longitudinal)

        for ll in transverse_line_tags:
            self.model.mesh.setTransfiniteCurve(
                ll, n_transverse, "Progression", PROGRESSION_FACTOR
            )

        for intersection in radial_line_tags:
            self.model.mesh.setTransfiniteCurve(intersection, self.n_radial)

        for surface in surface_tags:
            self.model.mesh.setTransfiniteSurface(surface)

        for volume in volume_tags:
            self.model.mesh.setTransfiniteVolume(volume)

    def add_volume(
        self,
        cortical_ext_surfs: ndarray,
        cortical_int_surfs: ndarray,
        slices_tags: ndarray,
        intersurface_surface_tags: List[int],
    ) -> List[int]:
        """
        Add volumes to the mesh based on the provided surface tags.

        This function creates volumes by defining surface loops from the provided
        cortical and intersurface surface tags.

        Args:
            cortical_ext_surfs (ndarray): Tags of external cortical surfaces.
            cortical_int_surfs (ndarray): Tags of internal cortical surfaces.
            slices_tags (ndarray): Tags of slice surfaces.
            intersurface_surface_tags (List[int]): Tags of intersurface surfaces.

        Returns:
            List[int]: Tags of the created volumes.
        """
        cortical_ext_surfs = np.array(cortical_ext_surfs, dtype=int).reshape((-1, 4))
        cortical_int_surfs = np.array(cortical_int_surfs, dtype=int).reshape((-1, 4))
        slices_tags_r = np.array(slices_tags, dtype=int).reshape((-1, 4))

        intersurface_surface_tags_r = np.array(intersurface_surface_tags).reshape(
            (-1, 4)
        )
        intersurface_surface_tags_s = np.append(
            intersurface_surface_tags_r,
            intersurface_surface_tags_r[:, 0][:, np.newaxis],
            axis=1,
        )

        surface_loop_tag = []
        for i in range(len(slices_tags_r) - 1):
            slices_tag_s = slices_tags_r[i]
            for j, _ in enumerate(slices_tag_s):
                surface_loop_t = self.factory.addSurfaceLoop(
                    [
                        intersurface_surface_tags_s[i][j],
                        slices_tags_r[i][j],
                        cortical_ext_surfs[i][j],
                        slices_tags_r[i + 1][j],
                        cortical_int_surfs[i][j],
                        intersurface_surface_tags_s[i][j + 1],
                    ],
                    tag=-1,
                )
                surface_loop_tag.append(surface_loop_t)

        volume_tag = []
        for i in range(len(surface_loop_tag)):
            volume_t = self.factory.addVolume([surface_loop_tag[i]], tag=-1)
            volume_tag.append(volume_t)

        return volume_tag

    def sort_by_indexes(self, lst, indexes, reverse=False):
        """
        Sort a list based on the provided indexes.

        This function sorts the elements of a list according to the specified indexes.

        Args:
            lst (List): The list to be sorted.
            indexes (List[int]): The indexes to sort the list by.
            reverse (bool, optional): Whether to sort in reverse order. Default is False.

        Returns:
            List: The sorted list.
        """
        return [
            val
            for (_, val) in sorted(
                zip(indexes, lst), key=lambda x: x[0], reverse=reverse
            )
        ]

    def query_closest_idx(self, list_1: list, list_2: list):
        """
        This function takes two lists of indices, retrieves the corresponding coordinates from the model,
        and sorts the second list in a counter-clockwise order based on the coordinates.

        Args:
            list_1 (list): A list of indices. Each index corresponds to a coordinate in the model.
                        This list represents the 'center of inertia' coordinates.

            list_2 (list): A list of indices. Each index corresponds to a coordinate in the model.
                        This list represents the 'trabecular' coordinates.

        Returns:
            list_2_sorted (list): The list of 'trabecular' indices sorted in a counter-clockwise order
                                based on the corresponding coordinates.
        """
        coi_coords = []
        trab_coords = []
        for i, _ in enumerate(list_1):
            coi_coords_s = [self.model.getValue(0, idx, []) for idx in list_1[i]]
            trab_coords_s = [self.model.getValue(0, idx, []) for idx in list_2[i]]
            coi_coords.append(coi_coords_s)
            trab_coords.append(trab_coords_s)

        trab_idx_closest_ccw = self.sort_ccw(trab_coords[0])
        list_2_np = np.array(list_2)
        list_2_sorted = [subarr[trab_idx_closest_ccw] for subarr in list_2_np]
        return list_2_sorted

    def trabecular_cortical_connection(
        self, coi_idx: ndarray, trab_point_tags: ndarray
    ) -> List[int]:
        """
        This function creates a connection (line) between each pair of points from two given lists in the model.
        The connections are created between each 'center of interest' point and the corresponding 'trabecular' point.

        Args:
            coi_idx (list[int]): A list of indices. Each index corresponds to a 'center of inertia' point in the model.

            trab_point_tags (list[int]): A list of indices. Each index corresponds to a 'trabecular' point in the model.
                                        The order of indices matches with the 'center of inerta' points in coi_idx.

        Returns:
            trab_cort_line_tags (list): A list of line tags. Each line tag represents a connection (line) created in the model.
        """
        coi_idx = np.array(coi_idx, dtype=int).tolist()
        trab_point_tags = np.array(trab_point_tags, dtype=int).tolist()

        trab_idx_closest = trab_point_tags

        trab_cort_line_tags = []
        for i, _ in enumerate(coi_idx):
            for j, _ in enumerate(coi_idx[i]):
                line_tag_s = self.factory.addLine(
                    coi_idx[i][j], trab_idx_closest[i][j], tag=-1
                )
                self.logger.debug(
                    f"AddLine {line_tag_s}: {coi_idx[i][j]} {trab_idx_closest[i][j]}"
                )
                trab_cort_line_tags.append(line_tag_s)
        return trab_cort_line_tags

    def trabecular_planes_inertia(
        self,
        trab_cort_line_tags: List[int],
        trab_line_tags_v: List[int],
        intersection_line_tags_int: ndarray,
    ) -> List[int]:
        """
        Create trabecular planes of inertia.

        This function generates trabecular planes of inertia by defining curve loops
        and filling surfaces based on the provided line tags.

        Args:
            trab_cort_line_tags (List[int]): Tags of cortical trabecular lines.
            trab_line_tags_v (List[int]): Tags of vertical trabecular lines.
            intersection_line_tags_int (ndarray): Tags of intersection lines.

        Returns:
            List[int]: Tags of the created trabecular planes of inertia.
        """
        trab_cort_line_tags_np = np.array(trab_cort_line_tags).reshape(
            (self.slicing_coefficient, -1)
        )
        trab_line_tags_v_np = np.array(trab_line_tags_v, dtype=int).reshape(
            (-1, self.slicing_coefficient - 1)
        )

        intersection_line_tags_int_np = np.array(
            intersection_line_tags_int, dtype=int
        ).reshape((-1, self.slicing_coefficient - 1))

        self.logger.debug("Trabecular planes of inertia")
        self.logger.debug("j, i")
        trab_plane_inertia_tags = []
        for j in range(0, len(trab_line_tags_v_np[:, 0])):
            for i in range(1, len(trab_cort_line_tags_np[:, 0])):
                self.logger.debug(
                    f"{j}, {i}:\t{trab_cort_line_tags_np[i-1][j]} {intersection_line_tags_int_np[j][i-1]} {trab_cort_line_tags_np[i][j]} {trab_line_tags_v_np[j][i-1]}"
                )
                curve_loop = self.factory.addCurveLoop(
                    [
                        trab_cort_line_tags_np[i - 1][j],
                        intersection_line_tags_int_np[j][i - 1],
                        trab_cort_line_tags_np[i][j],
                        trab_line_tags_v_np[j][i - 1],
                    ],
                    tag=-1,
                )
                trab_plane_s = self.factory.addSurfaceFilling(curve_loop, tag=-1)
                trab_plane_inertia_tags.append(trab_plane_s)

        return trab_plane_inertia_tags

    def trabecular_slices(
        self,
        trab_cort_line_tags: List[int],
        trab_line_tags_h: List[int],
        cort_int_bspline_tags: ndarray,
    ) -> List[int]:
        """
        Create trabecular slices.

        This function generates trabecular slices by defining curve loops and plane surfaces
        based on the provided line and B-spline tags.

        Args:
            trab_cort_line_tags (List[int]): Tags of cortical trabecular lines.
            trab_line_tags_h (List[int]): Tags of horizontal trabecular lines.
            cort_int_bspline_tags (ndarray): Tags of internal cortical B-splines.

        Returns:
            List[int]: Tags of the created trabecular slices.
        """
        trab_line_tags_h_np = np.array(trab_line_tags_h, dtype=int).reshape((-1, 4))

        trab_cort_line_tags_np_s = np.array(trab_cort_line_tags, dtype=int).reshape(
            (-1, 4)
        )
        trab_cort_line_tags_np = np.append(
            trab_cort_line_tags_np_s,
            trab_cort_line_tags_np_s[:, 0][:, np.newaxis],
            axis=1,
        )

        cort_int_bspline_tags_np = np.array(cort_int_bspline_tags, dtype=int).reshape(
            (-1, 4)
        )

        self.logger.debug("Trabecular slices")
        self.logger.debug("j, i")
        trabecular_slice_surf_tags = []
        for i in range(len(trab_cort_line_tags_np)):
            for j in range(1, len(trab_cort_line_tags_np[i])):
                self.logger.debug(
                    f"{j}, {i}:\t{trab_cort_line_tags_np[i][j-1]} {trab_line_tags_h_np[i][j-1]} {trab_cort_line_tags_np[i][j]} {cort_int_bspline_tags_np[i][j-1]}"
                )

                curve_loop = self.factory.addCurveLoop(
                    [
                        trab_cort_line_tags_np[i][j - 1],
                        trab_line_tags_h_np[i][j - 1],
                        trab_cort_line_tags_np[i][j],
                        cort_int_bspline_tags_np[i][j - 1],
                    ],
                    tag=-1,
                )

                trab_slice = self.factory.addPlaneSurface([curve_loop], tag=-1)
                trabecular_slice_surf_tags.append(trab_slice)
        return trabecular_slice_surf_tags

    def get_trabecular_cortical_volume_mesh(
        self,
        trab_slice_surf_tags: List[int],
        trab_plane_inertia_tags: List[int],
        cortical_int_surfs: ndarray,
        trab_surfs_v: List[int],
    ) -> List[int]:
        """
        Generate the trabecular-cortical volume mesh.

        This function creates the trabecular-cortical volume mesh by defining surface loops
        and volumes based on the provided surface and plane tags.

        Args:
            trab_slice_surf_tags (List[int]): Tags of trabecular slice surfaces.
            trab_plane_inertia_tags (List[int]): Tags of trabecular plane inertia surfaces.
            cortical_int_surfs (ndarray): Tags of internal cortical surfaces.
            trab_surfs_v (List[int]): Tags of vertical trabecular surfaces.

        Returns:
            List[int]: Tags of the created trabecular-cortical volumes.
        """
        trab_slice_surf_tags_np = np.array(trab_slice_surf_tags, dtype=int).reshape(
            (-1, 4)  # * mind the 4 (4 surfaces per trabecular slice)
        )

        trab_surfs_v_np = (
            np.array(trab_surfs_v, dtype=int)
            .reshape(-1, self.slicing_coefficient - 1)
            .T
        )

        trab_plane_inertia_tags_np = (
            np.array(trab_plane_inertia_tags, dtype=int)
            .reshape((-1, self.slicing_coefficient - 1))
            .T
        )
        trab_plane_inertia = np.concatenate(
            (trab_plane_inertia_tags_np, trab_plane_inertia_tags_np[:, 0][:, None]),
            axis=1,
        )

        cortical_int_surfs_np = np.array(cortical_int_surfs, dtype=int).reshape(
            (self.slicing_coefficient - 1, -1)
        )
        cortical_int_surfs_np_2 = np.concatenate(
            (cortical_int_surfs_np, cortical_int_surfs_np[:, 0][:, None]), axis=1
        )

        cort_trab_vol_tags = []
        self.logger.debug("Cortical-Trabecular volume mesh")
        self.logger.debug("j, i")
        for j in range(len(trab_plane_inertia[:, 0])):
            for i in range(1, len(trab_plane_inertia[0, :])):
                self.logger.debug(
                    f"{j}, {i}:\t{trab_plane_inertia[j][i-1]} {trab_slice_surf_tags_np[j][i-1]} "
                    f"{trab_plane_inertia[j][i]} {cortical_int_surfs_np_2[j][i-1]} "
                    f"{trab_slice_surf_tags_np[j+1][i-1]} {trab_surfs_v_np[j][i-1]}"
                )
                surf_loop = self.factory.addSurfaceLoop(
                    [
                        trab_plane_inertia[j][i - 1],
                        trab_slice_surf_tags_np[j][i - 1],
                        trab_plane_inertia[j][i],
                        cortical_int_surfs_np_2[j][i - 1],
                        trab_slice_surf_tags_np[j + 1][i - 1],
                        trab_surfs_v_np[j][i - 1],
                    ],
                    tag=-1,
                )
                vol_s = self.factory.addVolume([surf_loop], tag=-1)
                cort_trab_vol_tags.append(vol_s)

        return cort_trab_vol_tags

    def sort_ccw(self, coords: np.ndarray) -> np.ndarray:
        """
        Sort coordinates in counterclockwise order.

        This function sorts the given coordinates in counterclockwise order based on their
        angles relative to the centroid.

        Args:
            coords (ndarray): The coordinates to be sorted.

        Returns:
            ndarray: The sorted indices of the coordinates.
        """
        x_coords, y_coords, _ = zip(*coords)
        centroid = (sum(x_coords) / len(coords), sum(y_coords) / len(coords))

        # Calculate angle of each coordinate relative to negative x-axis and negative y-axis
        axis_s = (-1, -1)
        axis = (axis_s[0] - centroid[0], axis_s[1] - centroid[1])
        angles = []
        for coord in coords:
            vec = (coord[0] - centroid[0], coord[1] - centroid[1])
            angle = math.atan2(vec[1], vec[0]) - math.atan2(axis[1], axis[0])
            if angle < 0:
                angle += 2 * math.pi
            angles.append(angle)

        # Sort coordinates by angle and distance to centroid
        sorted_angles = np.argsort(angles)
        angles_idx = np.arange(len(angles))
        sorted_indices = angles_idx[sorted_angles]
        return sorted_indices

    def mesh_generate(self, dim: int, element_order: int, vol_tags: list):
        """
        Generate the mesh for the specified dimension and element order.

        This function sets various meshing options and generates the mesh for the specified
        dimension and element order.

        Args:
            dim (int): The dimension of the mesh (1, 2, or 3).
            element_order (int): The order of the elements in the mesh.
            vol_tags (list): Tags of the volumes to be meshed.

        Returns:
            None
        """
        self.option.setNumber("Mesh.RecombineAll", 1)
        self.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        self.option.setNumber("Mesh.Recombine3DLevel", 1)
        self.option.setNumber("Mesh.ElementOrder", element_order)
        self.option.setNumber("Mesh.Smoothing", 1000000)

        for dim in (1, 2, 3):
            for s in self.model.getEntities(dim):
                self.model.mesh.setRecombine(s[0], s[1])
                self.model.mesh.setSmoothing(s[0], s[1], 1000000)

            for s in self.model.getEntities(dim):
                self.model.mesh.setRecombine(s[0], s[1])
                self.model.mesh.setSmoothing(s[0], s[1], 1000000)

            for s in self.model.getEntities(dim):
                self.model.mesh.setRecombine(s[0], s[1])
                self.model.mesh.setSmoothing(s[0], s[1], 1000000)

        self.model.mesh.generate(dim)

    def analyse_mesh_quality(self, hiding_thresh: float) -> None:
        """
        Analyze the quality of the generated mesh.

        This function runs the "AnalyseMeshQuality" plugin to evaluate the quality of the
        generated mesh based on the specified hiding threshold.

        Args:
            hiding_thresh (float): The threshold for hiding elements based on quality.

        Returns:
            None
        """
        self.plugin.setNumber("AnalyseMeshQuality", "JacobianDeterminant", 1)
        self.plugin.setNumber("AnalyseMeshQuality", "CreateView", 1)
        self.plugin.setNumber("AnalyseMeshQuality", "DimensionOfElements", -1)
        self.plugin.setNumber("AnalyseMeshQuality", "HidingThreshold", hiding_thresh)
        self.plugin.run("AnalyseMeshQuality")
        return None

    def gmsh_get_nodes(self) -> Dict[uint64, ndarray]:
        """
        Get the nodes of the mesh.

        This function retrieves the nodes of the mesh for each physical group and returns
        them as a dictionary.

        Returns:
            Dict[uint64, ndarray]: A dictionary with node tags as keys and coordinates as values.
        """
        nTagsCoord_dict = {}
        for physical_group in self.model.getPhysicalGroups():
            nTags_s, nCoord_s = self.model.mesh.getNodesForPhysicalGroup(
                3, physical_group[1]
            )
            nTags_np = np.array(nTags_s)
            nCoord_np = np.array(nCoord_s).reshape(-1, 3)
            # create dictionary with key = nTags_np and value = nCoord_np
            nTagsCoord_dict_s = dict(zip(nTags_np, nCoord_np))
            nTagsCoord_dict.update(nTagsCoord_dict_s)
        return nTagsCoord_dict

    def gmsh_get_elms(self) -> Dict[uint64, ndarray]:
        """
        Get the elements of the mesh.

        This function retrieves the elements of the mesh for each volume and returns them
        as a dictionary.

        Returns:
            Dict[uint64, ndarray]: A dictionary with element tags as keys and node tags as values.
        """
        elms_dict = {}
        nodeTags = None
        for vol in self.model.getEntities(3):
            elmTypes_s, elmTags_s, nodeTags_s = self.model.mesh.getElements(3, vol[1])

            if elmTypes_s[0] == 5:
                # 8-node hexahedron
                nodeTags = np.array(nodeTags_s[0]).reshape(-1, 8)
            elif elmTypes_s[0] == 12:
                # 27-node second order hexahedron (8 nodes associated with the vertices, 12 with the edges, 6 with the faces and 1 with the volume)
                nodeTags = np.array(nodeTags_s[0]).reshape(-1, 27)

            elms_s = dict(zip(elmTags_s[0], nodeTags))
            elms_dict.update(elms_s)
        return elms_dict

    def gmsh_get_bnds(
        self, nodes: dict
    ) -> Tuple[Dict[uint64, ndarray], Dict[uint64, ndarray]]:
        """
        Get the boundary nodes of the mesh.

        This function identifies the boundary nodes of the mesh based on their z-coordinate
        and returns them as two dictionaries for the bottom and top boundaries.

        Args:
            nodes (dict): The node set of the mesh.

        Returns:
            Tuple[Dict[uint64, ndarray], Dict[uint64, ndarray]]: Dictionaries of bottom and top boundary nodes.
        """
        z_min = float("inf")
        z_max = float("-inf")
        tol = 0.01

        for arr in nodes.values():
            last_element = arr[-1]
            z_min = min(z_min, last_element)
            z_max = max(z_max, last_element)

        # mask the dict to get only the key value pairs with z_min and z_max
        # here z_max and z_min are inverted because of the nature of our csys
        # (the most distal part is actually at z=0 and the most proximal at z>0)
        bnds_bot = {
            k: v for k, v in nodes.items() if math.isclose(v[2], z_max, rel_tol=tol)
        }
        bnds_top = {
            k: v for k, v in nodes.items() if math.isclose(v[2], z_min, rel_tol=tol)
        }
        return bnds_bot, bnds_top

    def gmsh_get_reference_point_coord(self, nodes: dict) -> ndarray:
        """
        Get the coordinates of the reference point.

        This function calculates the coordinates of the reference point based on the center
        of mass of the nodes and the maximum z-coordinate.

        Args:
            nodes (dict): The node set of the mesh.

        Returns:
            ndarray: The coordinates of the reference point.
        """
        OFFSET_MM = 0  # RP offset in mm from top surface
        # get the center of mass of the nodes dictionary values
        center_of_mass = np.mean(list(nodes.values()), axis=0)
        max_z = np.max(np.array(list(nodes.values()))[:, 2])

        reference_point_coords = np.array(
            [
                center_of_mass[0],
                center_of_mass[1],
                abs(max_z) + OFFSET_MM,
            ]
        )
        return reference_point_coords

    def get_centroids_dict(self, centroids: ndarray) -> Dict[int, ndarray]:
        """
        Returns a dictionary of centroids with keys as indices and values as centroids.

        Args:
            centroids (np.array): array of centroids in (x, y, z) format (shape: (n_elms, 3))

        Returns:
            centroids_dict (dict): dictionary of centroids with keys as indices and values as centroids
        """
        centroids_dict = {}
        for i, centroid in enumerate(centroids):
            centroids_dict[i + 1] = centroid
        return centroids_dict

    def split_dict_by_array_len(
        self, input_dict: Dict[int, ndarray], len1: int
    ) -> Tuple[Dict[int, ndarray], Dict[int, ndarray]]:
        """
        Split a dictionary into two based on the length of the arrays.

        This function splits the input dictionary into two dictionaries based on the specified
        length of the arrays.

        Args:
            input_dict (Dict[int, ndarray]): The input dictionary to be split.
            len1 (int): The length of the arrays for the first dictionary.

        Returns:
            Tuple[Dict[int, ndarray], Dict[int, ndarray]]: The two resulting dictionaries.
        """
        dict1 = {}
        dict2 = {}
        count1 = 0
        for key, value in input_dict.items():
            if count1 < len1:
                dict1[key] = value
                count1 += 1
            else:
                dict2[key] = value
        return dict1, dict2

    def nodes2coords(self, nodes, elements):
        """
        Associate node coordinates to each element.
        https://github.com/dmelgarm/gmsh/blob/master/gmsh_tools.py#L254

        This function associates the coordinates of the nodes to each element based on the
        provided node and element arrays.

        Args:
            nodes (ndarray): The array of node coordinates.
            elements (ndarray): The array of element definitions.

        Returns:
            ndarray: The array of element coordinates.
        """

        ncoords = np.zeros((len(elements), 10))
        for k in range(len(elements)):
            node1 = int(elements[k, 5])
            node2 = int(elements[k, 6])
            node3 = int(elements[k, 7])
            ncoords[k, 0] = k + 1
            # Get node 1 coordinates
            i = np.where(nodes[:, 0] == node1)[0]
            ncoords[k, 1:4] = nodes[i, 1:4]
            # Get node 2 coordinates
            i = np.where(nodes[:, 0] == node2)[0]
            ncoords[k, 4:7] = nodes[i, 1:4]
            # Get node 1 coordinates
            i = np.where(nodes[:, 0] == node3)[0]
            ncoords[k, 7:10] = nodes[i, 1:4]
        return ncoords

    def get_barycenters(self, tag_s: list[int]) -> ndarray:
        """
        Get barycenters of each volumetric element.

        This function retrieves the barycenters of each volumetric element of type 5 (8-node hexahedron)
        or type 12 (27-node second order hexahedron) for the specified tags.

        Args:
            tag_s (list[int]): List of tags for the volumetric elements.

        Returns:
            ndarray: Array of barycenters in (x, y, z) format with shape (n_elms, 3).
        """

        # * add compartment discrimination for cort/trab physical groups

        fast_s = False
        primary_s = False

        barycenters_t = []
        # start by assuming that elementType=5 (8-node hexahedron)
        for i in range(min(tag_s), max(tag_s) + 1):
            b = gmsh.model.mesh.getBarycenters(
                elementType=5, tag=i, fast=fast_s, primary=primary_s
            )
            barycenters_t.append(b)

        if all(np.size(b) == 0 for b in barycenters_t):
            # assume there's no mixed linear/quadratic elements --> reinitialize barycenters_t
            barycenters_t = []
            # if elementType=5 is empty, then assume it is a second order mesh (elementType=12, 27-node hexahedron)
            for i in range(min(tag_s), max(tag_s) + 1):
                b = gmsh.model.mesh.getBarycenters(
                    elementType=12, tag=i, fast=fast_s, primary=primary_s
                )
                barycenters_t.append(b)

        barycenters_t = np.concatenate(barycenters_t, axis=0)
        barycenters_xyz = barycenters_t.reshape(-1, 3)
        return barycenters_xyz

    def get_elm_volume(self, tag_s: list[int]) -> ndarray:
        """
        Returns an array of containing the volume of each element for the given tags.

        Args:
            tag_s (list[int]): list of element tags to get volumes for

        Returns:
            volumes (np.ndarray): array of element volumes (shape: (n_elms, 1))
        """
        volumes = []
        for _, elementTags, _ in (
            self.model.mesh.getElements(dim=3, tag=tag) for tag in tag_s
        ):
            for i in elementTags:
                v = self.model.mesh.getElementQualities(
                    elementTags=i, qualityName="volume"
                )
                volumes.append(v)

        volumes = np.concatenate(volumes).reshape(-1, 1)

        return volumes

    def get_radius_longest_edge(self, tag_s: ndarray) -> float64:
        """
        Compute the radius of the longest edge for each element in the given tag_s.

        Args:
            tag_s: list([int]) List of element tags to compute the radius for.

        Returns:
            _:     (float) The radius of the longest edge among all elements in tag_s.
        """
        radius = np.array([])
        for _, elementTags, _ in (
            self.model.mesh.getElements(dim=3, tag=tag) for tag in tag_s
        ):
            for i in elementTags:
                edge_min = self.model.mesh.getElementQualities(
                    elementTags=i, qualityName="minEdge"
                )
                edge_max = self.model.mesh.getElementQualities(
                    elementTags=i, qualityName="maxEdge"
                )

                r = np.sqrt(edge_min**2 + edge_max**2) / 2
                radius = np.append(radius, r)
        return np.max(radius)


class TrabecularVolume(Mesher):
    def __init__(
        self,
        geo_file_path: str,
        mesh_file_path: str,
        slicing_coefficient: int,
        n_longitudinal: int,
        n_transverse_cort: int,
        n_transverse_trab: int,
        n_radial: int,
        QUAD_REFINEMENT: bool,
        ellipsoid_fitting: bool,
    ):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path
        self.slicing_coefficient = slicing_coefficient
        self.n_transverse_cort = n_transverse_cort
        self.n_transverse_trab = n_transverse_trab
        self.n_radial = n_radial
        self.ellipsoid_fitting = ellipsoid_fitting
        self.logger = logging.getLogger(LOGGING_NAME)
        self.coi_idx = []
        self.line_tags_v = []
        self.line_tags_h = []
        self.surf_tags = []
        self.vol_tags = []
        self.LENGTH_FACTOR = float(1.0)
        self.QUAD_REFINEMENT = QUAD_REFINEMENT
        super().__init__(
            geo_file_path,
            mesh_file_path,
            slicing_coefficient,
            n_longitudinal,
            n_transverse_cort,
            n_transverse_trab,
            n_radial,
            ellipsoid_fitting,
        )

    def principal_axes_length(self, array: ndarray) -> Tuple[float64, float64]:
        """
        Calculate the lengths of the axes of inertia.

        This function calculates the lengths of the principal axes for the given array of points.

        Args:
            array (ndarray): Array of points.

        Returns:
            Tuple[float64, float64]: Lengths of the principal axes.
        """
        l_i = np.linalg.norm(array[0] - array[1])
        l_j = np.linalg.norm(array[0] - array[2])
        return l_i, l_j

    def get_offset_points(
        self, array: ndarray, _center: List[float64], l_i: float64, l_j: float64
    ) -> ndarray:
        """
        Calculate offset points from the center.

        This function calculates the offset points from the center based on the principal axes lengths
        and the length factor.

        Args:
            array (ndarray): Array of points.
            _center (List[float64]): Center point.
            l_i (float64): Length of the first principal axis.
            l_j (float64): Length of the second principal axis.

        Returns:
            ndarray: Array with the offset points.
        """
        # calculate offset point from _center and float(LENGTH_FACTOR) * half of the principal axes length
        offset_i = [self.LENGTH_FACTOR * l_i / 2, 0, 0]
        offset_j = [0, self.LENGTH_FACTOR * l_j / 2, 0]

        # replace the offset point in the array
        array[0] = np.array(_center) + np.array(offset_i)
        array[1] = np.array(_center) - np.array(offset_i)
        array[2] = np.array(_center) + np.array(offset_j)
        array[3] = np.array(_center) - np.array(offset_j)
        return array

    def get_trabecular_position(self) -> ndarray:
        """
        Get the trabecular positions.

        This function calculates the trabecular positions by splitting the input array into subarrays,
        calculating the principal axes lengths, and determining the offset points.

        Returns:
            ndarray: Array of trabecular positions.
        """
        coi_idx_r = np.reshape(self.coi_idx, (-1, 3))
        # create subarrays of the coi_idx array for each slice (coi_idx[:, 2])
        coi_idx_every_4_points = np.split(
            coi_idx_r, np.where(np.diff(coi_idx_r[:, 2]))[0] + 1
        )

        # iterate over the subarrays and calculate the principal axes length
        trabecular_points = np.empty((len(coi_idx_every_4_points), 4, 3))
        for i, _ in enumerate(coi_idx_every_4_points):
            c_x = coi_idx_every_4_points[i][:, 0]
            c_y = coi_idx_every_4_points[i][:, 1]
            c_z = coi_idx_every_4_points[i][:, 2]
            _center = [np.mean(c_x), np.mean(c_y), np.mean(c_z)]

            # calculate the principal axes length
            l_i, l_j = self.principal_axes_length(coi_idx_every_4_points[i])
            trabecular_points[i] = self.get_offset_points(
                coi_idx_every_4_points[i], _center, l_i, l_j
            )
        return np.array(trabecular_points, dtype=np.float32).reshape((-1, 3))

    def sort_trab_coords(self, point_tags: List[int]) -> List[ndarray]:
        """
        Sort trabecular coordinates in counterclockwise order.

        This function sorts the trabecular coordinates in counterclockwise order based on the provided
        point tags.

        Args:
            point_tags (List[int]): List of point tags.

        Returns:
            List[ndarray]: List of sorted point tags.
        """
        self.factory.synchronize()
        start_point = [lst for lst in point_tags]
        coords = [self.model.getValue(0, point, []) for point in start_point]

        # split every 4 points
        point_tags = np.split(np.array(point_tags), self.slicing_coefficient)
        # split every x, y, z coordinate
        coords_split = np.split(np.array(coords), self.slicing_coefficient)

        # sort the sub-arrays in counterclockwise order
        point_tags_sorted0 = self.sort_ccw(coords_split[0])
        # for each sub-array point_tags, sort the points with point_tags_sorted0
        point_tags_sorted = [
            point_tags[point_tags_sorted0] for point_tags in point_tags
        ]
        return point_tags_sorted

    def get_trabecular_vol(
        self, coi_idx: ndarray
    ) -> Tuple[ndarray, List[int], List[int], List[int], List[int], List[int]]:
        """
        Get the trabecular volume.

        This function calculates the trabecular volume by determining the trabecular positions,
        sorting the coordinates, creating lines and surfaces, and finally generating the volume.

        Args:
            coi_idx (ndarray): Array of center of interest indices.

        Returns:
            Tuple[ndarray, List[int], List[int], List[int], List[int], List[int]]:
                - Array of point tags.
                - List of vertical line tags.
                - List of horizontal line tags.
                - List of vertical surface tags.
                - List of horizontal surface tags.
                - List of volume tags.
        """
        self.coi_idx = coi_idx
        # ? add point tag to center to create self.ellipse_arcs?

        trabecular_points = self.get_trabecular_position()
        point_tags = self.insert_points(trabecular_points)
        point_tags_sorted = self.sort_trab_coords(point_tags.tolist())
        point_tags_r = np.reshape(point_tags_sorted, (-1, 4))

        # concatenate first point to the end of each subarray
        points_first_column = point_tags_r[:, 0]
        point_tags_c = np.concatenate(
            (point_tags_r, points_first_column[:, None]), axis=1
        )

        line_tags_h = []
        CENTER_ARC = self.LENGTH_FACTOR * 1.5
        signs = [(1, -1), (1, 1), (-1, 1), (-1, -1)]
        for i in range(len(point_tags_c[:, 0])):
            if self.ellipsoid_fitting is True:
                # * NEW (POS, 05.10.2023)
                line_tags_s = self.create_centre_splines(
                    i, point_tags_c, CENTER_ARC, signs
                )
            else:
                line_tags_s = self.insert_lines(point_tags_c[i])

            line_tags_h.append(line_tags_s)

        surf_tags_h = []
        for i, _ in enumerate(line_tags_h):
            trab_curveloop_h = self.factory.addCurveLoop(line_tags_h[i], tag=-1)
            if self.ellipsoid_fitting is True:
                trab_tag_h = self.factory.addSurfaceFilling(trab_curveloop_h, tag=-1)
            else:
                trab_tag_h = self.factory.addPlaneSurface([trab_curveloop_h], tag=-1)
            surf_tags_h.append(trab_tag_h)

        line_tags_v = []
        for j in range(len(point_tags_c[0, :]) - 1):
            line_tags_s = self.insert_lines(point_tags_c[:, j])
            line_tags_v.append(line_tags_s)

        line_tags_v = np.array(line_tags_v, dtype=int).reshape(
            (-1, (self.slicing_coefficient - 1))
        )
        line_tags_v = np.concatenate((line_tags_v, line_tags_v[:, 0][:, None]), axis=1)
        line_tags_v = np.append(
            line_tags_v, line_tags_v[0, :][None, :], axis=0
        ).tolist()

        line_tags_h = np.array(line_tags_h, dtype=int).reshape((-1, 4))
        line_tags_h = np.concatenate(
            (line_tags_h, line_tags_h[:, 0][:, None]), axis=1
        ).tolist()

        surf_tags_v = []
        self.logger.debug("Inner trabecular surface")
        self.logger.debug("j, i")
        for j in range(len(line_tags_v) - 1):
            line_tag = line_tags_v[j]
            for i in range(len(line_tag) - 1):
                self.logger.debug(
                    f"{j}, {i}:\t{line_tags_h[i][j]} {line_tags_v[j][i]} {line_tags_h[i + 1][j]} {line_tags_v[j + 1][i]}"
                )
                trab_curveloop_v = self.factory.addCurveLoop(
                    [
                        line_tags_h[i][j],
                        line_tags_v[j][i],
                        line_tags_h[i + 1][j],
                        line_tags_v[j + 1][i],
                    ],
                    tag=-1,
                )
                trab_tag_v = self.factory.addSurfaceFilling(trab_curveloop_v, tag=-1)
                surf_tags_v.append(trab_tag_v)

        # create volumes
        surf_tags_v = np.array(surf_tags_v).reshape((4, -1))
        surf_tags_h = np.array(surf_tags_h)

        trab_surf_loop_tag = []
        for i in range(3, len(surf_tags_v[:, 0])):
            for j in range(1, len(surf_tags_h)):
                trab_surf_loop_s = self.factory.addSurfaceLoop(
                    [
                        surf_tags_h[j - 1],
                        surf_tags_v[i - 3, j - 1],
                        surf_tags_v[i - 2, j - 1],
                        surf_tags_v[i - 1, j - 1],
                        surf_tags_v[i, j - 1],
                        surf_tags_h[j],
                    ],
                    tag=-1,
                )
                trab_surf_loop_tag.append(trab_surf_loop_s)

        # make volume
        trab_vol_tag = []
        if self.QUAD_REFINEMENT is not True:
            for _, surf_tag in enumerate(trab_surf_loop_tag):
                volume_t = self.factory.addVolume([surf_tag], tag=-1)
                trab_vol_tag.append(volume_t)

        self.line_tags_v = list(map(int, np.unique(line_tags_v)))
        self.line_tags_h = list(map(int, np.unique(line_tags_h)))
        self.surf_tags_v = list(map(int, np.unique(surf_tags_v)))
        self.surf_tags_h = list(map(int, np.unique(surf_tags_h)))
        self.vol_tags = list(map(int, trab_vol_tag))

        return (
            point_tags_r,
            self.line_tags_v,
            self.line_tags_h,
            self.surf_tags_v,
            self.surf_tags_h,
            self.vol_tags,
        )

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

    def set_length_factor(self, length_factor: float):
        """
        Set the length factor.

        This function sets the length factor used to calculate
        the size of the central trabecular region.

        Args:
            length_factor (float): The length factor to be set.

        Returns:
            None
        """
        self.LENGTH_FACTOR = length_factor
