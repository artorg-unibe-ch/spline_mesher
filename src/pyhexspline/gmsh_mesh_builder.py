import time
import logging
import math

import cv2
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shpg
import scipy.spatial as spatial
from cython_functions import find_closed_curve as fcc
from itertools import chain

# flake8: noqa: E203
LOGGING_NAME = "MESHING"


class Mesher:
    def __init__(
        self,
        geo_file_path,
        mesh_file_path,
        slicing_coefficient,
        n_longitudinal,
        n_transverse_cort,
        n_transverse_trab,
        n_radial,
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
        self.logger = logging.getLogger(LOGGING_NAME)

    def polygon_tensor_of_inertia(self, ext_arr, int_arr) -> tuple:
        ext_arr = np.vstack((ext_arr, ext_arr[0]))
        int_arr = np.vstack((int_arr, int_arr[0]))
        extpol = shpg.polygon.Polygon(ext_arr)
        intpol = shpg.polygon.Polygon(int_arr)
        cortex = extpol.difference(intpol)
        cortex_centroid = (
            cortex.centroid.coords.xy[0][0],
            cortex.centroid.coords.xy[1][0],
        )
        return cortex_centroid

    def shapely_line_polygon_intersection(self, poly, line_1):
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

    def partition_lines(self, radius, centroid):
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

    def intersection_point(self, arr, intersection):
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

    def insert_tensor_of_inertia(self, array, centroid) -> tuple:
        """
        1. calculate centroid of the cortex
        2. calculate intersection of the centroid with the contours
        3. calculate nearest neighbor of the intersection point
        4. insert the intersection point into the contours
        """
        radius = 100
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

    def gmsh_add_points(self, x, y, z):
        """
        https://gitlab.onelab.info/gmsh/gmsh/-/issues/456
        https://bbanerjee.github.io/ParSim/fem/meshing/gmsh/quadrlateral-meshing-with-gmsh/
        """
        gmsh.option.setNumber("General.Terminal", 1)
        point_tag = self.factory.addPoint(x, y, z, tag=-1)
        return point_tag

    def insert_points(self, array):
        array_pts_tags = []
        for i, _ in enumerate(array):
            array_tag = self.gmsh_add_points(array[i][0], array[i][1], array[i][2])
            array_pts_tags = np.append(array_pts_tags, array_tag)
        array_pts_tags = np.asarray(array_pts_tags, dtype=int)
        return array_pts_tags

    def insert_lines(self, array):
        array_lines_tags = []
        for i in range(len(array) - 1):
            array_tag = self.factory.addLine(array[i], array[i + 1], tag=-1)
            array_lines_tags.append(array_tag)
        return array_lines_tags

    def sort_intersection_points_legacy(self, array):
        """
        Legacy function to sort the intersection points in cw direction
        Kept for reference, but now using self.sort_intersection_points()
        Sort the intersection points in cw direction
        """
        self.factory.synchronize()
        array_sorted = []
        for i, subarray in enumerate(array):
            point = [self.model.getValue(0, point, []) for point in subarray]
            point_np = np.array(point, dtype=float)
            centroid = np.mean(point_np, axis=0)
            dists = point_np - centroid
            angles_subarray = np.arctan2(dists[:, 1], dists[:, 0])
            idx = np.argsort(angles_subarray)
            array_sorted.append(array[i][idx])
        return np.array(array_sorted, dtype=int)

    def sort_intersection_points(self, array):
        """
        Sort the intersection points in ccw direction
        """
        self.factory.synchronize()
        points = []
        for _, subarray in enumerate(array):
            points_subarr = [self.model.getValue(0, point, []) for point in subarray]
            points.append(points_subarr)
        points_sorted = self.sort_ccw(points[0])
        array_sorted = [subarr[points_sorted] for subarr in array]
        return np.array(array_sorted, dtype=int)

    def insert_intersection_line(self, point_tags, idx_list: list):
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
        coords = np.asarray(coords)
        center = np.mean(coords, axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_indices = list(sorted_indices)
        point_tags_sorted = [point_tags[i] for i in sorted_indices]
        return point_tags_sorted

    def get_sort_coords(self, point_tags):
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

    def gmsh_insert_bspline(self, points):
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

    def add_bspline_filling(self, bsplines, intersections, slicing_coefficient):
        # ? see if it works
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
                )

                BSplineFilling = self.factory.addBSplineFilling(
                    WIRE, type="Stretch", tag=-1
                )
                # BSplineFilling = self.factory.addBSplineFilling(W1, type="Curved", tag=-1)

                BSplineFilling_tags = np.append(BSplineFilling_tags, BSplineFilling)
        return BSplineFilling_tags

    def add_interslice_segments(self, point_tags_ext, point_tags_int):
        point_tags_ext = point_tags_ext.flatten().tolist()
        point_tags_int = point_tags_int.flatten().tolist()

        interslice_seg_tag = []
        for i, _ in enumerate(point_tags_ext):
            interslice_seg_tag_s = self.factory.addLine(
                point_tags_ext[i], point_tags_int[i], tag=-1
            )
            interslice_seg_tag = np.append(interslice_seg_tag, interslice_seg_tag_s)

        return interslice_seg_tag

    def add_slice_surfaces(self, ext_tags, int_tags, interslice_seg_tags):
        ext_r = ext_tags.reshape((self.slicing_coefficient, -1)).astype(int)
        int_r = int_tags.reshape((self.slicing_coefficient, -1)).astype(int)
        inter_r = interslice_seg_tags.reshape((self.slicing_coefficient, -1))
        inter_c = np.concatenate(
            (inter_r[:, 0:], inter_r[:, 0].reshape(self.slicing_coefficient, 1)), axis=1
        )
        inter_a = np.concatenate((inter_c, inter_c[-1].reshape(1, -1)), axis=0).astype(
            int
        )

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

        ext_int_tags_l = np.array(ext_int_tags, dtype="int").tolist()
        slices_ext_int_tags = []
        for surf_tag in ext_int_tags_l:
            slice_tag = self.factory.addSurfaceFilling(surf_tag, tag=-1)
            slices_ext_int_tags.append(slice_tag)
        return np.array(slices_ext_int_tags)

    def add_intersurface_planes(
        self,
        intersurface_line_tags,
        intersection_line_tag_ext,
        intersection_line_tag_int,
    ):
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

    def gmsh_geometry_formulation(self, array: np.ndarray, idx_list: np.ndarray):
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
        longitudinal_line_tags,
        transverse_line_tags,
        radial_line_tags,
        surface_tags,
        volume_tags,
        phase="cort",
    ):
        self.factory.synchronize()

        longitudinal_line_tags = list(map(int, longitudinal_line_tags))
        radial_line_tags = list(map(int, radial_line_tags))
        surface_tags = list(map(int, surface_tags))
        volume_tags = list(map(int, volume_tags))

        n_transverse = None
        if phase == "cort":
            n_transverse = self.n_transverse_cort
        elif phase == "trab":
            n_transverse = self.n_transverse_trab

        for ll in longitudinal_line_tags:
            self.model.mesh.setTransfiniteCurve(ll, self.n_longitudinal)

        for ll in transverse_line_tags:
            self.model.mesh.setTransfiniteCurve(ll, n_transverse)

        for intersection in radial_line_tags:
            self.model.mesh.setTransfiniteCurve(intersection, self.n_radial)

        for surface in surface_tags:
            self.model.mesh.setTransfiniteSurface(surface)

        for volume in volume_tags:
            self.model.mesh.setTransfiniteVolume(volume)

    def add_volume(
        self,
        cortical_ext_surfs,
        cortical_int_surfs,
        slices_tags,
        intersurface_surface_tags,
    ):
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
        return [
            val
            for (_, val) in sorted(
                zip(indexes, lst), key=lambda x: x[0], reverse=reverse
            )
        ]

    def query_closest_idx(self, list_1: list, list_2: list):
        """# TODO: write docstring

        Args:
            coi (list): _description_
            array (list): _description_
        """ """"""
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

    def trabecular_cortical_connection(self, coi_idx, trab_point_tags):
        """
        # TODO: write docstring

        Args:
            coi_idx (list[int]): _description_
            trab_point_tags (list[int]): _description_
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
        self, trab_cort_line_tags, trab_line_tags_v, intersection_line_tags_int
    ):
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
        trab_cort_line_tags,
        trab_line_tags_h,
        cort_int_bspline_tags,
    ):
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
        trab_slice_surf_tags,
        trab_plane_inertia_tags,
        cortical_int_surfs,
        trab_surfs_v,
    ):
        """
        # TODO: write docstring
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

    def sort_cw_legacy(self, coords):
        """check if this functions is still needed"""
        coords = np.asarray(coords)
        center = np.mean(coords, axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_indices = list(sorted_indices)
        # guarantee that the first point is always in the first quadrant
        if angles[sorted_indices[0]] > 0:
            self.logger.warning("First point is not in 1st quadrant, modifying.")
            self.logger.warning(f"Coords: {coords[sorted_indices[0]]}")
            self.logger.warning(f"Angle = {angles[sorted_indices[0]]}")
            sorted_indices = sorted_indices[1:] + [sorted_indices[0]]
        return sorted_indices

    def sort_ccw_test(self, coords):
        coords = np.asarray(coords)
        center = np.mean(coords, axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        closest_to_3 = np.argmin(np.abs(angles))
        angles_idx = np.arange(len(angles))
        sorted_indices_t = np.argsort(angles, kind="stable")
        sorted_indices = np.roll(angles_idx[sorted_indices_t], -closest_to_3)
        return sorted_indices

    def sort_ccw(self, coords: np.ndarray) -> np.ndarray:
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

    def mesh_generate(self, dim: int, element_order: int, optimise: bool = True):
        self.option.setNumber("Mesh.RecombineAll", 1)
        self.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        self.option.setNumber("Mesh.Recombine3DLevel", 2)
        self.option.setNumber("Mesh.ElementOrder", element_order)
        self.model.mesh.generate(dim)
        if optimise:
            self.model.mesh.optimize(method="Netgen", niter=3)

    def analyse_mesh_quality(self, hiding_thresh: float) -> None:
        self.plugin.setNumber("AnalyseMeshQuality", "JacobianDeterminant", 1)
        self.plugin.setNumber("AnalyseMeshQuality", "CreateView", 1)
        self.plugin.setNumber("AnalyseMeshQuality", "DimensionOfElements", -1)
        self.plugin.setNumber("AnalyseMeshQuality", "HidingThreshold", hiding_thresh)
        self.plugin.run("AnalyseMeshQuality")
        return None

    def gmsh_get_nodes(self):
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

    def gmsh_get_elms(self):
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

    def gmsh_get_bnds(self, nodes: dict):
        """creates two dictionaries defining the boundary nodes of the mesh based on their z coordinate

        Args:
            nodes (dict): node set of the mesh

        Returns:
            bnds_bot, bnds_top (dict): node sets of the bottom and top boundary of the mesh
        """
        z_min = float("inf")
        z_max = float("-inf")
        tol = 0.05

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

    def gmsh_get_reference_point_coord(self, nodes: dict):
        OFFSET_MM = 2  # RP offset in mm from top surface
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

    def get_centroids_dict(self, centroids):
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

    def split_dict_by_array_len(self, input_dict, len1):
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
        https://github.com/dmelgarm/gmsh/blob/master/gmsh_tools.py#L254
        Associate node coordinates to each element
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

    def get_barycenters(self, tag_s: list[int]):
        """
        Gets barycenters of each volumetric element of type 5 (8-node hexahedron) or 12 (27-node second order hexahedron)
        Returns:
            barycenters (np.array): array of barycenters in (x, y, z) format (shape: (n_elms, 3))
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

    def get_elm_volume(self, tag_s: list[int]):
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


class TrabecularVolume(Mesher):
    def __init__(
        self,
        geo_file_path,
        mesh_file_path,
        slicing_coefficient,
        n_longitudinal,
        n_transverse_cort,
        n_transverse_trab,
        n_radial,
        QUAD_REFINEMENT,
    ):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path
        self.slicing_coefficient = slicing_coefficient
        self.n_transverse_cort = n_transverse_cort
        self.n_transverse_trab = n_transverse_trab
        self.n_radial = n_radial
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
        )

    def principal_axes_length(self, array):
        l_i = np.linalg.norm(array[0] - array[1])
        l_j = np.linalg.norm(array[0] - array[2])
        return l_i, l_j

    def get_offset_points(self, array, _center, l_i, l_j):
        # calculate offset point from _center and float(LENGTH_FACTOR) * half of the principal axes length
        offset_i = [self.LENGTH_FACTOR * l_i / 2, 0, 0]
        offset_j = [0, self.LENGTH_FACTOR * l_j / 2, 0]

        # replace the offset point in the array
        array[0] = np.array(_center) + np.array(offset_i)
        array[1] = np.array(_center) - np.array(offset_i)
        array[2] = np.array(_center) + np.array(offset_j)
        array[3] = np.array(_center) - np.array(offset_j)
        return array

    def get_trabecular_position(self):
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

            # sort points in cw direction
            # trabecular_points[i] = trabecular_points[i][[0, 2, 1, 3]]  # ? is this necessary?
        return np.array(trabecular_points, dtype=np.float32).reshape((-1, 3))

    def sort_trab_coords(self, point_tags):
        self.factory.synchronize()
        start_point = [lst for lst in point_tags]
        coords = [self.model.getValue(0, point, []) for point in start_point]
        point_tags = np.split(
            np.array(point_tags), self.slicing_coefficient
        )  # split every 4 points
        coords_split = np.split(
            np.array(coords), self.slicing_coefficient
        )  # split every x, y, z coordinate

        # sort the sub-arrays in counterclockwise order
        point_tags_sorted0 = self.sort_ccw(coords_split[0])  # ! HERE
        # for each sub-array point_tags, sort the points with point_tags_sorted0
        point_tags_sorted = [
            point_tags[point_tags_sorted0] for point_tags in point_tags
        ]
        return point_tags_sorted

    def get_trabecular_vol(self, coi_idx):
        self.coi_idx = coi_idx
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
        for i in range(len(point_tags_c[:, 0])):
            line_tags_s = self.insert_lines(point_tags_c[i])
            line_tags_h.append(line_tags_s)

        surf_tags_h = []
        for i, _ in enumerate(line_tags_h):
            trab_curveloop_h = self.factory.addCurveLoop(line_tags_h[i], tag=-1)
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
        self.factory.synchronize()
        vertices_coords = []
        for vertex in vertices_tags:
            vertices_coords.append(self.model.getValue(0, vertex, []))
        return np.array(vertices_coords, dtype=np.float32)

    def set_length_factor(self, length_factor):
        self.LENGTH_FACTOR = length_factor
