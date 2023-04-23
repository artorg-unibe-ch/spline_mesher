import logging
import math

import cv2
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shpg
from scipy import spatial
from skimage.util import view_as_windows

# flake8: noqa: E203
LOGGING_NAME = "SIMONE"


class Mesher:
    def __init__(
        self,
        geo_file_path,
        mesh_file_path,
        slicing_coefficient,
        n_transverse,
        n_radial,
    ):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.option = gmsh.option
        self.plugin = gmsh.plugin
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path
        self.slicing_coefficient = slicing_coefficient
        self.n_transverse = n_transverse
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

    def insert_tensor_of_inertia(self, array, centroid) -> np.ndarray:
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
            array_tag = self.factory.addLine(array[i], array[i + 1])
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

        point_tags_sliced = np.array(point_tags_sliced)
        # sort the sub-arrays in c-clockwise order
        point_tags_sliced_sorted = np.copy(point_tags_sliced)
        for i, coord_slice in enumerate(coords_sliced):
            point_tags_sorted = self.sort_ccw(coord_slice)
            point_tags_sliced_sorted[i] = point_tags_sliced[i][point_tags_sorted]
        return point_tags_sliced_sorted

    def gmsh_insert_bspline(self, points):
        points = np.array(points).tolist()
        b_spline = self.factory.addBSpline(points)
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

    def gmsh_geometry_formulation(self, array: np.ndarray, idx_list: list):
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
        self, intersection_tags, bspline_tags, surface_tags, volume_tags, test_list
    ):
        self.factory.synchronize()

        intersection_tags = list(map(int, intersection_tags))
        bspline_tags = list(map(int, bspline_tags))
        surface_tags = list(map(int, surface_tags))

        for test in test_list:
            self.model.mesh.setTransfiniteCurve(
                test, 8, "Progression", 1.0
            )  # TODO: works fine, but needs to be parametrized (number of trabecular radial points)

        for intersection in intersection_tags:
            self.model.mesh.setTransfiniteCurve(intersection, self.n_transverse)

        for bspline in bspline_tags:
            self.model.mesh.setTransfiniteCurve(bspline, self.n_radial)

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

    def trabecular_cortical_connection(
        self, coi_idx: list[int], trab_point_tags: list[int]
    ):
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
        trab_cort_line_tags: list[int],
        trab_line_tags_h: list[int],
        cort_int_bspline_tags: list[int],
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

    def get_mesh(self):
        # Get all the elementary entities in the model, as a vector of (dimension, tag)
        # pairs:
        self.factory.synchronize()
        physicalgroups = self.model.getPhysicalGroups(3)

        entities = []
        nodeTags = []
        elemTags = []
        for pg in physicalgroups:
            ent = self.model.getEntitiesForPhysicalGroup(3, pg[1])
            entities.append(ent)

        for sub_entities in entities:
            for e in sub_entities:
                # Dimension and tag of the entity:
                dim = 3
                tag = e

                # Mesh data is made of `elements' (points, lines, triangles, ...), defined
                # by an ordered list of their `nodes'. Elements and nodes are identified by
                # `tags' as well (strictly positive identification numbers), and are stored
                # ("classified") in the model entity they discretize. Tags for elements and
                # nodes are globally unique (and not only per dimension, like entities).

                # A model entity of dimension 0 (a geometrical point) will contain a mesh
                # element of type point, as well as a mesh node. A model curve will contain
                # line elements as well as its interior nodes, while its boundary nodes will
                # be stored in the bounding model points. A model surface will contain
                # triangular and/or quadrangular elements and all the nodes not classified
                # on its boundary or on its embedded entities. A model volume will contain
                # tetrahedra, hexahedra, etc. and all the nodes not classified on its
                # boundary or on its embedded entities.

                # Get the mesh nodes for the entity (dim, tag):
                nodeTags, nodeCoords, nodeParams = self.model.mesh.getNodes(dim, tag)

                # Get the mesh elements for the entity (dim, tag):
                elemTypes, elemTags, elemNodeTags = self.model.mesh.getElements(
                    dim, tag
                )

                # Elements can also be obtained by type, by using `getElementTypes()'
                # followed by `getElementsByType()'.

                # Let's print a summary of the information available on the entity and its
                # mesh.

                # * Type and name of the entity:
                type = self.model.getType(e[0], e[1])
                name = self.model.getEntityName(e[0], e[1])
                if len(name):
                    name += " "
                print("Entity " + name + str(e) + " of type " + type)

                # * Number of mesh nodes and elements:
                numElem = sum(len(i) for i in elemTags)
                print(
                    " - Mesh has "
                    + str(len(nodeTags))
                    + " nodes and "
                    + str(numElem)
                    + " elements"
                )

                # * Upward and downward adjacencies:
                up, down = self.model.getAdjacencies(e[0], e[1])
                if len(up):
                    print(" - Upward adjacencies: " + str(up))
                if len(down):
                    print(" - Downward adjacencies: " + str(down))

                # * Does the entity belong to physical groups?
                physicalTags = self.model.getPhysicalGroupsForEntity(dim, tag)
                if len(physicalTags):
                    s = ""
                    for p in physicalTags:
                        n = self.model.getPhysicalName(dim, p)
                        if n:
                            n += " "
                        s += n + "(" + str(dim) + ", " + str(p) + ") "
                    print(" - Physical groups: " + s)

                # * Is the entity a partition entity? If so, what is its parent entity?
                partitions = self.model.getPartitions(e[0], e[1])
                if len(partitions):
                    print(
                        " - Partition tags: "
                        + str(partitions)
                        + " - parent entity "
                        + str(self.model.getParent(e[0], e[1]))
                    )

                # * List all types of elements making up the mesh of the entity:
                for t in elemTypes:
                    (
                        name,
                        dim,
                        order,
                        numv,
                        parv,
                        _,
                    ) = self.model.mesh.getElementProperties(t)
                    print(
                        " - Element type: "
                        + name
                        + ", order "
                        + str(order)
                        + " ("
                        + str(numv)
                        + " nodes in param coord: "
                        + str(parv)
                        + ")"
                    )
                    nodeTags.append(nodeTags)
                    elemTags.append(elemTags)

            return nodeTags, elemTags


class TrabecularVolume(Mesher):
    def __init__(
        self,
        geo_file_path,
        mesh_file_path,
        slicing_coefficient,
        n_transverse,
        n_radial,
        QUAD_REFINEMENT,
    ):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path
        self.slicing_coefficient = slicing_coefficient
        self.n_transverse = n_transverse
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
            n_transverse,
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

    def trabecular_transfinite(self, line_tags_v, line_tags_h, surf_tags, vol_tags):
        # make transfinite
        self.factory.synchronize()
        for line in line_tags_v:
            self.model.mesh.setTransfiniteCurve(line, self.n_transverse)
        for line in line_tags_h:
            self.model.mesh.setTransfiniteCurve(line, self.n_radial)
        for surface in surf_tags:
            self.model.mesh.setTransfiniteSurface(surface)
        for volume in vol_tags:
            self.model.mesh.setTransfiniteVolume(volume)

    def set_length_factor(self, length_factor):
        self.LENGTH_FACTOR = length_factor


class QuadRefinement(TrabecularVolume):
    def __init__(
        self,
        nb_layers: int,
        DIM: int = 2,
        SQUARE_SIZE_0_MM: int = 1,
        MAX_SUBDIVISIONS: int = 3,
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

    def center_of_mass(self, vertices_coords):
        return np.mean(vertices_coords, axis=0)

    def get_vertices_coords(self, vertices_tags):
        self.factory.synchronize()
        vertices_coords = []
        for vertex in vertices_tags:
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
        # line_tags_temp = line_tags + (ext_lines,)

        # connecting lines
        connecting_lines = [
            [initial_point_tags[0], pt[93]],
            [initial_point_tags[1], pt[86]],
            [initial_point_tags[2], pt[16]],
            [initial_point_tags[3], pt[23]],
        ]

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

        # create the surfaces
        surf_tags = []

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

        for cl in curve_loops:
            _surf = self.factory.addPlaneSurface([cl])
            surf_tags.append(_surf)

        # insert connecting line to connecting_lines_app
        connecting_lines.insert(0, connecting_lines[-1])
        self.factory.synchronize()
        surf_tags = list(map(int, surf_tags))
        for i in range(1, len(surf_tags) + 1):
            corners = (
                connecting_lines[i][0],
                connecting_lines[i][1],
                connecting_lines[i - 1][1],
                connecting_lines[i - 1][0],
            )
            self.logger.debug(surf_tags[i - 1])
            self.logger.debug(corners)
            self.model.mesh.setTransfiniteSurface(
                surf_tags[i - 1],
                arrangement="Left",
                cornerTags=corners,
            )
        return line_tags_new, surf_tags

    def quad_refinement(
        self,
        vertices_tags: list[int],
    ):
        SQUARE_SIZE_0_MM = self.SQUARE_SIZE_0_MM
        MAX_SUBDIVISIONS = self.MAX_SUBDIVISIONS

        vertices_tags = np.array(vertices_tags)

        # * 1. get the vertices coordinates from self.vertices_tags
        vertices_coords = self.get_vertices_coords(vertices_tags)
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
        line_tags, surf_tags = self.gmsh_add_outer_connectivity(
            line_tags, vertices_tags, point_tags
        )
        return line_tags, surf_tags
