import gmsh
import numpy as np
import shapely.geometry as shpg
from scipy import spatial
from scipy.spatial import KDTree


class Mesher:
    def __init__(
        self, geo_file_path, mesh_file_path, slicing_coefficient, n_transverse, n_radial
    ):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path
        self.slicing_coefficient = slicing_coefficient
        self.n_transverse = n_transverse
        self.n_radial = n_radial

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
            # print(
            #     f"closest index where to insert the intersection point: \
            # {np.min(closest_idx_2)} (indices: {closest_idx_2})"
            # )
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
        radius = 50
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

    def sort_intersection_points(self, array):
        """
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

        # sort the sub-arrays in clockwise order
        point_tags_sliced_sorted = []
        for i, coord_slice in enumerate(coords_sliced):
            point_tags_sorted = self.sort_bspline_cw(coord_slice, point_tags_sliced[i])
            point_tags_sliced_sorted.append(point_tags_sorted)
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
        ext_r = ext_tags.reshape((self.slicing_coefficient, -1))
        int_r = int_tags.reshape((self.slicing_coefficient, -1))
        inter_r = interslice_seg_tags.reshape((self.slicing_coefficient, -1))
        inter_c = np.concatenate(
            (inter_r[:, 0:], inter_r[:, 0].reshape(self.slicing_coefficient, 1)), axis=1
        )
        inter_a = np.concatenate((inter_c, inter_c[-1].reshape(1, -1)), axis=0)

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
        for i in range(len(intersurface_line_tags_r) - 1):
            intersurface_line_tags_r_i = intersurface_line_tags_r[i]
            for j in range(len(intersurface_line_tags_r_i)):
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
        self, intersection_tags, bspline_tags, surface_tags, volume_tags
    ):
        self.factory.synchronize()

        intersection_tags = list(map(int, intersection_tags))
        bspline_tags = list(map(int, bspline_tags))
        surface_tags = list(map(int, surface_tags))

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

    def query_closest_idx(self, list_1: list, list_2: list):
        """# TODO: write docstring

        Args:
            coi (list): _description_
            array (list): _description_
        """ """"""
        coi_coords = []
        trab_coords = []
        for i, idx in enumerate(list_1):
            coi_coords_s = [self.model.getValue(0, idx, []) for idx in list_1[i]]
            trab_coords_s = [
                self.model.getValue(0, idx, []) for idx in list_2[i]
            ]  # noqa
            coi_coords.append(coi_coords_s)
            trab_coords.append(trab_coords_s)

        # coi_points = [self.model.getValue(0, idx, []) for idx in coi_idx]
        # trab_points = [self.model.getValue(0, idx, []) for idx in trab_point_tags]
        coi_idx_s = np.array(list_1, dtype=int).flatten()
        trab_point_tags_s = np.array(list_2, dtype=int).flatten()
        # for each trab_point_tags, find the closest point in coi_idx with KDTree
        tree = KDTree(np.array(coi_coords).reshape((-1, 3)))
        _, ind = tree.query(np.array(trab_coords).reshape((-1, 3)))
        coi_closest = (coi_idx_s[ind]).reshape((-1, 4))
        trab_idx_closest = trab_point_tags_s.reshape((-1, 4))
        return coi_closest, trab_idx_closest

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

        coi_closest, trab_idx_closest = self.query_closest_idx(coi_idx, trab_point_tags)

        for i, _ in enumerate(coi_closest):
            for j, _ in enumerate(coi_closest[i]):
                # print(f"{coi_closest[i][j]}\t{trab_idx_closest[i][j]}")
                self.factory.addLine(coi_closest[i][j], trab_idx_closest[i][j], tag=-1)

        print("testing")

    def mesh_generate(self, dim):
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)
        gmsh.option.setNumber("Mesh.Recombine3DLevel", 2)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        self.model.mesh.generate(dim)


class TrabecularVolume(Mesher):
    def __init__(
        self, geo_file_path, mesh_file_path, slicing_coefficient, n_transverse, n_radial
    ):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path
        self.slicing_coefficient = slicing_coefficient
        self.n_transverse = n_transverse
        self.n_radial = n_radial
        self.coi_idx = []
        self.line_tags_v = []
        self.line_tags_h = []
        self.surf_tags = []
        self.vol_tags = []
        self.LENGTH_FACTOR = float(1.0)
        super().__init__(
            geo_file_path, mesh_file_path, slicing_coefficient, n_transverse, n_radial
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

    def sort_cw(self, coords):
        coords = np.asarray(coords)
        center = np.mean(coords, axis=0)
        angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_indices = list(sorted_indices)
        return sorted_indices

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
            trabecular_points[i] = trabecular_points[i][[0, 2, 1, 3]]
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

        # sort the sub-arrays in clockwise order
        point_tags_sorted0 = self.sort_cw(coords_split[0])
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
        for j in range(len(line_tags_v) - 1):
            line_tag = line_tags_v[j]
            for i in range(len(line_tag) - 1):
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
        for _, surf_tag in enumerate(trab_surf_loop_tag):
            volume_t = self.factory.addVolume([surf_tag], tag=-1)
            trab_vol_tag.append(volume_t)

        self.line_tags_h = list(map(int, np.unique(line_tags_h)))
        self.line_tags_v = list(map(int, np.unique(line_tags_v)))
        self.surf_tags = np.append(np.unique(surf_tags_h), np.unique(surf_tags_v))
        self.surf_tags = list(map(int, self.surf_tags))
        self.vol_tags = list(map(int, trab_vol_tag))

        return (
            point_tags_r,
            self.line_tags_v,
            self.line_tags_h,
            self.surf_tags,
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
