import gmsh
from pathlib import Path
import numpy as np
import shapely.geometry as shpg
from scipy import spatial


class Mesher:
    def __init__(self, geo_file_path, mesh_file_path):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path

    def write_msh(self):
        gmsh.model.mesh.generate(2)
        gmsh.write(self.mesh_file_path)
        return None

    def build_msh(self):
        modelname = Path(self.geo_file_path).stem
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add(modelname)
        gmsh.merge(self.geo_file_path)

        self.factory.synchronize()

        # self.write_msh()
        gmsh.fltk.run()
        gmsh.finalize()

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

    def close_loop(self, array):
        array = np.vstack((array, array[0]))
        return array

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
        return array, idx_list

    # adding here all the functions related to the mesh building

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

        line_tags = []

        sorted_indexed_points = self.sort_intersection_points(indexed_points)
        for j in range(len(sorted_indexed_points[0, :])):
            for i in range(len(sorted_indexed_points[:, j]) - 1):
                line = self.factory.addLine(
                    sorted_indexed_points[i][j], sorted_indexed_points[i + 1][j]
                )
                line_tags = np.append(line_tags, line)
        return line_tags, sorted_indexed_points

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
        #  add last column at the beginning of the array
        idx_list_sorted = np.insert(idx_list_sorted, 0, idx_list_sorted[:, -1], axis=1)

        array_pts_tags_split = [
            np.append(array_pts_tags_split[i], array_pts_tags_split[i])
            for i in range(len(array_pts_tags_split))
        ]

        array_bspline = []
        for j in range(len(idx_list_sorted[0, :]) - 1):
            for i, _ in enumerate(array_pts_tags_split):
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

                array_bspline_s = self.gmsh_insert_bspline(array_split_idx)
                array_bspline = np.append(array_bspline, array_bspline_s)
        return array_bspline

    def gmsh_geometry_formulation(self, array: np.ndarray, idx_list: list):

        array_pts_tags = self.insert_points(array)

        intersection_line_tag, indexed_points_coi = self.insert_intersection_line(
            array_pts_tags, idx_list
        )

        array_bspline = self.insert_bspline(array, array_pts_tags, indexed_points_coi)

        return array_pts_tags, array_bspline, intersection_line_tag
