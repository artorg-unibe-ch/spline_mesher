import gmsh
from pathlib import Path
import numpy as np
import shapely.geometry as shpg
from scipy import spatial
import matplotlib.pyplot as plt
import sys


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

        coords = np.array(
            [
                gmsh.model.getValue(0, point[1], [])
                for point in gmsh.model.getEntities(0)
            ]
        )
        z_list = np.unique(coords[:, 2])
        # for z in z_list:
        #     print(f"z: {z}")
        #     slices = coords[coords[:, 2] == z]
        #     print(slices)
        #     plt.plot(slices[:, 0], slices[:, 1], "o")
        #     plt.show()
        #     print("a")
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
        # fmt: off
        dists, closest_idx_2 = spatial.KDTree(arr).query(intersection, k=2)
        if (closest_idx_2 == [0, len(arr) - 1]).all():
            return dists, closest_idx_2[1]
        else:
            print(f"closest index where to insert the intersection point: {np.min(closest_idx_2)} (indices: {closest_idx_2})")
            return dists, np.min(closest_idx_2)
        # fmt: on

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

        # TODO: rm after debug
        fig, ax = plt.subplots()
        # ax.set_title("Calculating nearest neighbor of the intersection point")
        # ax.scatter(array[:, 0], array[:, 1])
        # ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
        # ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
        # for i, txt in enumerate(array[:, 0]):
        #     ax.annotate(i, (array[:, 0][i], array[:, 1][i]))
        # plt.show()

        idx_list = []
        for i, intersection in enumerate(intersections):
            for j, inters in enumerate(intersection):
                dists, closest_idx = self.intersection_point(array, inters)
                # array = self.insert_closest_point(array, closest_idx, inters)

                # TODO: rm after debug
                # fig, ax = plt.subplots()
                # ax.set_title("BEFORE")
                # ax.plot(array[:, 0], array[:, 1])
                # ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
                # ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
                # for n, txt in enumerate(array[:, 0]):
                #     ax.annotate(n, (array[:, 0][n], array[:, 1][n]))
                # plt.savefig(f"before_{i}_{j}.png")
                array[closest_idx] = inters

                """
                if abs(dists[0] - dists[1]) < 1e-5:
                    array = np.insert(array, closest_idx, inters, axis=0)
                else:
                    try:
                        array = np.insert(array, closest_idx + 1, inters, axis=0)
                    except:
                        array = np.insert(array, 0, inters, axis=0)
                """

                # TODO: rm after debug
                # fig, ax = plt.subplots()
                # ax.set_title("AFTER")
                # ax.plot(array[:, 0], array[:, 1])
                # ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
                # ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
                # for n, txt in enumerate(array[:, 0]):
                #     ax.annotate(n, (array[:, 0][n], array[:, 1][n]))
                # plt.savefig(f"after_{i}_{j}.png")

                idx_list.append(closest_idx)
        # TODO: rm after debug
        #  fig, ax = plt.subplots()
        #  ax.set_title("Inserting intersection points into the contours")
        #  ax.plot(array[:, 0], array[:, 1])
        #  ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
        #  ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
        #  for i, txt in enumerate(array[:, 0]):
        #      ax.annotate(i, (array[:, 0][i], array[:, 1][i]))
        #  plt.show()
        # array = self.close_loop(array)  # TODO: maybe reactivate it?
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

    def gmsh_insert_bspline(self, point_tags_tuple, points):
        point_tags_list = [tup[1] for tup in point_tags_tuple]
        point_tags_np = np.array(point_tags_list)
        indexed_points = point_tags_np[points - 1].tolist()
        line = self.factory.addBSpline(indexed_points)
        return line

    def insert_intersection_line(self, point_tags_tuple, idx_list: list):
        point_tags = [tup[1] for tup in point_tags_tuple]
        #  for each slice, take only the point_tags indexed by the idx_list
        reshaped_point_tags = np.reshape(point_tags, (len(idx_list), -1))
        indexed_points = reshaped_point_tags[
            np.arange(len(idx_list))[:, np.newaxis], idx_list
        ]

        line_tags = []
        for j in range(len(indexed_points[0, :])):
            for i in range(len(indexed_points[:, j]) - 1):
                line = self.factory.addLine(
                    indexed_points[i][j], indexed_points[i + 1][j]
                )
                line_tags = np.append(line_tags, line)
        return line_tags, indexed_points

    def gmsh_geometry_formulation(self, array: np.ndarray, idx_list: list):
        # points
        array_pts_tags = []
        for i, _ in enumerate(array):
            array_tag = self.gmsh_add_points(array[i][0], array[i][1], array[i][2])
            array_pts_tags = np.append(array_pts_tags, array_tag)
        array_pts_tags = np.asarray(array_pts_tags, dtype=int)

        #  insert line at the intersection point with the center of inertia
        intersection_line_tag, indexed_points_coi = self.insert_intersection_line(
            self.factory.get_entities(0), idx_list
        )

        # bsplines
        array_pts_tags_split = np.array_split(
            array_pts_tags, len(np.unique(array[:, 2]))
        )
        idx_list_sorted = np.sort(idx_list)
        #  add last column at the beginning of the array
        idx_list_sorted = np.insert(idx_list_sorted, 0, idx_list_sorted[:, -1], axis=1)
        # array_bspline = np.empty([len(array_pts_tags_split)])
        array_bspline = []

        for j in range(len(idx_list_sorted[:, 0])):
            for i, _ in enumerate(array_pts_tags_split):
                # 1. duplicate the array
                array_pts_tags_split[i] = np.vstack(
                    np.vstack((array_pts_tags_split[i], array_pts_tags_split[i]))
                ).flatten()
                # 2. section the array with the idx_list
                array_split_idx = array_pts_tags_split[i][
                    min(
                        np.where(
                            array_pts_tags_split[i]
                            == array_pts_tags_split[i][idx_list_sorted[j, 0]]
                        )[0]
                    ) : max(
                        np.where(
                            array_pts_tags_split[i]
                            == array_pts_tags_split[i][idx_list_sorted[j, 1]]
                        )
                    )[
                        1
                    ]
                    + 1
                ]

                array_bspline_s = self.gmsh_insert_bspline(
                    self.factory.get_entities(0), array_split_idx
                )
                array_bspline = np.append(array_bspline, array_bspline_s)
        return array_pts_tags  # , array_bspline, intersection_line_tag
