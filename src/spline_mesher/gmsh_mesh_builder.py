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
        _, closest_idx_2 = spatial.KDTree(arr).query(intersection, k=2)
        if (closest_idx_2 == [0, len(arr) - 1]).all():
            return closest_idx_2[1]
        else:
            print(f"closest index where to insert the intersection point: {np.min(closest_idx_2)} (indices: {closest_idx_2})")
            return np.min(closest_idx_2)
        # fmt: on

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

    def close_loop_if_open(self, array):
        if array[0][0] != array[-1][0] and array[0][1] != array[-1][1]:
            array[0][0] = array[-1][0]
            array[0][1] = array[-1][1]
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
        ax.set_title("Calculating nearest neighbor of the intersection point")
        ax.scatter(array[:, 0], array[:, 1])
        ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
        ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
        for i, txt in enumerate(array[:, 0]):
            ax.annotate(i, (array[:, 0][i], array[:, 1][i]))
        plt.show()

        idx_list = []
        for i, intersection in enumerate(intersections):
            for j, inters in enumerate(intersection):
                closest_idx = self.intersection_point(array, inters)
                # array = self.insert_closest_point(array, closest_idx, inters)

                # # TODO: rm after debug
                # fig, ax = plt.subplots()
                # ax.set_title("BEFORE")
                # ax.plot(array[:, 0], array[:, 1])
                # ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
                # ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
                # for i, txt in enumerate(array[:, 0]):
                #     ax.annotate(i, (array[:, 0][i], array[:, 1][i]))
                # plt.show()
                if closest_idx == len(array) - 1:
                    array = np.vstack((array, array[-1]))
                else:
                    array = np.insert(array, closest_idx + 1, inters, axis=0)

                # # TODO: rm after debug
                # fig, ax = plt.subplots()
                # ax.set_title("AFTER")
                # ax.plot(array[:, 0], array[:, 1])
                # ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
                # ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
                # for i, txt in enumerate(array[:, 0]):
                #     ax.annotate(i, (array[:, 0][i], array[:, 1][i]))
                # plt.show()

                idx_list.append(closest_idx)
        # TODO: rm after debug
        fig, ax = plt.subplots()
        ax.set_title("Inserting intersection points into the contours")
        ax.plot(array[:, 0], array[:, 1])
        ax.scatter(intersections[0][:, 0], intersections[0][:, 1], color="red")
        ax.scatter(intersections[1][:, 0], intersections[1][:, 1], color="red")
        for i, txt in enumerate(array[:, 0]):
            ax.annotate(i, (array[:, 0][i], array[:, 1][i]))
        plt.show()
        array = self.close_loop_if_open(array)  # TODO: maybe reactivate it?
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

    def gmsh_insert_bspline(self, points):
        line = self.factory.addBSpline(points)
        return line

    def gmsh_geometry_formulation(self, array: np.ndarray):
        # points
        array_pts_tags = []
        for i, _ in enumerate(array):
            array_tag = self.gmsh_add_points(
                array[i][0],
                array[i][1],
                array[i][2],
            )
            array_pts_tags = np.append(array_pts_tags, array_tag)
        array_pts_tags = np.asarray(array_pts_tags, dtype=int)

        # bsplines
        array_pts_tags = np.array_split(array_pts_tags, len(np.unique(array[:, 2])))
        array_bspline = np.empty([len(array_pts_tags)])
        for i, _ in enumerate(array_pts_tags):
            array_bspline[i] = self.gmsh_insert_bspline(array_pts_tags[i])
        return array_pts_tags, array_bspline


if __name__ == "__main__":
    geo_file_path = r"/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/04_OUTPUT/C0002237/fake_example.geo_unrolled"
    mesh_file_path = r"/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/04_OUTPUT/C0002237/C0002237.msh"
    mesher = Mesher(geo_file_path, mesh_file_path)
    mesher.build_msh()
