import gmsh
from pathlib import Path
import numpy as np
import shapely.geometry as shpg
from scipy import spatial
import matplotlib.pyplot as plt


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
        for z in z_list:
            print(f"z: {z}")
            slices = coords[coords[:, 2] == z]
            print(slices)
            plt.plot(slices[:, 0], slices[:, 1], "o")
            plt.show()
            print("a")
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

    def shapely_line_polygon_intersection(poly, line_1):
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

    def partition_lines(radius, centroid):
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

    def intersection_point(self, arr, intersection_1):
        """
        Insert intersection point between two lines in the array
        Args:
            arr (ndarray): array of points
            intersection_1 (ndarray): intersection point 1
            intersection_2 (ndarray): intersection point 2
        Returns:
            ndarray: new array with intersection points
        """
        _, closest_idx = spatial.KDTree(arr).query(intersection_1)
        print(f"closest index where to insert the intersection point: {closest_idx}")
        return closest_idx

    def insert_closest_point(arr, closest_idx, values):
        """
        Insert intersection point between two lines in the array
        Args:
            arr (ndarray): array of points
            closest_idx (ndarray): index of the closest point
        Returns:
            ndarray: new array with intersection points
        """
        arr[closest_idx] = values
        return arr

    def insert_tensor_of_inertia(self, array, centroid) -> np.ndarray:
        """
        1. calculate centroid of the cortex
        2. calculate intersection of the centroid with the contours
        3. calculate nearest neighbor of the intersection point
        4. insert the intersection point into the contours
        """
        intersection_1 = self.shapely_line_polygon_intersection(
            array, self.partition_lines(50, centroid)[0]
        )
        intersection_2 = self.shapely_line_polygon_intersection(
            array, self.partition_lines(50, centroid)[1]
        )
        intersections = np.array([intersection_1, intersection_2])

        for _, intersection in enumerate(intersections):
            for i, inters in enumerate(intersection):
                print(f"intersection point {i}: {inters}")
                closest_idx = self.intersection_point(array, inters)
                array = self.insert_closest_point(array, closest_idx, inters)


if __name__ == "__main__":
    geo_file_path = r"/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/04_OUTPUT/C0002237/fake_example.geo_unrolled"
    mesh_file_path = r"/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/04_OUTPUT/C0002237/C0002237.msh"
    mesher = Mesher(geo_file_path, mesh_file_path)
    mesher.build_msh()
