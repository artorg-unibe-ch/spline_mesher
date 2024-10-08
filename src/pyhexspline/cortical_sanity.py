import math
from logging import Logger
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shpg
import shapely.ops as shpops
from numpy import ndarray
from scipy import spatial
from shapely.ops import unary_union

# plt.style.use('02_CODE/src/spline_mesher/cfgdir/pos_monitor.mplstyle')  # https://github.com/matplotlib/matplotlib/issues/17978

# flake8: noqa: E501

LOGGING_NAME = "MESHING"


class CorticalSanityCheck:
    def __init__(
        self,
        MIN_THICKNESS: float,
        ext_contour: ndarray,
        int_contour: ndarray,
        model: str,
        save_plot: bool,
        logger: Logger,
    ) -> None:
        self.min_thickness = (
            MIN_THICKNESS  # minimum thickness between internal and external contour
        )
        self.ext_contour = ext_contour  # external contour
        self.int_contour = int_contour  # internal contour
        self.model = str(model)
        self.save_plot = bool(save_plot)
        self.logger = logger

    def unit_vector(self, vector):
        """
        Returns the unit vector in numpy form

        Args:
            vector (numpy.ndarray): 2D numpy array, e.g. ([5, 5])

        Returns:
            list: comprehension list of 2D unit vectors (calculated "row-by-row"), e.g. [0.7, 0.7]
        """
        return [
            vector[0] / np.linalg.norm(vector, axis=-1),
            vector[1] / np.linalg.norm(vector, axis=-1),
        ]

    def ccw_angle(self, array1, array2, idx1, idx2):
        """
        Returns the angle between two 2D arrays in the range [0, 2*pi)

        Args:
            array1 (numpy.ndarray): array of shape (2,) containing [x, y] coords
            array2 (numpy.ndarray): array of shape (2,) containing [x, y] coords

        Returns:
            numpy.ndarray: array of shape (1,) containing angles between array1 and array2
        """
        # Get the angle between the two arrays
        angle = np.arctan2(array2[idx2][:, 1], array2[idx2][:, 0]) - np.arctan2(
            array1[idx1][:, 1], array1[idx1][:, 0]
        )
        # If the angle is negative, add 2*pi
        angle = np.where(angle < 0, angle + 2 * np.pi, angle)
        return angle

    def convertRadiansToDegrees(self, radians):
        """Converts radians to degrees"""
        return radians * 180 / np.pi

    def reset_numpy_index(self, arr, idx=1):
        """
        Reset the index of a numpy array to a given index
        Legacy function, use numpy_roll_index() instead
        Args:
            arr (numpy.ndarray): entry array
            idx (int): index to reset to. Defaults to 1.

        Returns:
            numpy.ndarray: numpy array with index reset to idx
        """

        return np.r_[arr[idx:, :], arr[:idx, :]]

    def roll_index(self, arr, idx=1):
        """
        Roll the index of a numpy array to a given index.
        A little faster than reset_numpy_index()

        Args:
            arr (numpy.ndarray): entry array
            idx (int): index to roll to. Defaults to 1.

        Returns:
            numpy.ndarray: numpy array with index rolled to idx
        """
        return np.roll(arr, -idx, axis=0)

    def is_angle_bigger_bool(self, alpha_int, alpha_ext):
        """
        Returns True if alpha_int is bigger than alpha_ext, False otherwise
        If alpha_int is bigger than alpha_ext, then the internal angle is bigger than the external angle
        If np.nan, then return False and print a warning
        Else, return False

        Args:
            alpha_int (numpy.ndarray): array of shape (1,) containing internal angles
            alpha_ext (numpy.ndarray): array of shape (1,) containing external angles

        Raises:
            ValueError: print a warning if alpha_int or alpha_ext is np.nan (0-degree angle)
            RuntimeWarning: error if comparison is not possible

        Returns:
            numpy.ndarray: array containing True if alpha_int is bigger than alpha_ext, False otherwise
        """
        if alpha_int >= alpha_ext:
            bool_angle = True
        elif alpha_int < alpha_ext:
            bool_angle = False
        elif alpha_ext == np.nan() or alpha_int == np.nan():
            bool_angle = False
            raise ValueError("value is nan, 0-angle vector is undefined")
        else:
            bool_angle = False
            raise RuntimeWarning("angle comparison is not possible and/or edge case")
        return bool_angle

    def center_of_mass(self, array: np.ndarray):
        """
        Calculate the center of mass of a 2D array.

        This function computes the center of mass (centroid) of a given 2D array.
        The center of mass is calculated based on the sum of the array values and
        their respective coordinates.

        Args:
            array (np.ndarray): A 2D NumPy array representing the data.

        Returns:
            Tuple[float, float]: The x and y coordinates of the center of mass.
        """
        total = array.sum()
        x_coord = (array.sum(axis=1) @ range(array.shape[0])) / total
        y_coord = (array.sum(axis=0) @ range(array.shape[1])) / total
        return x_coord, y_coord

    def is_internal_radius_bigger_than_external(self, p_e_0, p_i_0, idx_ext, idx_int):
        """
        Returns True if the internal radius is bigger than the external radius, False otherwise

        Args:
            p_e_0 (numpy.ndarray): array of shape (2,) containing external radius
            p_i_0 (numpy.ndarray): array of shape (2,) containing internal radius

        Returns:
            bool_radius: True if internal radius is bigger than external radius, False otherwise
        """
        # check if internal radius is bigger than external radius element-wise and return boolean element-wise
        p_e_0_x = np.mean(p_e_0[:, 0])
        p_e_0_y = np.mean(p_e_0[:, 1])
        r_e = np.sqrt(
            (p_e_0[idx_ext][:, 0] - p_e_0_x) ** 2
            + (p_e_0[idx_ext][:, 1] - p_e_0_y) ** 2
        )
        r_i = np.sqrt(
            (p_i_0[idx_int][:, 0] - p_e_0_x) ** 2
            + (p_i_0[idx_int][:, 1] - p_e_0_y) ** 2
        )
        bool_radius = [r_e < r_i for (r_e, r_i) in zip(r_e, r_i)]
        return bool_radius

    def is_internal_inside_external(self, p_e_0, p_i_0, idx_ext, idx_int):
        """
        Rationale for deciding if internal contour is inside external contour based on
        the angle between the first and second point of the external contour, and the first
        point and the first point of the internal contour.

        Args:
            p_e_0 (numpy.ndarray): external contour
            p_i_0 (numpy.ndarray): internal contour

        Returns:
            list: Booleans where True if internal contour is OUTSIDE external contour, False otherwise

        pseudo-code:
        Pe: Point of external perimeter
        Pi: Point of internal perimeter
        P0: First point
        P1: Pivot point
        P2: Second point
        # Compute vectors between Pe0-Pe1 and Pe1-Pe2
        # v1_e_u = ext[i+1] - ext[i]
        # v2_e_u = ext[i+2] - ext[i+1]

        # Compute angles between vectors
        # alpha_e = angle_between(v1_e_u, v2_e_u)
        # alpha_i = angle_between(v1_e_u, v2_i_u)
        """
        p_e_1 = self.roll_index(p_e_0, idx=1)
        p_e_2 = self.roll_index(p_e_0, idx=2)
        # p_i_1 = self.roll_index(p_i_0, idx=1)

        alpha_ext = self.ccw_angle(p_e_2 - p_e_1, p_e_0 - p_e_1, idx_ext, idx_ext)
        alpha_int = self.ccw_angle(
            p_e_2 - p_e_1, p_i_0[idx_int] - p_e_1, idx_ext, idx_int
        )
        boolean_angle = [
            self.is_angle_bigger_bool(alpha_int, alpha_ext)
            for alpha_int, alpha_ext in zip(alpha_int, alpha_ext)
        ]
        return boolean_angle

    def check_min_thickness(self, ext, int, idx_ext, idx_int, min_thickness=1):
        """
        # ! delete this function if not used
        Checks thickness of ext, int arrays and returns a list of booleans
        of form [True, ..., False] where True means the thickness if below tolerance.

        Arguments:
            ext (numpy.ndarray): array of [x, y] points of external polygon
            int (numpy.ndarray): array of [x, y] points of internal polygon
            min_thickness (float): minimum thickness tolerance between ext/int

        Returns:
            bool_min_thickness (list): list indicating where the thickness is below tolerance
        """
        dist_x = ext[idx_ext][:, 0] - int[idx_int][:, 0]
        dist_y = ext[idx_ext][:, 1] - int[idx_int][:, 1]
        dist = np.sqrt(dist_x**2 + dist_y**2)

        bool_min_thickness = [i < min_thickness for i in dist]
        return bool_min_thickness

    def correct_internal_point(self, arr, base_idx, dx, dy):
        """
        Corrects [x, y] position of points of internal array

        Args:
            arr (ndarray): array of internal perimeter
            dx (float): normal component x * minimum thickness
            dy (float): normal component y * minimum thickness

        Returns:
            ndarray: new position of internal points in array 'arr'
        """
        return np.array(
            [arr[base_idx[:, 0]][:, 0] - dy, arr[base_idx[:, 0]][:, 1] + dx]
        ).transpose()

    def correct_intersection(self, ext_arr, int_arr, base_idx, dx, dy, bool_arr):
        """
        Takes ext and int arrays and applies nodal displacement to the elements where bool_min_thickness == True

        Args:
            ext_arr (ndarray): array of [x, y] external points composing the perimeter
            int_arr (ndarray): array of [x, y] internal points composing the perimeter
            dx (float): normal component x * minimum thickness
            dy (float): normal component y * minimum thickness
            bool_min_thickness (list): list indicating where the thickness is below tolerance

        Returns:
            ndarray: new array of internal polygon
        """
        bool_arr = bool_arr[:-1]

        ext_arr = ext_arr[:-1]
        base_idx = base_idx[:-1]
        corr_arr = int_arr
        dx = dx[:-1]
        dy = dy[:-1]

        for idx_, bool_ in enumerate(bool_arr[:-1]):
            if bool_ is True:
                corr_arr[idx_] = (
                    ext_arr[idx_][0] - dy[idx_],
                    ext_arr[idx_][1] + dx[idx_],
                )
            else:
                # find closest point
                closest_idx = self.KDT_nn(ext_arr[idx_], int_arr)
                ext_intp1_hyp = np.linalg.norm(ext_arr[idx_] - int_arr[closest_idx + 1])
                ext_intm1_hyp = np.linalg.norm(ext_arr[idx_] - int_arr[closest_idx - 1])
                if ext_intp1_hyp < ext_intm1_hyp:
                    corr_arr[idx_] = self.project_point_on_line(
                        ext_arr[idx_], int_arr[idx_], int_arr[idx_ + 1]
                    )
                elif ext_intm1_hyp > ext_intp1_hyp:
                    corr_arr[idx_] = self.project_point_on_line(
                        ext_arr[idx_], int_arr[idx_ - 1], int_arr[idx_]
                    )

        corr_arr = np.append(corr_arr, corr_arr[0].reshape(1, 2), axis=0)
        return corr_arr

    def draw_arrow(self, ax, arr_start, arr_end, text, color):
        """
        Helper that draws normals with arrow and vector annotation

        Args:
            ax (AxesSubplot): ax where to add the arrow
            arr_start (tuple): tuple of arrow starting point
            arr_end (tuple): tuple of arrow ending point
            text (str): string of text to add as annotation
            color (str): color of arrow + annotation
        """
        dx = arr_end[0] - arr_start[0]
        dy = arr_end[1] - arr_start[1]
        ax.arrow(
            arr_start[0],
            arr_start[1],
            dx,
            dy,
            head_width=0.01,
            head_length=0.01,
            length_includes_head=True,
            color=color,
        )
        ax.annotate(
            text,
            xy=(arr_end[0], arr_end[1]),
            xytext=(arr_end[0] - 25, arr_end[1] - 35),
            color=color,
            xycoords="data",
            textcoords="offset points",
            size=16,
            va="center",
        )
        return None

    def get_normals(self, xs, ys, ax1, ax2, thickness=1, debug=False):
        """
        Get normal arrays point-wise for array [xs, ys]

        Args:
            xs (ndarray): x-component of contour array
            ys (ndarray): y-component of contour array
            ax1 (AxesSubplot): ax subplot 1
            ax2 (AxesSubplot): ax subplot 2
            thickness (int, optional): Minimum thickness of int-ext interface. Defaults to 1.

        Returns:
            dx_arr (ndarray): array of dx component of the normals
            dy_arr (ndarray): array of dy component of the normals
            dx_med (ndarray): array of resultant normal between (x_n and x_n+1)
            dy_med (ndarray): array of resultant normal between (y_n and y_n+1)
        """
        dx_arr = np.zeros((len(xs)) - 1)  # appending dx_arr[0] to the end
        dy_arr = np.zeros((len(ys)) - 1)  # appending dy_arr[0] to the end
        for idx in range(len(xs) - 1):
            x0, y0, xa, ya = xs[idx], ys[idx], xs[idx + 1], ys[idx + 1]
            dx, dy = xa - x0, ya - y0
            if debug is True:
                print(f"dx: {dx}, dy: {dy}")
            norm = math.hypot(dx, dy) * 1 / thickness
            if debug is True:
                print(f"norm: {norm}")
            dx /= norm
            dy /= norm

            dx_arr[idx] = dx
            dy_arr[idx] = dy

            ax1.plot(
                ((x0 + xa) / 2, (x0 + xa) / 2 - dy),
                ((y0 + ya) / 2, (y0 + ya) / 2 + dx),
                color="tab:grey",
            )  # plot the normals
            self.draw_arrow(
                ax2, (x0, y0), (x0 - dy, y0 + dx), text=" ", color="tab:grey"
            )
            self.draw_arrow(
                ax2, (xa, ya), (xa - dy, ya + dx), text=" ", color="tab:grey"
            )

        dx_arr = np.insert(dx_arr, 0, dx_arr[-1])
        dy_arr = np.insert(dy_arr, 0, dy_arr[-1])
        dx_arr = np.append(dx_arr, dx_arr[0])
        dy_arr = np.append(dy_arr, dy_arr[0])

        dx_med = np.zeros(len(dx_arr) - 1)
        for dx in range(len(dx_arr) - 1):
            dx_med_s = (dx_arr[dx] + dx_arr[dx + 1]) * 0.5
            dx_med[dx] = dx_med_s

        dy_med = np.zeros(len(dy_arr) - 1)
        for dy in range(len(dy_arr) - 1):
            dy_med_s = (dy_arr[dy] + dy_arr[dy + 1]) * 0.5
            dy_med[dy] = dy_med_s

        for idx in range(len(xs) - 1):
            x0, y0, xa, ya = xs[idx], ys[idx], xs[idx + 1], ys[idx + 1]
            self.draw_arrow(
                ax2,
                (x0, y0),
                (x0 - dy_med[idx], y0 + dx_med[idx]),
                text="$\\vec{n}_{res}$",
                color="tab:green",
            )

        return dx_med, dy_med

    def nearest_point(self, arr, pt):
        """
        Find nearest point between an array and a point (e.g. the first point of ext contour)

        Args:
            arr (ndarray): 2D array of contour
            pt (list): (x, y) coordinates of point

        Returns:
            loc (numpy.ndarray): location of nearest point in the array
            dist (float): distance between point and nearest point in array
            idx (numpy.int64): index of point in array
        """
        _, idx = spatial.KDTree(arr).query(pt)
        loc = arr[spatial.KDTree(arr).query(pt)[1]]
        return loc, idx

    def nearest_pairs_arrs(self, arr1, arr2):
        """
        Find nearest point between two arrays and create a an array containing nearest-pairs indices

        Args:
            arr1 (np.ndarray): array of contour 1 (e.g. ext)
            arr2 (np.ndarray): array of contour 2 (e.g. int)

        Returns:
            closest_idx (np.ndarray): array of indices of nearest index in arr2 for each point in arr1
            base_idx (np.ndarray): array of ordered indices of arr1
        """
        dist_nearest_neighbour = np.empty(len(arr1), dtype=float)
        closest_idx = np.empty((len(arr1), 2), dtype=int)
        base_idx = np.empty((len(arr1), 2), dtype=int)
        for i, idx in enumerate(arr1):
            d_nn, closest_idx_s = spatial.KDTree(arr2).query(idx)
            dist_nearest_neighbour[i] = d_nn
            closest_idx[i] = closest_idx_s
            base_idx[i] = i
        print(
            f"dist_nearest_neighbour\tbase_idx\tclosest_idx\n{dist_nearest_neighbour}\t{base_idx}\t{closest_idx}"
        )
        return dist_nearest_neighbour, base_idx, closest_idx

    def min_thickness_nearest_pairs(self, neighbour_dists):
        if np.where(neighbour_dists < self.min_thickness)[0]:
            bool_min_thickness = True
        else:
            bool_min_thickness = False
        return bool_min_thickness

    def lineseg_dists(self, p, a, b):
        """
        https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line
        Cartesian distance from point to line segment

        Edited to support arguments as series, from:
        https://stackoverflow.com/a/54442561/11208892

        Args:
            - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
            - a: np.array of shape (x, 2)
            - b: np.array of shape (x, 2)
        """
        # normalized tangent vectors
        d_ba = b - a
        d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))

        # signed parallel distance components
        # rowwise dot products of 2D vectors
        s = np.multiply(a - p, d).sum(axis=1)
        t = np.multiply(p - b, d).sum(axis=1)

        # clamped parallel distance
        h = np.maximum.reduce([s, t, np.zeros(len(s))])

        # perpendicular distance component
        # rowwise cross products of 2D vectors
        d_pa = p - a
        c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
        return np.hypot(h, c)

    def hyp_min_thickness(self, ext_contour, int_contour, idx_):
        """
        Check if the thickness between external and internal contours is below a minimum threshold.

        This function evaluates the thickness between corresponding points on the external
        and internal contours. It checks if the thickness at any point is below a specified
        minimum threshold and returns a boolean array indicating where this condition is met.

        Args:
            ext_contour (ndarray): The external contour points.
            int_contour (ndarray): The internal contour points.
            idx_ (ndarray): Indices for the internal contour points.

        Returns:
            ndarray: A boolean array indicating where the thickness is below the minimum threshold.
        """
        ext_contour = ext_contour[:-1]
        int_contour = int_contour[:-1]
        idx_ = idx_[:-1]
        roll_int_array = np.roll(int_contour, -1, axis=0)
        bool_thickness = np.full(len(ext_contour), False)
        for idx, p in enumerate(ext_contour[:, 0]):
            dists = self.lineseg_dists(
                p,
                np.take(int_contour, idx_),
                np.take(roll_int_array, idx_),
            )
            bool_ = np.where(dists < self.min_thickness, True, False)
            bool_thickness = np.logical_or(bool_thickness, bool_)

        bool_thickness = np.append(bool_thickness, bool_thickness[0])
        return bool_thickness

    def plot_contours(
        self,
        min_thickness: float,
        x_ext: np.ndarray,
        y_ext: np.ndarray,
        x_int: np.ndarray,
        y_int: np.ndarray,
        debug: bool = False,
    ):
        """
        Create contours, get normals, plot

        Args:
            min_thickness (int): Minimum thickness of int-ext interface.
            x_ext (ndarray): array of [x] component of external array
            y_ext (ndarray): array of [y] component of external array
            x_int (ndarray): array of [x] component of internal array
            y_int (ndarray): array of [y] component of internal array

        Returns:
            ext_a (ndarray): 2D array of [x, y] points
            int_a (ndarray): 2D array of [x, y] points
            dx_arr (ndarray): array of dx component of the normals
            dy_arr (ndarray): array of dy component of the normals
            dx_med (ndarray): array of resultant normal between (x_n and x_n+1)
            dy_med (ndarray): array of resultant normal between (y_n and y_n+1)
            fig (matplotlib.figure.Figure): figure
            ax1 (AxesSubplot): ax subplot
            ax2 (AxesSubplot): ax subplot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        dx_med, dy_med = self.get_normals(
            x_ext, y_ext, ax1, ax2, thickness=min_thickness, debug=False
        )

        ext_a = np.c_[x_ext, y_ext]
        int_a = np.c_[x_int, y_int]
        if debug:
            print(f"Plotting ext_a: {ext_a}")
            print(f"Plotting int_a: {int_a}")

        ax1.scatter(x_ext, y_ext, label="external contour", s=5)
        ax2.scatter(x_ext, y_ext)
        ax1.scatter(x_int, y_int, label="initial internal contour", s=5)
        ax2.scatter(x_int, y_int)

        for i, txt in enumerate(range(len(x_ext))):
            ax1.annotate(txt, (x_ext[i], y_ext[i]), color="tab:blue")
            ax1.annotate(txt, (x_int[i], y_int[i]), color="tab:orange")
            ax2.annotate(txt, (x_ext[i], y_ext[i]), color="tab:blue")
            ax2.annotate(txt, (x_int[i], y_int[i]), color="tab:orange")

        return ext_a, dx_med, dy_med, fig, ax1, ax2

    def resample_contour(self, xy: np.ndarray, n_points: int = 100) -> np.ndarray:
        """
        Cumulative Euclidean distance between successive polygon points.

        Args:
            xy (np.ndarray): 2D-array of [x, y] contour
            n_points (int, optional): Number of points to resample contour. Usually len(ext contour). Defaults to 100.

        Returns:
            np.ndarray: Resampled contour [x, y]
        """

        d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

        # get linearly spaced points along the cumulative Euclidean distance
        d_sampled = np.linspace(0, d.max(), n_points)

        # interpolate x and y coordinates
        xy_interp = np.c_[
            np.interp(d_sampled, d, xy[:, 0]),
            np.interp(d_sampled, d, xy[:, 1]),
        ]
        return xy_interp

    def offset_surface(self, line: ndarray, offset: float) -> ndarray:
        """
        Create artificial internal surface based on an offset
        Args:
            np.ndarray: 2D array of [x, y] points of the external surface
            offset (float): offset distance
        Returns:
            np.ndarray: 2D array of [x, y] points of the offset surface
        """
        poly = None
        noffpoly = None
        noffafpolypts = None
        noffpoly_union = None

        # Create a Polygon from the 2d array
        poly = shpg.Polygon(line)

        # Create offset in inward direction
        noffpoly = poly.buffer(offset)  # offset

        # If the result is a MultiPolygon, merge them into a single Polygon
        if isinstance(noffpoly, shpg.MultiPolygon):
            noffpoly = unary_union(noffpoly)

        # Turn polygon points into numpy arrays for plotting
        try:
            noffafpolypts = np.array(noffpoly.exterior.coords)
        except AttributeError:
            self.logger.warning(
                "Multipolygon was found, taking the convex hull as the offset surface"
            )
            noffpoly_union = shpops.unary_union(noffpoly)
            noffafpolypts = np.array(noffpoly_union.convex_hull.exterior.coords)
        return noffafpolypts

    def plot_corrected_contours(
        self,
        fig,
        ax1,
        ax2,
        ext_s,
        dx_med,
        dy_med,
        ext_spline,
        ext_offset,
        int_spline,
        new_int,
        iterator,
        save=False,
    ):
        """
        Plot and optionally save corrected contours for cortical thickness analysis.

        This function plots the corrected internal and external contours for cortical
        thickness analysis. It also provides an option to save the plots and the
        corresponding data to files.

        Args:
            fig (matplotlib.figure.Figure): The figure object for the plot.
            ax1 (matplotlib.axes.Axes): The first axes object for the plot.
            ax2 (matplotlib.axes.Axes): The second axes object for the plot.
            ext_s (ndarray): The external contour points.
            dx_med (float): The x-offset for the external contour.
            dy_med (float): The y-offset for the external contour.
            ext_spline (ndarray): The external spline contour points.
            ext_offset (ndarray): The offset external contour points.
            int_spline (ndarray): The internal spline contour points.
            new_int (ndarray): The corrected internal contour points.
            iterator (int): The current iteration index.
            save (bool, optional): Whether to save the plot and data to files. Default is False.

        Returns:
            None
        """
        ax1.scatter(ext_spline[0, 0], ext_spline[0, 1], marker="x", s=300)
        ax1.scatter(int_spline[0, 0], int_spline[0, 1], marker="x", s=300)
        ax1.plot(
            new_int[:, 0],
            new_int[:, 1],
            linestyle=":",
            linewidth=0.5,
            label="corrected internal contour",
            color="tab:green",
        )

        ax1.plot(
            ext_offset[:, 0],
            ext_offset[:, 1],
            label="offset external contour",
            linewidth=0.5,
            color="tab:purple",
        )

        ax1.scatter(
            new_int[:, 0],
            new_int[:, 1],
            color="tab:green",
            s=5,
            marker=".",
        )
        ax2.plot(new_int[:, 0], new_int[:, 1], linestyle=":")
        ax1.set_aspect("equal")
        ax2.set_aspect("equal")
        ax2.set_xlim(8, 15)
        ax2.set_ylim(1, 8)
        fig.suptitle(
            "Correction of minimal thickness and external position ($\\vec{n}$-based)",
            weight="bold",
            fontsize=20,
        )
        fig.legend(loc=1)
        fig.tight_layout()
        ax1.plot(
            (ext_s[:, 0], ext_s[:, 0] - dy_med),
            (ext_s[:, 1], ext_s[:, 1] + dx_med),
            color="tab:blue",
        )

        if save is not False:
            parentpath = Path(self.model).parent
            filepath = Path(self.model).stem
            savepath = Path(parentpath, filepath)
            filename_png = f"{savepath}_CORT_Correction_int_ext_pos_{iterator}.png"
            filename_npy_ext = f"{savepath}_CORT_ext_{iterator}.npy"
            filename_npy_int = f"{savepath}_CORT_int_{iterator}.npy"
            fig.savefig(filename_png, dpi=150)
            np.save(filename_npy_ext, ext_spline)
            np.save(filename_npy_int, int_spline)
        else:
            pass

        return None

    def project_point_on_line(self, p, a, b):
        """
        Project point p on line a-b
        """
        ap = p - a
        ab = b - a
        ab2 = np.dot(ab, ab)
        ap_ab = np.dot(ap, ab)
        t = ap_ab / ab2
        return a + ab * t

    def KDT_nn(self, p, arr):
        """
        Find nearest point between two arrays and create an array containing nearest-pairs indices

        Args:
            arr1 (np.ndarray): array of contour 1
            arr2 (np.ndarray): array of contour 2

        Returns:
            _type_: _description_
        """
        _, p_nn = spatial.KDTree(arr).query(p)
        return p_nn

    def push_contour(
        self, ext_contour: np.ndarray, int_contour: np.ndarray, offset: float
    ) -> Tuple[ndarray, ndarray]:
        """
        Adjust the internal contour to ensure it is within the offset external contour.

        This function adjusts the internal contour points to ensure they are within the
        offset external contour. It resamples both contours to a higher resolution,
        checks if the internal points are within the external contour, and if not,
        moves them to the closest points on the external contour.

        Args:
            ext_contour (np.ndarray): The external contour points.
            int_contour (np.ndarray): The internal contour points.
            offset (float): The offset distance for the external contour.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The adjusted internal contour and the offset external contour.
        """
        is_inside_shpg = []  # initialise

        RESAMPLING = int(1500)
        ext_offset = self.offset_surface(ext_contour, offset)
        ext_offset_hres = self.resample_contour(ext_offset, n_points=RESAMPLING)
        int_contour_hres = self.resample_contour(int_contour, n_points=RESAMPLING)

        is_inside_shpg = [
            shpg.Point(int_contour_hres[i]).within(shpg.Polygon(ext_offset_hres))
            for i in range(len(int_contour_hres))
        ]
        is_inside = np.c_[is_inside_shpg, is_inside_shpg]
        closest_points = [
            ext_offset_hres[
                spatial.KDTree(ext_offset_hres).query(int_contour_hres[i])[1]
            ]
            for i in range(len(int_contour_hres))
        ]
        closest_points = np.array(closest_points).reshape(-1, 2)
        np.copyto(
            dst=int_contour_hres, src=closest_points, where=np.logical_not(is_inside)
        )
        int_contour = self.resample_contour(int_contour_hres, n_points=len(int_contour))
        return int_contour, ext_offset

    def cortical_sanity_check(
        self,
        ext_contour: ndarray,
        int_contour: ndarray,
        iterator: int,
        show_plots: bool = True,
    ) -> ndarray:
        """
        Check if the internal contour is within the external contour.

        Args:
            ext_contour (ndarray): 2D array of [x, y] points
            int_contour (ndarray): 2D array of [x, y] points

        Returns:

        """
        if show_plots is True:
            ext_s, dx_med, dy_med, fig, ax1, ax2 = self.plot_contours(
                min_thickness=self.min_thickness,
                x_ext=ext_contour[:, 0],
                y_ext=ext_contour[:, 1],
                x_int=int_contour[:, 0],
                y_int=int_contour[:, 1],
                debug=False,
            )

        new_int, ext_offset = self.push_contour(
            ext_contour, int_contour, -self.min_thickness
        )

        if self.save_plot is True:
            self.plot_corrected_contours(
                fig,
                ax1,
                ax2,
                ext_s,
                dx_med,
                dy_med,
                self.ext_contour,
                ext_offset,
                int_contour,
                new_int,
                iterator,
                save=True,
            )
        else:
            pass
        return new_int
