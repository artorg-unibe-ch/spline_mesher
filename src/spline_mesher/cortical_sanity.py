import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import spatial
from pathlib import Path
import matplotlib.cm as cm
import operator
import shapely.geometry as shpg


# plt.style.use('02_CODE/src/spline_mesher/cfgdir/pos_monitor.mplstyle')  # https://github.com/matplotlib/matplotlib/issues/17978


class CorticalSanityCheck:
    def __init__(
        self, MIN_THICKNESS, ext_contour, int_contour, model, save_plot
    ) -> None:
        self.min_thickness = (
            MIN_THICKNESS  # minimum thickness between internal and external contour
        )
        self.ext_contour = ext_contour  # external contour
        self.int_contour = int_contour  # internal contour
        self.model = str(model)
        self.save_plot = bool(save_plot)

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

    def correct_intersection(
        self, ext_arr, int_arr, base_idx, dx, dy, bool_arr
    ):  # closest_idx,
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

        # bool_arr = np.c_[bool_arr, bool_arr]
        bool_arr = bool_arr[:-1]

        ext_arr = ext_arr[:-1]
        base_idx = base_idx[:-1]
        # int_arr = int_arr[:-1]
        # closest_idx = closest_idx[:-1]
        corr_arr = int_arr
        dx = dx[:-1]
        dy = dy[:-1]

        for idx_, bool_ in enumerate(bool_arr[:-1]):
            if bool_ == True:
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

        # int_corr = self.correct_internal_point(ext_arr, base_idx, dx, dy)
        # np.copyto(dst=int_arr, src=int_corr, where=bool_arr)
        # return int_arr

        # bool_arr = np.c_[bool_arr, bool_arr]
        # int_corr = self.correct_internal_point(ext_arr, pairs, dx, dy)
        # np.copyto(dst=int_arr[pairs[:, 1]], src=int_corr, where=bool_arr)
        # return int_arr[pairs[:, 1]]

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

    def get_normals(self, xs, ys, ax1, ax2, thickness=1):
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
            print(f"dx: {dx}, dy: {dy}")
            norm = math.hypot(dx, dy) * 1 / thickness
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

        return dx_arr, dy_arr, dx_med, dy_med

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

    def plot_contours(self, min_thickness, x_ext, y_ext, x_int, y_int):
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

        dx, dy, dx_med, dy_med = self.get_normals(
            x_ext, y_ext, ax1, ax2, thickness=min_thickness
        )

        ext_a = np.c_[x_ext, y_ext]
        int_a = np.c_[x_int, y_int]
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

        return ext_a, int_a, dx_med, dy_med, fig, ax1, ax2

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

    def offset_surface(self, ext_line, offset):
        """
        Create artificial internal surface based on an offset
        Don't implement in main script it's just for testing
        """
        # Create a Polygon from the 2d array
        poly = shpg.Polygon(ext_line)

        # Create offset in inward direction
        noffpoly = poly.buffer(offset)  # offset

        # Turn polygon points into numpy arrays for plotting
        afpolypts = np.array(poly.exterior)
        noffafpolypts = np.array(noffpoly.exterior)
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
            fig.savefig(filename_png, dpi=600)
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

    def push_contour(self, ext_contour, int_contour, offset):
        ext_offset = self.offset_surface(ext_contour, offset)
        resampling = int(500)
        ext_offset_hres = self.resample_contour(ext_offset, n_points=resampling)
        int_contour_hres = self.resample_contour(int_contour, n_points=resampling)

        is_inside = [
            shpg.Point(int_contour_hres[i]).within(shpg.Polygon(ext_offset_hres))
            for i in range(len(int_contour_hres))
        ]
        is_inside = np.c_[is_inside, is_inside]
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

    def cortical_sanity_check(self, ext_contour, int_contour, iterator):
        """
        Check if the internal contour is within the external contour.

        Args:
            ext_contour (ndarray): 2D array of [x, y] points
            int_contour (ndarray): 2D array of [x, y] points

        Returns:

        """
        ext_s, int_s, dx_med, dy_med, fig, ax1, ax2 = self.plot_contours(
            min_thickness=self.min_thickness,
            x_ext=ext_contour[:, 0],
            y_ext=ext_contour[:, 1],
            x_int=int_contour[:, 0],
            y_int=int_contour[:, 1],
        )

        new_int, ext_offset = self.push_contour(
            ext_contour, int_contour, -self.min_thickness
        )

        # dists_ = self.KD_2_Tree(ext_s, int_s, ext_contour_offset)
        # bool_kdtree = dists_ < float(0)

        # print(f"bool_kdtree: {bool_kdtree.sum()}")
        # Calculate nearest point pairs between the two contours
        # dist_nn, base_idx, closest_idx = self.nearest_pairs_arrs(
        #     ext_contour, int_contour
        # )
        # # Compute boolean array of points that are within the minimum thickness
        # bool_thickness_nn = dist_nn < self.min_thickness

        # print(f"Number of points within min thickness: {bool_thickness_nn.sum()}")
        # print(
        #     f"Is the Euclidean distance between the two contours < min thickness?\n{bool_thickness_nn}"
        # )
        # # Correct the internal contour
        # new_int = self.correct_intersection(
        #     ext_arr=ext_s,
        #     int_arr=int_s,
        #     base_idx=base_idx,
        #     dx=dx_med,
        #     dy=dy_med,
        #     bool_arr=bool_thickness_nn,
        # )

        # bool_thickness = self.hyp_min_thickness(
        #     ext_contour=ext_contour,
        #     int_contour=int_s,
        #     idx_=base_idx,
        # )

        # # print(f"Is the thickness requirement violated?\n{bool_thickness}")

        # bool_combined = np.logical_or(bool_thickness_nn, bool_thickness)

        # new_int = self.correct_intersection(
        #     ext_arr=ext_s,
        #     int_arr=int_s,
        #     base_idx=base_idx,
        #     dx=dx_med,
        #     dy=dy_med,
        #     bool_arr=bool_combined,
        # )

        # # Calculate nearest point pairs between the two contours
        # dist_nn, base_idx, _ = self.nearest_pairs_arrs(ext_contour, new_int)
        # # Compute boolean array of points that are within the minimum thickness
        # bool_thickness_nn = dist_nn < self.min_thickness

        # new_int = self.correct_intersection(
        #     ext_arr=ext_s,
        #     int_arr=new_int,
        #     base_idx=base_idx,
        #     dx=dx_med,
        #     dy=dy_med,
        #     bool_arr=bool_thickness_nn,
        # )

        """
        boolean_radius = self.is_internal_radius_bigger_than_external(
            ext_s, new_int, idx_ext, idx_int
        )
        print(f"Is internal radius bigger than external contour?\n{boolean_radius}")

        dx_med[-1] = dx_med[0]  # guarantees continuity / closes the loop
        dy_med[-1] = dy_med[0]  # guarantees continuity / closes the loop

        new_int = self.correct_intersection(
            ext_arr=ext_s,
            int_arr=new_int,
            pairs=np.c_[idx_ext, idx_int],
            dx=dx_med,
            dy=dy_med,
            bool_arr=boolean_radius,
        )
        

        boolean_angle = self.is_internal_inside_external(ext_s, int_s, idx_ext, idx_int)
        print(f"Is internal contour outside external contour?\n{boolean_angle}")

        new_int = self.correct_intersection(
            ext_arr=ext_s,
            int_arr=new_int,
            pairs=np.c_[idx_ext, idx_int],
            dx=dx_med,
            dy=dy_med,
            bool_arr=boolean_angle,
        )


        bool_min_thickness_s_0 = self.check_min_thickness(
            ext=ext_s,
            int=new_int,
            idx_ext=idx_ext,
            idx_int=idx_int,
            min_thickness=self.min_thickness,
        )
        print(f"Is minimal thickness respected?\n{bool_min_thickness_s_0}")

        bool_min_thickness_s_1 = self.check_min_thickness(
            ext=ext_s,
            int=np.roll(new_int, 1),
            idx_ext=idx_ext,
            idx_int=idx_int,
            min_thickness=self.min_thickness,
        )
        bool_min_thickness_s_m1 = self.check_min_thickness(
            ext=ext_s,
            int=np.roll(new_int, -1),
            idx_ext=idx_ext,
            idx_int=idx_int,
            min_thickness=self.min_thickness,
        )
        bool_min_thickness_s = np.logical_or(
            np.array(bool_min_thickness_s_0),
            np.array(bool_min_thickness_s_1),
            np.array(bool_min_thickness_s_m1),
        )

        bool_arr = np.c_[bool_min_thickness_s, bool_min_thickness_s]
        int_corr = np.array(
            [
                ext_s[idx_ext][:, 0] - dy_med[idx_ext],
                ext_s[idx_ext][:, 1] + dx_med[idx_ext],
            ]
        ).transpose()
        np.copyto(dst=new_int, src=int_corr, where=bool_arr)

        new_int = self.correct_intersection(
            ext_arr=ext_s,
            int_arr=new_int,
            pairs=np.c_[idx_ext, idx_int],
            dx=dx_med,
            dy=dy_med,
            bool_arr=bool_min_thickness_s,
        )
        """

        if self.save_plot is not False:
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
            )  # self.int_contour?
        else:
            print("Plotting of corrected contours is disabled")

        return new_int
