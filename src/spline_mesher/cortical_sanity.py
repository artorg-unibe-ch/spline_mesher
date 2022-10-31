import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import spatial
import sys
# plt.style.use('simone')

# TODO: find a way to include plt.style.use inside package
#       labels: enhancement
#       assignees: @simoneponcioni
#       milestone: v0.1.0


class CorticalSanityCheck:
    def __init__(self, MIN_THICKNESS, ext_contour, int_contour) -> None:
        self.min_thickness = MIN_THICKNESS  # minimum thickness between internal and external contour
        self.ext_contour = ext_contour  # external contour
        self.int_contour = int_contour  # internal contour

    def unit_vector(self, vector):
        """
        Returns the unit vector in numpy form

        Args:
            vector (numpy.ndarray): 2D numpy array, e.g. ([5, 5])

        Returns:
            list: comprehension list of 2D unit vectors (calculated "row-by-row"), e.g. [0.7, 0.7]
        """
        return [vector[0] / np.linalg.norm(vector, axis=-1), vector[1] / np.linalg.norm(vector, axis=-1)]

    def ccw_angle(self, array1, array2):
        """
        Returns the angle between two 2D arrays in the range [0, 2*pi)

        Args:
            array1 (numpy.ndarray): array of shape (2,) containing [x, y] coords
            array2 (numpy.ndarray): array of shape (2,) containing [x, y] coords

        Returns:
            numpy.ndarray: array of shape (1,) containing angles between array1 and array2
        """
        # Get the angle between the two arrays
        angle = np.arctan2(array2[:, 1], array2[:, 0]) - \
            np.arctan2(array1[:, 1], array1[:, 0])
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
            raise ValueError('value is nan, 0-angle vector is undefined')
        else:
            bool_angle = False
            raise RuntimeWarning('angle comparison is not possible and/or edge case')
        return bool_angle

    def is_internal_inside_external(self, p_e_0, p_i_0):
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
        p_i_1 = self.roll_index(p_i_0, idx=2)

        alpha_ext = self.ccw_angle(p_e_2 - p_e_1, p_e_0 - p_e_1)
        alpha_int = self.ccw_angle(p_e_2 - p_e_1, p_i_1 - p_e_1)
        boolean_angle = [self.is_angle_bigger_bool(
            alpha_int, alpha_ext) for alpha_int, alpha_ext in zip(alpha_int, alpha_ext)]
        return boolean_angle

    def check_min_thickness(self, ext, int, min_thickness=1):
        """
        Checks thickness of ext, int arrays and returns a list of booleans
        of form [True, ..., False] where True means the thickness if below tolerance.

        Arguments:
            ext (numpy.ndarray): array of [x, y] points of external polygon
            int (numpy.ndarray): array of [x, y] points of internal polygon
            min_thickness (float): minimum thickness tolerance between ext/int

        Returns:
            bool_min_thickness (list): list indicating where the thickness is below tolerance
        """
        dist_x = ext[:, 0] - int[:, 0]
        dist_y = ext[:, 1] - int[:, 1]
        dist = np.sqrt(dist_x**2 + dist_y**2)

        bool_min_thickness = [i < min_thickness for i in dist]
        return bool_min_thickness

    def correct_internal_point(self, arr, dx, dy):
        """
        Corrects [x, y] position of points of internal array

        Args:
            arr (ndarray): array of internal perimeter
            dx (float): normal component x * minimum thickness
            dy (float): normal component y * minimum thickness

        Returns:
            ndarray: new position of internal points in array 'arr'
        """
        return np.array([arr[:, 0] - dy, arr[:, 1] + dx]).transpose()

    def correct_intersection(self, ext_arr, int_arr, dx, dy, bool_arr):
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
        bool_arr = np.c_[bool_arr, bool_arr]
        int_corr = self.correct_internal_point(ext_arr, dx, dy)
        np.copyto(dst=int_arr, src=int_corr, where=bool_arr)
        return int_arr

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
        ax.arrow(arr_start[0], arr_start[1], dx, dy, head_width=0.01,
                 head_length=0.01, length_includes_head=True, color=color)
        ax.annotate(text, xy=(arr_end[0], arr_end[1]), xytext=(arr_end[0] - 25, arr_end[1] - 35), color=color,
                    xycoords='data', textcoords="offset points",
                    size=16, va="center")
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

            ax1.plot(((x0 + xa) / 2, (x0 + xa) / 2 - dy), ((y0 + ya) / 2, (y0 + ya) / 2 + dx), color='tab:grey')    # plot the normals
            self.draw_arrow(ax2, (x0, y0), (x0 - dy, y0 + dx),
                            text=' ', color='tab:grey')
            self.draw_arrow(ax2, (xa, ya), (xa - dy, ya + dx),
                            text=' ', color='tab:grey')

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
                ax2, (x0, y0), (x0 - dy_med[idx], y0 + dx_med[idx]), text='$\\vec{n}_{res}$', color='tab:green')

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

        dx, dy, dx_med, dy_med = self.get_normals(x_ext, y_ext, ax1, ax2, thickness=min_thickness)

        ext_a = np.c_[x_ext, y_ext]
        int_a = np.c_[x_int, y_int]
        print(f'Plotting ext_a: {ext_a}')
        print(f'Plotting int_a: {int_a}')

        ax1.plot(x_ext, y_ext, label='external contour')
        ax2.plot(x_ext, y_ext)
        ax1.plot(x_int, y_int, label='initial internal contour')
        ax2.plot(x_int, y_int)

        for i, txt in enumerate(range(len(x_ext))):
            ax1.annotate(txt, (x_ext[i], y_ext[i]))
            ax1.annotate(txt, (x_int[i], y_int[i]))
            ax2.annotate(txt, (x_ext[i], y_ext[i]))
            ax2.annotate(txt, (x_int[i], y_int[i]))

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

    def plot_corrected_contours(self, fig, ax1, ax2, ext_s, dx_med, dy_med, ext_spline, int_spline, loc_int, new_int, save=False):
        ax1.scatter(ext_spline[0, 0], ext_spline[0, 1], marker='x', s=300)
        ax1.scatter(int_spline[0, 0], int_spline[0, 1], marker='x', s=300)
        ax1.scatter(loc_int[0], loc_int[1], marker='x', s=300)
        ax1.plot(new_int[:, 0], new_int[:, 1], linestyle=':', label='corrected internal contour')
        ax2.plot(new_int[:, 0], new_int[:, 1], linestyle=':')
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax2.set_xlim(8, 17)
        ax2.set_ylim(10, 20)
        fig.suptitle('Correction of minimal thickness and external position ($\\vec{n}$-based)', weight='bold', fontsize=20)
        fig.legend(loc=1)
        fig.tight_layout()
        ax1.plot((ext_s[:, 0], ext_s[:, 0] - dy_med), (ext_s[:, 1], ext_s[:, 1] + dx_med), color='tab:blue')

        if save is not False:
            fig.savefig('04_OUTPUT/tmp/C0002231_CORT_Correction_int_ext_pos.svg')
            fig.savefig('04_OUTPUT/tmp/C0002231_CORT_Correction_int_ext_pos.png', dpi=150)
        else:
            pass

        # plt.show()
        return None

    def cortical_sanity_check(self, ext_contour, int_contour):
        """
        Check if the internal contour is within the external contour.

        Args:
            ext_contour (ndarray): 2D array of [x, y] points
            int_contour (ndarray): 2D array of [x, y] points

        Returns:

        """
        # TODO: Find better solution than resampling (or make sure it works)
        #       And make sure that the resampling is done in the same way for both contours
        #       (i.e. same number of points, same interpolation method, etc.)
        #       labels: enhancement
        #       assignees: @simoneponcioni
        #       milestone: v0.1.0

        int_contour = self.resample_contour(int_contour, n_points=len(ext_contour))
        ext_contour = self.resample_contour(ext_contour, n_points=len(ext_contour))

        loc_int, idx_int = self.nearest_point(int_contour, [ext_contour[0, 0], ext_contour[0, 1]])
        int_contour = np.roll(int_contour, -idx_int, axis=0)
        print(f'Internal contour was shifted by {idx_int} point(s)')
        # if 0 not in [idx_int] and int(len(ext_contour)-1) not in [idx_int]:
        #     # check that 1st and last elements are the same
        if not np.allclose(int_contour[0], int_contour[-1], atol=1e-6):
            print('External contour is not closed')
            _, idx_int_contour = np.unique(int_contour.round(decimals=6), return_index=True, axis=0)
            int_contour = int_contour[np.sort(idx_int_contour)]
            int_contour = np.append(int_contour, [int_contour[0]], axis=0)
            print('External contour was closed')
        else:
            print('External contour is closed')

        # else:
        #     # check if two lines of data do not appear after each other inside any position of int_contour:
        #     print(f'Internal contour was not shifted')

        ext_s, int_s, dx_med, dy_med, fig, ax1, ax2 = self.plot_contours(min_thickness=self.min_thickness,
                                                                         x_ext=ext_contour[:, 0],
                                                                         y_ext=ext_contour[:, 1],
                                                                         x_int=int_contour[:, 0],
                                                                         y_int=int_contour[:, 1])

        boolean_angle = self.is_internal_inside_external(ext_s, int_s)
        print(f'Is internal contour outside external contour?\n{boolean_angle}')

        dx_med[-1] = dx_med[0]  # guarantees continuity / closes the loop
        dy_med[-1] = dy_med[0]  # guarantees continuity / closes the loop

        new_int = self.correct_intersection(ext_arr=ext_s,
                                            int_arr=int_s,
                                            dx=dx_med,
                                            dy=dy_med,
                                            bool_arr=boolean_angle
                                            )

        bool_min_thickness_s = self.check_min_thickness(ext=ext_s, int=new_int, min_thickness=self.min_thickness)
        new_int = self.correct_intersection(ext_arr=ext_s,
                                            int_arr=new_int,
                                            dx=dx_med,
                                            dy=dy_med,
                                            bool_arr=bool_min_thickness_s
                                            )

        self.plot_corrected_contours(fig, ax1, ax2, ext_s, dx_med, dy_med, self.ext_contour, int_contour, loc_int, new_int, save=True)  # self.int_contour?
        return new_int
