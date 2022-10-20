import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import spatial


class CorticalSanity:
    def __init__(self, c, N_POINTS, MIN_THICKNESS) -> None:
        self.c = c
        self.n_points = N_POINTS
        self.min_thickness = MIN_THICKNESS

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
        angle = np.where(angle < 0, angle + 2*np.pi, angle)
        return angle

    def convertRadiansToDegrees(self, radians):
        """Converts radians to degrees"""
        return radians * 180 / np.pi

    def reset_numpy_index(self, arr, idx=1):
        """
        Reset the index of a numpy array to a given index
        Args:
            arr (numpy.ndarray): _description_
            idx (int): index to reset to. Defaults to 1.

        Returns:
            numpy.ndarray: numpy array with index reset to idx
        """
        return np.r_[arr[idx:, :], arr[:idx, :]]

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
            raise RuntimeWarning()
        return bool_angle

    def is_internal_inside_external(self, ext_c, int_c):
        '''
        # pseudo-code:
        # Compute unit vectors between Pe0-Pe1 and Pe1-Pe2
        # v1_e_u = unit_vector(ext[i+1]-ext[i])
        # v2_e_u = unit_vector(ext[i+2]-ext[i+1])

        # Compute unit vector between Pe1-Pe2 and Pe1-Pi_1  #TODO: careful, always CCW angle!
        # v1_e_u = unit_vector(ext[i+1] - ext[i])  #TODO: already computed, just for reference
        # v2_i_u = unit_vector(int[i] - ext[i+1])

        # Compute angles between vectors
        # alpha_e = angle_between(v1_e_u, v2_e_u)
        # alpha_i = angle_between(v1_e_u, v2_i_u)


        # TESTS
        n_points = 50
        angles = np.linspace(0 * np.pi, 2 * np.pi, n_points)
        v1 = np.full((n_points, 2), [1, 0])
        x = np.cos(angles)
        y = np.sin(angles)
        v2 = np.c_[x, y]
        angles = ccw_angle(v1, v2)
        '''
        p_e_0 = ext_c
        p_i_0 = int_c
        p_e_1 = self.reset_numpy_index(p_e_0, idx=1)
        p_e_2 = self.reset_numpy_index(p_e_0, idx=2)
        p_i_1 = self.reset_numpy_index(p_i_0, idx=1)

        alpha_ext = self.ccw_angle(p_e_2 - p_e_1, p_e_0 - p_e_1)
        alpha_int = self.self.ccw_angle(p_e_2 - p_e_1, p_i_1 - p_e_1)
        boolean_angle = [self.is_angle_bigger_bool(
            alpha_int, alpha_ext) for alpha_int, alpha_ext in zip(alpha_int, alpha_ext)]
        return boolean_angle

    def check_intersection(self, ext, int, min_thickness=1):
        """
        Checks thickness of ext, int arrays and returns a list of booleans
        of form [True, ..., False] where True means the thickness if below tolerance.

        Arguments:
            ext {ndarray} -- array of [x, y] points of external polygon
            int {ndarray} -- array of [x, y] points of internal polygon
            min_thickness {float} -- minimum thickness tolerance between ext/int

        Returns:
            bool_min_thickness {list}  -- list indicating where the thickness is below tolerance
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
        ax.annotate(text, xy=(arr_end[0], arr_end[1]), xytext=(arr_end[0]-25, arr_end[1]-35), color=color,
                    xycoords='data', textcoords="offset points",
                    size=16, va="center")
        return None

    def get_normals(self, xs, ys, ax1, ax2, thickness=1):
        """
        Get normal arrays point-wise for array [xs, ys]

        # TODO: needs some optimization (array operations)


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
        dx_arr = []
        dy_arr = []
        for idx in range(len(xs)-1):
            x0, y0, xa, ya = xs[idx], ys[idx], xs[idx+1], ys[idx+1]
            dx, dy = xa-x0, ya-y0
            norm = math.hypot(dx, dy) * 1/thickness
            dx /= norm
            dy /= norm

            dx_arr = np.append(dx_arr, dx)
            dy_arr = np.append(dy_arr, dy)
            ax1.plot(((x0+xa)/2, (x0+xa)/2-dy), ((y0+ya)/2, (y0+ya) /
                     2+dx), color='tab:grey')    # plot the normals
            self.draw_arrow(ax2, (x0, y0), (x0-dy, y0+dx),
                            text=' ', color='tab:grey')
            self.draw_arrow(ax2, (xa, ya), (xa-dy, ya+dx),
                            text=' ', color='tab:grey')

        dx_arr = np.append(dx_arr, dx_arr[0])
        dy_arr = np.append(dy_arr, dy_arr[0])

        dx_med = []
        for dx in range(len(dx_arr)-1):
            dx_med_s = (dx_arr[dx] + dx_arr[dx+1]) * 0.5
            dx_med = np.append(dx_med, dx_med_s)

        dy_med = []
        for dy in range(len(dy_arr)-1):
            dy_med_s = (dy_arr[dy] + dy_arr[dy+1]) * 0.5
            dy_med = np.append(dy_med, dy_med_s)

        dx_med = np.insert(dx_med, 0, dx_med[0])
        dy_med = np.insert(dy_med, 0, dy_med[0])

        for idx in range(len(xs)-1):
            x0, y0, xa, ya = xs[idx], ys[idx], xs[idx+1], ys[idx+1]
            self.draw_arrow(
                ax2, (x0, y0), (x0-dy_med[idx], y0+dx_med[idx]), text='$\\vec{n}_{res}$', color='tab:green')

        return dx_arr, dy_arr, dx_med, dy_med

    def get_circle(self, radius, center, n_points=15):
        """
        Get a circle to create synthetic test data

        Args:
            radius (float): radius of the circle
            center (list): center of the circle
            n_points (int): number of points that compose the circle. Defaults to 15.

        Returns:
            x_s (ndarray): array containing x position of circle
            y_s (ndarray): array containing y position of circle
        """
        angles = np.linspace(0 * np.pi, 2 * np.pi, n_points)
        xs = radius * np.cos(angles) + center[0]
        ys = radius * np.sin(angles) + center[1]
        return xs, ys

    def show_circles(self, center, n_points=15, min_thickness=1):
        """
        Create circles, get normals, plot

        Args:
            center (list): center of the circle
            n_points (int): number of points that compose the circle. Defaults to 15.
            min_thickness (int, optional): Minimum thickness of int-ext interface. Defaults to 1.

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
        ax1.plot(center[0], center[1], color='red', marker='o', label='center')
        ax1.annotate(f'O {center}', xy=(
            center[0] + 1, center[1] - 1), xycoords='data')
        x_ext, y_ext = self.get_circle(
            radius=15, center=center, n_points=n_points)
        x_int, y_int = self.get_circle(
            radius=15.1, center=center, n_points=n_points)

        dx, dy, dx_med, dy_med = self.get_normals(
            x_ext, y_ext, ax1, ax2, thickness=min_thickness)

        ext_a = np.c_[x_ext, y_ext]
        int_a = np.c_[x_int, y_int]

        ax1.plot(x_ext, y_ext, label='external contour')
        ax2.plot(x_ext, y_ext)
        ax1.plot(x_int, y_int, label='initial internal contour')
        ax2.plot(x_int, y_int)
        return ext_a, int_a, dx, dy, dx_med, dy_med, fig, ax1, ax2

    def nearest_point(arr, pt):
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
        dist, idx = spatial.KDTree(arr).query(pt)
        loc = arr[spatial.KDTree(arr).query(pt)[1]]
        return loc, dist, idx

    def correct_contour(self, ext_s, int_s, dx, dy, dx_med, dy_med, min_thickness=1):
        ext_s, int_s, dx, dy, dx_med, dy_med, fig, ax1, ax2 = self.show_circles(
            center=self.c, n_points=self.n_points, min_thickness=self.min_thickness)
        boolean_angle = self.is_internal_inside_external(ext_s, int_s)
        print(f'Is internal contour inside external contour? {boolean_angle}')
        if np.any(boolean_angle) is False:
            print('Internal contour is outside external contour')
            new_int = self.correct_intersection(
                ext_arr=ext_s,
                int_arr=int_s,
                dx=dx_med,
                dy=dy_med,
                # take the opposite of the boolean
                bool_arr=~np.array(boolean_angle)
            )
        elif np.any(boolean_angle) is True:
            print('Internal contour is inside external contour')
            bool_min_thickness_s = self.check_intersection(
                ext=ext_s, int=int_s, min_thickness=self.min_thickness)
            new_int = self.correct_intersection(
                ext_arr=ext_s,
                int_arr=int_s,
                dx=dx_med,
                dy=dy_med,
                bool_arr=bool_min_thickness_s
            )
        else:
            print('Could not check position between contours')
        return new_int
