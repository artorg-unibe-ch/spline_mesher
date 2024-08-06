import logging
import sys
from typing import List, Tuple

import cv2
import gmsh
import imutils
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.spatial as ss
import shapely.geometry as shpg
import SimpleITK as sitk
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from scipy import spatial
from scipy.interpolate import UnivariateSpline, splev, splprep, splrep
from scipy.optimize import minimize
from shapely import Polygon
from SimpleITK.SimpleITK import Image
from skimage.measure import find_contours

LOGGING_NAME = "MESHING"
# flake8: noqa: E203


class OCC_volume:
    def __init__(
        self,
        sitk_image: Image,
        img_path: None,
        filepath: str,
        filename: str,
        ASPECT: int,
        SLICE: int,
        UNDERSAMPLING: int,
        SLICING_COEFFICIENT: int,
        INSIDE_VAL: float,
        OUTSIDE_VAL: float,
        LOWER_THRESH: float,
        UPPER_THRESH: float,
        S: int,
        K: int,
        INTERP_POINTS: int,
        DP_SIMPLIFICATION_OUTER: int,
        DP_SIMPLIFICATION_INNER: int,
        debug_orientation: int,
        show_plots: bool,
        thickness_tol: float,
        phases: int,
    ):
        """
        Class that imports a voxel-based model and converts it to a geometrical simplified representation
        through the use of splines for each slice in the transverse plane.
        Following spline reconstruction, the model is meshed with GMSH.
        """
        self.logger = logging.getLogger(LOGGING_NAME)
        self.model = gmsh.model
        self.factory = self.model.occ

        self.sitk_image = sitk_image
        self.img_path = img_path
        self.filepath = filepath
        self.filename = filename
        self.debug_orientation = debug_orientation
        self.show_plots = bool(show_plots)

        self.ASPECT = ASPECT
        self.SLICE = SLICE
        self.THRESHOLD_PARAM = [INSIDE_VAL, OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH]
        self.UNDERSAMPLING = UNDERSAMPLING
        self.SLICING_COEFFICIENT = SLICING_COEFFICIENT
        self.S = S
        self.K = K
        self.INTERP_POINTS_S = INTERP_POINTS
        self.dp_simplification_outer = DP_SIMPLIFICATION_OUTER
        self.dp_simplification_inner = DP_SIMPLIFICATION_INNER
        self.height = 1.0
        self.spacing = []
        self.coordsX = []
        self.coordsY = []
        self.xy_sorted_closed = []
        self.x_mahalanobis = []
        self.y_mahalanobis = []
        self.xnew = []
        self.ynew = []
        self.slice_index = np.ndarray([])

        self.cortex_outer_tags = list()
        self.cortex_inner_tags = list()

        self.MIN_THICKNESS = float(thickness_tol)
        self.phases = int(phases)

        # Figure layout
        self.layout = go.Layout(
            plot_bgcolor="#FFF",  # Sets background color to white
            xaxis=dict(
                title="Medial - Lateral position (mm)",
                linecolor="#BCCCDC",  # Sets color of X-axis line
                showgrid=True,  # Removes X-axis grid lines
            ),
            yaxis=dict(
                title="Palmar - Dorsal position (mm)",
                linecolor="#BCCCDC",  # Sets color of Y-axis line
                showgrid=True,  # Removes Y-axis grid lines
            ),
        )

    def plot_mhd_slice(self, img):
        """
        Helper function to plot a slice of the MHD file

        Returns:
            None
        """
        img_view = sitk.GetArrayViewFromImage(img)

        fig, ax = plt.subplots(
            figsize=(
                np.shape(img_view)[0] / self.ASPECT,
                np.shape(img_view)[1] / self.ASPECT,
            )
        )
        plt.subplots_adjust(bottom=0.25)

        l = plt.imshow(
            img_view[self.SLICE, :, :],
            cmap="cividis",
            interpolation="nearest",
            aspect="equal",
        )
        plt.title(f"Slice n. {self.SLICE} of masked object", weight="bold")

        axcolor = "lightgoldenrodyellow"
        axSlider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(
            axSlider,
            "Slice",
            0,
            img_view.shape[0] - 1,
            valinit=self.SLICE,
            valstep=1,
        )

        def update(val):
            slice_index = int(slider.val)
            l.set_data(img_view[slice_index, :, :])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        if matplotlib.get_backend() == "agg":
            # set the backend to TkAgg
            matplotlib.use("TkAgg")
        plt.show()
        plt.close()
        return None

    def plot_slice(self, image, SLICE, title, ASPECT):
        img_view = sitk.GetArrayViewFromImage(image)

        fig, ax = plt.subplots(figsize=(ASPECT, ASPECT))
        plt.subplots_adjust(bottom=0.25)
        l = plt.imshow(
            img_view[SLICE, :, :], cmap="gray", interpolation="None", aspect="equal"
        )
        plt.title(title, weight="bold")

        axcolor = "lightgoldenrodyellow"
        axSlider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(
            axSlider, "Slice", 0, img_view.shape[0] - 1, valinit=SLICE, valstep=1
        )

        def update(val):
            slice_index = int(slider.val)
            l.set_data(img_view[slice_index, :, :])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        if matplotlib.get_backend() == "agg":
            # set the backend to TkAgg
            matplotlib.use("TkAgg")
        plt.show()
        plt.close()
        return None

    def exec_thresholding(self, image: Image, THRESHOLD_PARAM: List[float]) -> Image:
        # Binary threshold
        btif = sitk.BinaryThresholdImageFilter()
        btif.SetInsideValue(int(THRESHOLD_PARAM[0]))
        btif.SetOutsideValue(int(THRESHOLD_PARAM[1]))
        btif.SetLowerThreshold(THRESHOLD_PARAM[2])
        btif.SetUpperThreshold(THRESHOLD_PARAM[3])
        image_thr = btif.Execute(image)
        return image_thr

    def draw_contours(
        self, img: ndarray, loc: str = str("outer"), approximation: bool = True
    ) -> ndarray:
        """
        Find the contours of an image.

        Args:
            img (numpy.ndarray): The input image as a 2D numpy array.
            loc (str): The location of the contour. Can be "outer" or "inner". Defaults to "outer".
            approximation (bool): If True, contour is approximated using the Ramer-Douglas-Peucker (RDP) algorithm. Defaults to True.

        Returns:
            numpy.ndarray: The contour image as a 2D numpy array.

        Raises:
            ValueError: If the location of the contour is not valid.

        Credits to:
            https://stackoverflow.com/questions/25733694/process-image-to-find-external-contour

        Docs:
            https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
            https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/
            https://doi.org/10.1016/0167-8655(82)90016-2
        """
        eps = 0.001
        # eps = 0.015
        if loc == "outer":
            _contours, hierarchy = cv2.findContours(
                img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            out = np.zeros(np.shape(img), dtype=np.uint8)
            if approximation is True:
                cnts = imutils.grab_contours((_contours, hierarchy))
                c = max(cnts, key=cv2.contourArea)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
                contour = cv2.drawContours(out, [approx], -1, 1, 1)
            else:
                # all contours, in white, with thickness 1
                contour = cv2.drawContours(out, _contours, -1, 1, 1)
        elif loc == "inner":
            _contours, hierarchy = cv2.findContours(
                img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            inn = np.zeros(np.shape(img), dtype=np.uint8)
            if approximation is True:
                cnts = imutils.grab_contours((_contours, hierarchy))
                c = min(cnts, key=cv2.contourArea)
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
                contour = cv2.drawContours(inn, [approx], -1, 1, 1)
            else:
                contour = cv2.drawContours(inn, _contours, 2, 1, 1)
        else:
            raise ValueError(
                "The location of the contour is not valid. Please choose between 'outer' and 'inner'."
            )
        return contour

    def get_binary_contour(self, image: Image) -> Image:
        # https://itk.org/pipermail/community/2017-August/013464.html
        img_thr_join = sitk.JoinSeries(
            [
                sitk.BinaryContour(image[z, :, :], fullyConnected=True)
                for z in range(image.GetSize()[0])
            ]
        )
        img_thr_join = sitk.PermuteAxes(img_thr_join, [2, 0, 1])
        img_thr_join.SetSpacing(image.GetSpacing())
        return img_thr_join

    def get_draw_contour(self, image: Image, loc: str = str("outer")) -> ndarray:
        img_np = np.transpose(sitk.GetArrayFromImage(image), [2, 1, 0])
        contour_np = [
            self.draw_contours(img_np[z, :, :], loc, approximation=True)
            for z in np.arange(np.shape(img_np)[0])
        ]
        contour_np = np.flip(contour_np, axis=1)
        return contour_np

    def pad_image(self, image: Image, iso_pad_size: int) -> Image:
        """
        Pads the input image with a constant value (background value) to increase its size.
        Padding is used to prevent having contours on the edges of the image,
        which would cause the spline fitting to fail.
        Padding is performed on the transverse plane only
        (image orientation is assumed to be z, y, x)

        Args:
            image (SimpleITK.Image): The input image to be padded.
            iso_pad_size (int): The size of the padding to be added to each dimension.

        Returns:
            SimpleITK.Image: The padded image.
        """
        constant = int(sitk.GetArrayFromImage(image).min())
        image_thr = sitk.ConstantPad(
            image,
            (iso_pad_size, iso_pad_size, 0),
            (iso_pad_size, iso_pad_size, 0),
            constant,
        )
        return image_thr

    def binary_threshold(self, image: Image) -> Tuple[ndarray, ndarray]:
        """
        THRESHOLD_PARAM = [INSIDE_VAL, OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH]
        """
        THRESHOLD_PARAM = self.THRESHOLD_PARAM
        ASPECT = self.ASPECT
        SLICE = self.SLICE

        # if self.sitk_image is None:
        #     image = sitk.ReadImage(img_path)
        # else:
        #     image = self.sitk_image
        #     image = sitk.PermuteAxes(image, [2, 0, 1])
        #     image.SetSpacing(self.sitk_image.GetSpacing())

        # image_thr = self.exec_thresholding(image, THRESHOLD_PARAM)

        print(f"Image size before padding: {image.GetSize()}")

        image_pad = self.pad_image(image, iso_pad_size=10)

        print(f"Image size after padding: {image_pad.GetSize()}")

        if self.show_plots is True:
            self.plot_slice(
                image_pad, SLICE, f"Padded Image on slice n. {SLICE}", ASPECT
            )
        else:
            self.logger.info(f"Padded Image\t\t\tshow_plots:\t{self.show_plots}")

        print(f"Image size after padding: {image_pad.GetSize()}")
        return image_pad

    """
    def binary_threshold(self, img_path: str) -> Tuple[ndarray, ndarray]:
        THRESHOLD_PARAM = [INSIDE_VAL, OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH]
        THRESHOLD_PARAM = self.THRESHOLD_PARAM
        ASPECT = self.ASPECT
        SLICE = self.SLICE

        if self.sitk_image is None:
            image = sitk.ReadImage(img_path)
        else:
            image = self.sitk_image
            image = sitk.PermuteAxes(image, [2, 0, 1])
            image.SetSpacing(self.sitk_image.GetSpacing())

        # Subsample the image for faster processing
        image = sitk.Shrink(image, [1, self.UNDERSAMPLING, self.UNDERSAMPLING])

        image_thr = self.exec_thresholding(image, THRESHOLD_PARAM)

        image_thr = self.pad_image(image_thr, iso_pad_size=10)

        if self.show_plots is True:
            self.plot_slice(
                image_thr, SLICE, f"Padded Image on slice n. {SLICE}", ASPECT
            )
        else:
            self.logger.info(f"Padded Image\t\t\tshow_plots:\t{self.show_plots}")

        img_thr_join = self.get_binary_contour(image_thr)
        self.spacing = image.GetSpacing()

        if self.show_plots is True:
            self.plot_slice(
                img_thr_join, SLICE, f"Binary threshold on slice n. {SLICE}", ASPECT
            )
        else:
            self.logger.info(f"Binary threshold\t\tshow_plots:\t{self.show_plots}")

        if self.phases >= 1:
            outer_contour_np = self.get_draw_contour(img_thr_join, loc="outer")
            contour_ext = np.transpose(outer_contour_np, [2, 1, 0])

            if self.show_plots is True:
                outer_contour_sitk = sitk.GetImageFromArray(contour_ext)
                outer_contour_sitk.CopyInformation(image_thr)

                if self.phases == 1:
                    self.plot_slice(
                        outer_contour_sitk,
                        SLICE,
                        f"Outer contour on slice n. {SLICE}",
                        ASPECT,
                    )
            else:
                self.logger.info(f"Binary threshold\t\tshow_plots:\t{self.show_plots}")

        if self.phases == 2:
            inner_contour_np = self.get_draw_contour(img_thr_join, loc="inner")
            contour_int = np.transpose(inner_contour_np, [2, 1, 0])

            if self.show_plots is True:
                inner_contour_sitk = sitk.GetImageFromArray(contour_int)
                inner_contour_sitk.CopyInformation(image_thr)
                # put together both contours in sitk
                # contours_sitk = sitk.Compose(outer_contour_sitk, inner_contour_sitk)

                self.plot_slice(
                    inner_contour_sitk,
                    # contours_sitk,
                    SLICE,
                    f"Inner contour on slice n. {SLICE}",
                    ASPECT,
                )
            else:
                self.logger.info(f"Binary threshold\t\tshow_plots:\t{self.show_plots}")

        if self.phases > 2:
            raise ValueError(
                "The number of phases is greater than 2. Only biphasic materials are accepted (e.g. cort+trab)."
            )

        img_size = image_thr.GetSize()
        img_spacing = image_thr.GetSpacing()

        xx = np.arange(
            0,
            img_size[1] * img_spacing[1],
            img_size[1] * img_spacing[1] / float(img_size[1]),
        )
        yy = np.arange(
            0,
            img_size[2] * img_spacing[2],
            img_size[2] * img_spacing[2] / float(img_size[2]),
        )
        coordsX, coordsY = np.meshgrid(xx, yy)
        self.coordsX = np.transpose(coordsX, [1, 0])
        self.coordsY = np.transpose(coordsY, [1, 0])
        return contour_ext, contour_int
    """

    def sort_xy(self, x: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
        # https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data

        x0 = np.mean(x)
        y0 = np.mean(y)
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

        angles = np.where(
            (y - y0) > 0, np.arccos((x - x0) / r), 2 * np.pi - np.arccos((x - x0) / r)
        )
        mask = np.argsort(angles)

        x_sorted = x[mask]
        y_sorted = y[mask]
        return x_sorted, y_sorted

    def plotly_add_traces(self, fig, original, spline_interp):
        fig.add_traces(
            [
                go.Scatter(
                    mode="lines+markers",
                    marker=dict(color="Black", size=3),
                    visible=False,
                    line=dict(color=px.colors.qualitative.Dark2[0], width=2),
                    name="Original",
                    x=original[:, 0],
                    y=original[:, 1],
                ),
                go.Scatter(
                    visible=False,
                    line=dict(color=px.colors.qualitative.Dark2[1], width=5),
                    name=f"B-spline order {self.K}",
                    x=spline_interp[:, 0],
                    y=spline_interp[:, 1],
                ),
            ]
        )
        return fig

    def plotly_makefig(self, fig):
        fig.data[self.SLICING_COEFFICIENT].visible = True

        # Create and add slider
        steps = []
        height = self.spacing[0] * (max(self.slice_index) + 1)
        for i in range(0, len(fig.data)):
            if i % 2 == 0:
                step = dict(
                    method="update",
                    args=[
                        {"visible": [False] * len(fig.data)},
                        {
                            "title": f"Slice position {(((i) * height) / (len(fig.data) - 2)):.2f} mm"
                        },
                    ],
                    label=i,
                )
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][i] = True
                step["args"][0]["visible"][i + 1] = True
                steps.append(step)

        sliders = [
            dict(
                active=10,
                currentvalue={"prefix": "Slice: ", "suffix": " [-]"},
                pad={"t": 50},
                steps=steps,
            )
        ]

        fig.update_layout(sliders=sliders, autosize=False, width=800, height=800)

        fig.add_annotation(
            text="Slice representation through splines",
            xref="paper",
            yref="paper",
            x=0.1,
            y=1,
            showarrow=False,
            font=dict(size=18, family="stix"),
        )

        fig.update_xaxes(range=[0, 50])
        fig.update_yaxes(range=[0, 50])
        fig.show()
        return fig

    def check_orient(
        self, x: ndarray, y: ndarray, direction: int = 1
    ) -> Tuple[ndarray, ndarray]:
        """
        Author: Simone Poncioni, MSB
        Date: 18.08.2022
        Functionality: Orient all array in the same direction (cw or ccw)
        (Sometimes planes would reorient in opposite direction, making them unsuitable for interplane connectivity)

        Args:
        x = 1D-arr (x coords of points)
        y = 1D-arr (y coords of points)
        directions: 1 = cw, 2 = ccw

        Returns:
        x_o = reoriented x 1D-arr
        y_o = reoriented y 1D-arr
        """

        if direction == 1:
            if self.debug_orientation == 1:
                self.logger.debug("Desired direction: cw")
            if y[1] > y[0]:
                if self.debug_orientation == 1:
                    self.logger.debug("Not flipping")
                else:
                    pass
                x_o = x
                y_o = y
            elif y[1] < y[0]:
                if self.debug_orientation == 1:
                    self.logger.debug("Flipping")
                else:
                    pass
                x_o = np.flip(x, axis=0)
                y_o = np.flip(y, axis=0)
            else:
                self.logger.debug("Something went wrong while flipping the array")

        if direction == 2:
            if self.debug_orientation == 1:
                self.logger.debug("Desired direction: ccw")
            if y[1] < y[0]:
                if self.debug_orientation == 1:
                    self.logger.debug("Not flipping")
                x_o = x
                y_o = y
            elif y[1] > y[0]:
                if self.debug_orientation == 1:
                    self.logger.debug("Flipping")
                x_o = np.flip(x, axis=0)
                y_o = np.flip(y, axis=0)
            else:
                self.logger.warning("Something went wrong while flipping the array")
        return x_o, y_o

    def sort_mahalanobis_legacy(self, data, start):
        dist_m = ss.distance.squareform(ss.distance.pdist(data.T, "mahalanobis"))
        total_points = data.shape[1]
        points_index = set(range(total_points))
        sorted_index = []
        target = start

        points_index.discard(target)
        while len(points_index) > 0:
            candidate = list(points_index)
            nneigbour = candidate[dist_m[target, candidate].argmin()]
            points_index.discard(nneigbour)
            points_index.discard(target)
            sorted_index.append(target)
            target = nneigbour
        sorted_index.append(target)

        x_mahalanobis = data[0][sorted_index]
        y_mahalanobis = data[1][sorted_index]
        return x_mahalanobis, y_mahalanobis

    def remove_self_intersections(self, sorted_data):
        pass

    def sort_mahalanobis(self, data: ndarray) -> Tuple[ndarray, ndarray]:
        # Compute pairwise distances between all points in the contour
        dist_matrix = ss.distance.cdist(data.T, data.T)

        # Initialize variables
        n_points = data.shape[1]
        visited = np.zeros(n_points, dtype=bool)
        sorted_indices = np.zeros(n_points, dtype=int)
        sorted_indices[0] = 0
        visited[0] = True

        # Iterate over remaining points in the contour
        for i in range(1, n_points):
            # Compute distances from the previous point to all unvisited points
            dist_to_prev = dist_matrix[sorted_indices[i - 1], :]
            dist_to_prev[visited] = np.inf

            # Find the closest unvisited point to the previous point
            closest_unvisited = np.argmin(dist_to_prev)
            sorted_indices[i] = closest_unvisited
            visited[closest_unvisited] = True

        # Reorder the data based on the sorted indices
        sorted_data = data[:, sorted_indices]

        # Ensure the contour does not self-intersect
        # sorted_data = self.remove_self_intersections(sorted_data)
        return sorted_data[0], sorted_data[1]

    def sort_surface(self, image_slice: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """
        Sort surface points in a clockwise direction and with Mahalanobis distance to add robustness


        Args:
            slices (int): slice number

        Returns:
            self.xy_sorted_closed (numpy.ndarray): xy array of sorted points
            self.x_mahanalobis (numpy.ndarray): x array of sorted points with mahalanobis distance
            self.y_mahanalobis (numpy.ndarray): y array of sorted points with mahalanobis distance
            self.xnew (numpy.ndarray): x array of sorted points interpolated with bspline and in the same direction
            self.ynew (numpy.ndarray): y array of sorted points interpolated with bspline and in the same direction
        """

        x = self.coordsX[image_slice == 1][0 :: self.UNDERSAMPLING]
        y = self.coordsY[image_slice == 1][0 :: self.UNDERSAMPLING]

        x_s, y_s = self.sort_xy(x, y)
        xy_sorted = np.c_[x_s, y_s]
        xy_sorted_closed = np.vstack([xy_sorted, xy_sorted[0]])
        x_mahalanobis, y_mahalanobis = self.sort_mahalanobis(xy_sorted.T)

        x_mahalanobis = np.append(x_mahalanobis, x_mahalanobis[0])
        y_mahalanobis = np.append(y_mahalanobis, y_mahalanobis[0])

        x_copy = x_mahalanobis.copy()
        y_copy = y_mahalanobis.copy()

        BNDS = 10  # was 5, then 10
        x_bnds, y_bnds = x_copy[BNDS:-BNDS], y_copy[BNDS:-BNDS]
        # find the knot points
        tckp, _ = splprep(
            [x_bnds, y_bnds],
            s=self.S,
            k=self.K,
            per=1,
            ub=[x_copy, y_copy][0],
            ue=[x_copy, y_copy][0],
            quiet=1,
        )

        # evaluate spline, including interpolated points
        xnew, ynew = splev(np.linspace(0, 1, self.INTERP_POINTS_S), tckp)
        if np.allclose(xnew[0], xnew[-1], rtol=1e-05, atol=1e-08):
            xnew = np.append(xnew, xnew[0])
        else:
            pass
        if np.allclose(ynew[0], ynew[-1], rtol=1e-05, atol=1e-08):
            ynew = np.append(ynew, ynew[0])
        else:
            pass

        # Sanity check to ensure directionality of sorting in cw- or ccw-direction
        xnew_oriented, ynew_oriented = self.check_orient(xnew, ynew, direction=1)

        return (
            xy_sorted_closed,
            xnew_oriented,
            ynew_oriented,
        )

    def input_sanity_check(self, ext_contour_s: np.ndarray, int_contour_s: np.ndarray):
        """
        Sanity check for the input data before cortical sanity check

        Args:
            ext_contour_s (np.ndarray): array of external contour points
            int_contour_s (np.ndarray): array of internal contour points

        Returns:
            ext_contour_s (np.ndarray): array of external contour points
            with defined shape and structure (no duplicates, closed contour)

            int_contour_s (np.ndarray): array of internal contour points
            with defined shape and structure (no duplicates, closed contour)
        """
        # sanity check of input data ext_contour_s and int_contour_s
        # make sure that arr[0] is the same as arr[-1]
        if not np.allclose(int_contour_s[0], int_contour_s[-1], rtol=1e-05, atol=1e-08):
            # append first element to the end of the array
            int_contour_s = np.append(int_contour_s, [int_contour_s[0]], axis=0)
        # if arr[0] is equal to arr[1] then remove the first element
        if np.allclose(ext_contour_s[0], ext_contour_s[1], rtol=1e-05, atol=1e-08):
            ext_contour_s = ext_contour_s[1:]
        if np.allclose(int_contour_s[0], int_contour_s[1], rtol=1e-05, atol=1e-08):
            int_contour_s = int_contour_s[1:]
        # if arr[-1] is equal to arr[-2] then remove the last element
        if np.allclose(ext_contour_s[-1], ext_contour_s[-2], rtol=1e-05, atol=1e-08):
            ext_contour_s = ext_contour_s[:-1]
        if np.allclose(int_contour_s[-1], int_contour_s[-2], rtol=1e-05, atol=1e-08):
            int_contour_s = int_contour_s[:-1]
        if not np.allclose(ext_contour_s[0], ext_contour_s[-1], rtol=1e-05, atol=1e-08):
            # append first element to the end of the array
            ext_contour_s = np.append(ext_contour_s, [ext_contour_s[0]], axis=0)
        # check that shape of ext_contour_s and int_contour_s are the same
        if ext_contour_s.shape != int_contour_s.shape:
            self.logger.warning(
                f"Error in the shape of the external and internal contours of the slice:\t{slice}"
            )
            sys.exit(90)
        return ext_contour_s, int_contour_s

    def output_sanity_check(self, initial_contour: np.ndarray, contour_s: np.ndarray):
        # check that after csc.CorticalSanityCheck elements of arrays external and
        # internal contours have the same structure and shape as before csc.CorticalSanityCheck
        # sanity check of input data ext_contour_s and int_contour_s
        if np.allclose(initial_contour[0], initial_contour[1], rtol=1e-05, atol=1e-08):
            self.logger.warning("External contour has a duplicate first point")
            if not np.allclose(contour_s[0], contour_s[1], rtol=1e-05, atol=1e-08):
                self.logger.warning(
                    "New external contour does not have a duplicate first point"
                )
                contour_s = np.insert(contour_s, 0, contour_s[0], axis=0)
                self.logger.info("New external contour now has a duplicate first point")
        if np.allclose(initial_contour[-1], initial_contour[-2]):
            self.logger.warning("External contour has a duplicate last point")
            if not np.allclose(contour_s[-1], contour_s[-2], rtol=1e-05, atol=1e-08):
                self.logger.warning(
                    "New external contour does not have a duplicate last point"
                )
                contour_s = np.append(contour_s, [contour_s[-1]], axis=0)
                self.logger.info("New external contour now has a duplicate last point")
        if np.shape(initial_contour) != np.shape(contour_s):
            self.logger.warning(
                "External contour has a different shape than the initial contour"
            )
        else:
            self.logger.log(
                self.logger.INFO,
                "External contour has the same shape as before csc.CorticalSanityCheck",
            )
        return contour_s

    def _pnts_on_line_(self, a, spacing=1, is_percent=False):  # densify by distance
        """Add points, at a fixed spacing, to an array representing a line.
        https://stackoverflow.com/questions/64995977/generating-equidistance-points-along-the-boundary-of-a-polygon-but-cw-ccw

        **See**  `densify_by_distance` for documentation.

        Parameters
        ----------
        a : array
            A sequence of `points`, x,y pairs, representing the bounds of a polygon
            or polyline object.
        spacing : number
            Spacing between the points to be added to the line.
        is_percent : boolean
            Express the densification as a percent of the total length.

        Notes
        -----
        Called by `pnt_on_poly`.
        """
        N = len(a) - 1  # segments
        dxdy = a[1:, :] - a[:-1, :]  # coordinate differences
        leng = np.sqrt(np.einsum("ij,ij->i", dxdy, dxdy))  # segment lengths
        if is_percent:  # as percentage
            spacing = abs(spacing)
            spacing = min(spacing / 100, 1.0)
            steps = (sum(leng) * spacing) / leng  # step distance
        else:
            steps = leng / spacing  # step distance
        deltas = dxdy / (steps.reshape(-1, 1))  # coordinate steps
        pnts = np.empty((N,), dtype="O")  # construct an `O` array
        for i in range(N):  # cycle through the segments and make
            num = np.arange(steps[i])  # the new points
            pnts[i] = np.array((num, num)).T * deltas[i] + a[i]
        a0 = a[-1].reshape(1, -1)  # add the final point and concatenate
        return np.concatenate((*pnts, a0), axis=0)

    def volume_splines(self) -> Tuple[ndarray, ndarray]:
        contour_ext_fig, contour_int_fig = self.binary_threshold(img_path=self.img_path)

        self.slice_index = np.linspace(
            1, len(contour_ext_fig[0, 0, :]) - 1, self.SLICING_COEFFICIENT, dtype=int
        )

        if self.show_plots is True:
            fig = go.Figure(layout=self.layout)
        else:
            self.logger.info(f"Volume_splines\t\tshow_plots:\t{self.show_plots}")
            fig = None

        if self.phases >= 1:
            img_contours_ext = sitk.GetImageFromArray(contour_ext_fig, isVector=True)
            image_data_ext = np.transpose(
                sitk.GetArrayViewFromImage(img_contours_ext), [2, 1, 0]
            )
        if self.phases == 2:
            img_contours_int = sitk.GetImageFromArray(contour_int_fig, isVector=True)
            image_data_int = np.transpose(
                sitk.GetArrayViewFromImage(img_contours_int), [2, 1, 0]
            )

        contour_ext = np.ndarray(shape=(0, 3))
        contour_int = np.ndarray(shape=(0, 3))
        image_slice_ext = []
        for i, _slice in enumerate(self.slice_index):
            self.logger.debug(f"Slice:\t{_slice}")
            if self.phases >= 1:
                image_slice_ext = image_data_ext[_slice, :, :][::-1, ::-1]
                original, cortical_ext_x, cortical_ext_y = self.sort_surface(
                    image_slice_ext
                )
                z = np.ones(len(cortical_ext_x)) * (self.spacing[0] * _slice)
                contour_ext = np.append(
                    contour_ext, np.c_[cortical_ext_x, cortical_ext_y, z]
                )

                if self.phases == 1:
                    if self.show_plots is True:
                        fig = self.plotly_add_traces(
                            fig, original, np.c_[contour_ext[:, 0], contour_ext[:, 1]]
                        )
                    else:
                        fig = None
            else:
                self.logger.warning(f"Phases is not >= 1: {self.phases}")

            if self.phases == 2:
                image_slice_int = image_data_int[_slice, :, :][::-1, ::-1]
                original, cortical_int_x, cortical_int_y = self.sort_surface(
                    image_slice_int
                )
                contour_int = np.append(
                    contour_int, np.c_[cortical_int_x, cortical_int_y, z]
                )

                if self.phases == 2:
                    if self.show_plots is True:
                        fig = self.plotly_add_traces(
                            fig, original, np.c_[cortical_int_x, cortical_int_y, z]
                        )
                    else:
                        fig = None
            else:
                self.logger.warning(f"Phases =/= 2: {self.phases}")

        if self.show_plots is True:
            self.plotly_makefig(fig)
        elif self.show_plots is False:
            fig = None

        contour_ext = contour_ext.reshape(-1, 3)
        contour_int = contour_int.reshape(-1, 3)

        return contour_ext, contour_int

    def guess(self, x, y, k, s, w=None):
        """Do an ordinary spline fit to provide knots"""
        return splrep(x, y, w, k=k, s=s)

    def err(self, c, x, y, t, k, w=None):
        """The error function to minimize"""
        diff = y - splev(x, (t, c, k))
        if w is None:
            diff = np.einsum("...i,...i", diff, diff)
        else:
            diff = np.dot(diff * diff, w)
        return np.abs(diff)

    def spline_neumann(self, x, y, k=3, s=0, w=None):
        t, c0, k = self.guess(x, y, k, s, w=w)
        x0 = x[0]  # point at which zero slope is required
        con = {
            "type": "eq",
            "fun": lambda c: splev(x0, (t, c, k), der=1),
        }
        opt = minimize(self.err, c0, (x, y, t, k, w), constraints=con)
        copt = opt.x
        return UnivariateSpline._from_tck((t, copt, k))

    def interpolate_vertical_lines_3d(self, line_sets):
        """
        https://stackoverflow.com/questions/32046582/spline-with-constraints-at-border
        """
        interpolated_lines = []
        for line in line_sets:
            line = np.array(line)

            # Check for NaN or Inf values
            if np.any(np.isnan(line)) or np.any(np.isinf(line)):
                raise ValueError("Data contains NaN or Inf values.")

            # Check for duplicate points
            line = np.unique(line, axis=0)
            if len(line) < 4:
                raise ValueError(
                    "Insufficient unique data points for cubic spline interpolation."
                )

            # Sort by Z-values
            line_sorted_by_z = line[np.argsort(line[:, 2])]
            x_sorted = line_sorted_by_z[:, 0]
            y_sorted = line_sorted_by_z[:, 1]
            z_sorted = line_sorted_by_z[:, 2]

            # Perform cubic spline interpolation with Neumann boundary conditions
            splrep_eval_x = self.spline_neumann(z_sorted, x_sorted, k=3, s=1e6)
            splrep_eval_y = self.spline_neumann(z_sorted, y_sorted, k=3, s=1e6)

            # Generate new Z values for interpolation
            znew = np.linspace(z_sorted[0], z_sorted[-1], self.SLICING_COEFFICIENT)
            xnew = splrep_eval_x(znew)
            ynew = splrep_eval_y(znew)

            interpolated_lines.append(np.vstack((xnew, ynew, znew)).T)

        all_points_array = np.concatenate(interpolated_lines, axis=0)

        # Plot final interpolated points
        if self.show_plots:
            plt.figure(figsize=(10, 10))
            plt.scatter(
                all_points_array[:, 0],
                all_points_array[:, 1],
                label="Interpolated Points",
            )
            plt.legend()
            plt.title("Interpolated Points")
            plt.show()

        return all_points_array

    def classify_and_store_contours(self, mask, slice_idx):
        contours = find_contours(mask[:, :, slice_idx])  # , level=0.5)
        if contours:
            outer_contours = contours[0]  # First contour as outer
            if len(contours) > 1:
                inner_contours = contours[1]  # Second contour as inner
        return outer_contours, inner_contours

    def evaluate_bspline(self, contour):
        tckp, _ = splprep(
            [contour[:, 0], contour[:, 1]],
            s=self.S,
            k=self.K,
            per=1,
            ub=[contour[0][0], contour[-1][0]],
            ue=[contour[0][1], contour[-1][1]],
            quiet=1,
        )
        xnew, ynew = splev(np.linspace(0, 1, self.INTERP_POINTS_S), tckp)
        xnew = np.append(xnew, xnew[0])
        ynew = np.append(ynew, ynew[0])
        return xnew, ynew

    def calculate_outer_offset(self, contour):
        MIN_THICKNESS = self.MIN_THICKNESS
        contour = np.array(contour)[:, :2]
        # Assuming the contour is (x, y, z) and we only need (x, y)
        outer_polygon = Polygon(contour)
        offset = int(MIN_THICKNESS)
        buffered_polygon = outer_polygon.buffer(-offset)
        try:
            if len(list(buffered_polygon.geoms)) > 1:
                polygons = list(buffered_polygon.geoms)
                self.logger.warning(
                    f"Buffered polygon has more than one geometry: {len(polygons)}"
                )
        except AttributeError:
            pass
        offset_polygon = np.array(buffered_polygon.exterior.coords)
        return offset_polygon

    def check_inner_offset(self, outer_polygon, inn_contour):
        outer_polygon = shpg.Polygon(outer_polygon)
        outer_contour = np.array(outer_polygon.exterior.coords)
        inn_contour = (np.array(inn_contour).reshape(-1, 3)[:, :2]).reshape(-1, 2)
        is_inside_shpg = [
            shpg.Point(point).within(outer_polygon) for point in inn_contour
        ]
        is_inside = np.c_[is_inside_shpg, is_inside_shpg]
        closest_points = [
            outer_contour[spatial.KDTree(outer_contour).query(point)[1]]
            for point in inn_contour
        ]
        closest_points = np.array(closest_points).reshape(-1, 2)
        np.copyto(dst=inn_contour, src=closest_points, where=np.logical_not(is_inside))
        return inn_contour

    def process_slice(self, mask, slice_idx):
        DP_SIMPLIFICATION_OUTER = self.dp_simplification_outer
        DP_SIMPLIFICATION_INNER = self.dp_simplification_inner
        out_cont, inn = self.classify_and_store_contours(mask, slice_idx)

        out_dp = out_cont.copy()
        out_dp = [shpg.Point(point) for point in out_dp]
        out_dp = shpg.Polygon(out_dp)
        out_dp = out_dp.simplify(DP_SIMPLIFICATION_OUTER, preserve_topology=True)

        xout, yout = self.evaluate_bspline(np.array(out_dp.exterior.coords))

        out_bspline = np.column_stack((xout, yout))
        out_bspline = np.column_stack(
            (out_bspline, np.full(out_bspline.shape[0], slice_idx))
        )

        inn_dp = inn.copy()
        inn_dp = [shpg.Point(point) for point in inn_dp]
        inn_dp = shpg.Polygon(inn_dp)
        inn_dp = inn_dp.simplify(DP_SIMPLIFICATION_INNER, preserve_topology=True)

        xin, yin = self.evaluate_bspline(np.array(inn_dp.exterior.coords))

        inn_bspline = np.column_stack((xin, yin))
        inn_bspline = np.column_stack(
            (inn_bspline, np.full(inn_bspline.shape[0], slice_idx))
        )
        return out_bspline, inn_bspline

    def get_line_sets(self, contour):
        num_points = contour[0].shape[0]
        line_sets = []

        # Align slices based on the first point
        first_point = contour[0][0]
        for i in range(1, len(contour)):
            distances = np.linalg.norm(contour[i] - first_point, axis=1)
            min_idx = np.argmin(distances)
            contour[i] = np.roll(contour[i], -min_idx, axis=0)

        for pt_idx in range(num_points):
            vertical_ll = []
            for _slice in contour:
                # Collecting the pt_idx-th point from each slice
                vertical_ll.append(_slice[pt_idx])
            line_sets.append(vertical_ll)

        if self.show_plots:
            for line_set in line_sets:
                line_set = np.array(line_set)
                plt.plot(line_set[:, 0], line_set[:, 1])
                plt.title("Matched vertical lines")
            plt.show()

        return line_sets

    def plot_slices_with_slider(self, imnp, total_slices, results):
        """
        Function to plot slices with a slider to navigate through slices.

        Args:
            imnp (numpy.ndarray): 3D numpy array representing the image.
            total_slices (int): Total number of slices.
            results (list): List of tuples containing outer and inner contour plots for each slice.

        Returns:
            None
        """
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        img_display = ax.imshow(imnp[:, :, 0], cmap="gray")
        outer_plot, inner_plot = results[0]
        (outer_contour_plot,) = ax.plot(
            outer_plot[:, 1], outer_plot[:, 0], "r", label="Outer Contour"
        )
        (inner_contour_plot,) = ax.plot(
            inner_plot[:, 1], inner_plot[:, 0], "b", label="Inner Contour"
        )
        ax.legend()

        axcolor = "lightgoldenrodyellow"
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        slider = Slider(
            ax_slider, "Slice", 0, len(total_slices) - 1, valinit=0, valstep=1
        )

        def update(val):
            slice_idx = int(slider.val)
            img_display.set_data(imnp[:, :, total_slices[slice_idx]])
            outer_plot, inner_plot = results[slice_idx]
            outer_contour_plot.set_data(outer_plot[:, 1], outer_plot[:, 0])
            inner_contour_plot.set_data(inner_plot[:, 1], inner_plot[:, 0])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        if matplotlib.get_backend() == "agg":
            matplotlib.use("TkAgg")
        plt.show()
        plt.close()
        return None

    def volume_splines_optimized(self, imsitk_pad) -> Tuple[ndarray, ndarray]:
        imnp = sitk.GetArrayFromImage(imsitk_pad)
        self.spacing = imsitk_pad.GetSpacing()
        imnp = np.transpose(imnp, [2, 1, 0])
        imnp = np.flip(imnp, axis=1)
        height = imnp.shape[2]
        self.logger.debug(f"Height:\t{height}")

        NUM_SLICES = height // 10  # TODO: parametrize this value
        total_slices = np.linspace(0, height - 1, NUM_SLICES, dtype=int)

        processed_slices = []
        for slice_idx in total_slices:
            print(f"Processing slice {slice_idx}")
            result = self.process_slice(imnp, slice_idx)
            processed_slices.append(result)

        if self.show_plots is True:
            self.plot_slices_with_slider(imnp, total_slices, processed_slices)

        outer_contours, inner_contours = zip(*processed_slices)
        out_arr_mm = np.array(outer_contours).reshape(-1, 3) * 0.061
        inn_arr_mm = np.array(inner_contours).reshape(-1, 3) * 0.061

        z_unique = np.unique(out_arr_mm[:, 2])
        cortical_ext_split = np.array_split(out_arr_mm, len(z_unique))
        cortical_int_split = np.array_split(inn_arr_mm, len(z_unique))

        line_sets_ext = self.get_line_sets(cortical_ext_split)
        line_sets_int = self.get_line_sets(cortical_int_split)

        cortical_ext_interp = self.interpolate_vertical_lines_3d(line_sets_ext)
        cortical_int_interp = self.interpolate_vertical_lines_3d(line_sets_int)

        if self.show_plots is True:
            # plot cortical_ext_interp and cortical_int_interp
            plt.figure(figsize=(10, 10))
            plt.plot(
                cortical_ext_interp[:, 0],
                cortical_ext_interp[:, 1],
                label="Outer Contour",
            )
            plt.plot(
                cortical_int_interp[:, 0],
                cortical_int_interp[:, 1],
                label="Inner Contour",
            )
            plt.legend()
            plt.show()

        slices_ext = {}
        for point in cortical_ext_interp:
            z_value = point[2]
            if z_value not in slices_ext:
                slices_ext[z_value] = []
            slices_ext[z_value].append(point)

        slices_int = {}
        for point in cortical_int_interp:
            z_value = point[2]
            if z_value not in slices_int:
                slices_int[z_value] = []
            slices_int[z_value].append(point)

        # TODO: I really don't like this back and forth conversion, but tentative fix for now

        all_coords_int = []
        for key in slices_int:
            for coord_array in slices_int[key]:
                all_coords_int.append(coord_array)
        cortical_int_array = np.vstack(all_coords_int)

        all_coords_ext = []
        for key in slices_ext:
            for coord_array in slices_ext[key]:
                all_coords_ext.append(coord_array)
        cortical_ext_array = np.vstack(all_coords_ext)  #! change of name !

        inn_sanity = []
        for z_value, slice_ext_points in slices_ext.items():
            out_offset = self.calculate_outer_offset(slice_ext_points)
            inn_dp = self.check_inner_offset(out_offset, slices_int[z_value])
            inn_dp = np.column_stack((inn_dp, np.full(inn_dp.shape[0], z_value)))
            inn_sanity.append(inn_dp)
        inn_sanity = np.array(inn_sanity).reshape(-1, 3)

        return cortical_ext_array, cortical_int_array
