"""
Geometry representation and meshing through spline reconstruction
Author: Simone Poncioni, MSB
Date: 07-09.2022
"""
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import SimpleITK as sitk
from scipy.interpolate import splev, splprep
from pathlib import Path
import scipy.spatial as ss
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gmsh
import sys
import coloredlogs
import logging
import cortical_sanity as csc
from gmsh_mesh_builder import Mesher, TrabecularVolume
import os
from itertools import chain


pio.renderers.default = "browser"
coloredlogs.install()


class OCC_volume:
    def __init__(
        self,
        img_path,
        filepath,
        filename,
        ASPECT,
        SLICE,
        UNDERSAMPLING,
        SLICING_COEFFICIENT,
        INSIDE_VAL,
        OUTSIDE_VAL,
        LOWER_THRESH,
        UPPER_THRESH,
        S,
        K,
        INTERP_POINTS,
        debug_orientation,
        show_plots,
        location,
        offset,
        thickness_tol,
        phases,
    ):
        """
        Class that imports a voxel-based model and converts it to a geometrical simplified representation
        through the use of splines for each slice in the transverse plane.
        Following spline reconstruction, the model is meshed with GMSH.
        """
        self.model = gmsh.model
        self.factory = self.model.occ

        self.img_path = img_path
        self.filepath = filepath
        self.filename = filename
        self.debug_orientation = debug_orientation
        self.show_plots = bool(show_plots)
        self.location = str(location)
        self.offset = int(offset)

        self.ASPECT = ASPECT
        self.SLICE = SLICE
        self.THRESHOLD_PARAM = [INSIDE_VAL, OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH]
        self.UNDERSAMPLING = UNDERSAMPLING
        self.SLICING_COEFFICIENT = SLICING_COEFFICIENT
        self.S = S
        self.K = K
        self.INTERP_POINTS_S = INTERP_POINTS
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

    def plot_mhd_slice(self):
        """
        Helper function to plot a slice of the MHD file

        Returns:
            None
        """
        if self.show_plots is True:
            img = sitk.PermuteAxes(sitk.ReadImage(self.img_path), [1, 2, 0])
            img_view = sitk.GetArrayViewFromImage(img)

            plt.figure(
                f"Plot MHD slice n.{self.SLICE}",
                figsize=(
                    np.shape(img_view)[0] / self.ASPECT,
                    np.shape(img_view)[1] / self.ASPECT,
                ),
            )
            plt.imshow(
                img_view[self.SLICE, :, :],
                cmap="cividis",
                interpolation="nearest",
                aspect="equal",
            )
            plt.title(f"Slice n. {self.SLICE} of masked object", weight="bold")

            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(cax=cax)
            plt.show()
        else:
            logging.info(f"MHD slice\t\tshow_plots:\t{self.show_plots}")
        return None

    def plot_slice(self, image, SLICE, title, ASPECT):
        plt.figure("Binary contour", figsize=(ASPECT, ASPECT))
        plt.imshow(
            sitk.GetArrayViewFromImage(image)[:, :, SLICE],
            cmap="gray",
            interpolation="None",
            aspect="equal",
        )
        plt.title(title, weight="bold")
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(cax=cax)
        plt.show()
        plt.close()
        return None

    def exec_thresholding(self, image, THRESHOLD_PARAM):
        # Binary threshold
        btif = sitk.BinaryThresholdImageFilter()
        btif.SetInsideValue(THRESHOLD_PARAM[0])
        btif.SetOutsideValue(THRESHOLD_PARAM[1])
        btif.SetLowerThreshold(THRESHOLD_PARAM[2])
        btif.SetUpperThreshold(THRESHOLD_PARAM[3])
        image_thr = btif.Execute(image)
        return image_thr

    def draw_contours(self, img, contour_s, loc=str("outer")):
        """
        https://stackoverflow.com/questions/25733694/process-image-to-find-external-contour
        """
        # fmt: off
        if loc == "outer":
            _contours, hierarchy = cv2.findContours(
                img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            out = np.empty(np.shape(img))
            contour = cv2.drawContours(out, _contours, -1, 1, 1)  # all contours, in white, with thickness 1
            contour_s.append(contour)
        elif loc == "inner":
            _contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            inn = np.empty(np.shape(img))
            contour = cv2.drawContours(inn, _contours, 2, 1, 1)
        else:
            raise ValueError("The location of the contour is not valid. Please choose between 'outer' and 'inner'.")
        # fmt: on
        return contour

    def get_binary_contour(self, image):
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

    def get_draw_contour(self, image, contour_s, loc=str("outer")):
        img_np = np.transpose(sitk.GetArrayFromImage(image), [2, 1, 0])
        outer_contour_np = [
            self.draw_contours(img_np[z, :, :], contour_s, loc)
            for z in np.arange(np.shape(img_np)[0])
        ]
        outer_contour_np = np.flip(outer_contour_np, axis=1)
        return outer_contour_np

    def binary_threshold(self, contour_ext, contour_int, img_path: str):
        """
        THRESHOLD_PARAM = [INSIDE_VAL, OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH]
        """
        THRESHOLD_PARAM = [0, 1, 0, 0.9]
        ASPECT = 50
        SLICE = 50

        image = sitk.ReadImage(img_path)
        image_thr = self.exec_thresholding(image, THRESHOLD_PARAM)
        img_thr_join = self.get_binary_contour(image_thr)
        self.spacing = image.GetSpacing()

        if self.show_plots is True:
            self.plot_slice(
                img_thr_join, SLICE, f"Binary threshold on slice n. {SLICE}", ASPECT
            )
        else:
            logging.info(f"Binary threshold\tshow_plots:\t{self.show_plots}")

        if self.phases >= 1:
            outer_contour_np = self.get_draw_contour(
                img_thr_join, contour_ext, loc="outer"
            )
            contour_ext = np.transpose(outer_contour_np, [2, 1, 0])

            if self.phases == 1:
                if self.show_plots is True:
                    outer_contour_sitk = sitk.GetImageFromArray(contour_ext)
                    outer_contour_sitk.CopyInformation(image)

                    self.plot_slice(
                        outer_contour_sitk,  # outer_contour_sitk,
                        SLICE,
                        f"Outer contour on slice n. {SLICE}",
                        ASPECT,
                    )
                else:
                    logging.info(f"Binary threshold\tshow_plots:\t{self.show_plots}")

        if self.phases == 2:
            inner_contour_np = self.get_draw_contour(
                img_thr_join, contour_int, loc="inner"
            )
            contour_int = np.transpose(inner_contour_np, [2, 1, 0])

            if self.show_plots is True:
                inner_contour_sitk = sitk.GetImageFromArray(contour_int)
                inner_contour_sitk.CopyInformation(image)

                self.plot_slice(
                    inner_contour_sitk,
                    SLICE,
                    f"Inner contour on slice n. {SLICE}",
                    ASPECT,
                )
            else:
                logging.info(f"Binary threshold\tshow_plots:\t{self.show_plots}")

        if self.phases > 2:
            raise ValueError(
                "The number of phases is greater than 2. Only biphasic materials are accepted (e.g. cort+trab)."
            )

        img_size = image.GetSize()
        img_spacing = image.GetSpacing()

        coordsX = np.arange(
            0,
            img_size[1] * img_spacing[1],
            img_size[1] * img_spacing[1] / float(img_size[1]),
        )
        coordsY = np.arange(
            0,
            img_size[2] * img_spacing[2],
            img_size[2] * img_spacing[2] / float(img_size[2]),
        )
        self.coordsX, self.coordsY = np.meshgrid(coordsX, coordsY)
        self.coordsX = np.transpose(self.coordsX, [1, 0])
        self.coordsY = np.transpose(self.coordsY, [1, 0])
        return contour_ext, contour_int

    def sort_xy(self, x, y):
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

        fig.update_xaxes(range=[0, 40])
        fig.update_yaxes(range=[0, 40])
        fig.show()
        return fig

    def check_orient(self, x, y, direction=1):
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
                logging.debug("Desired direction: cw")
            if y[1] > y[0]:
                if self.debug_orientation == 1:
                    logging.debug("Not flipping")
                else:
                    pass
                x_o = x
                y_o = y
            elif y[1] < y[0]:
                if self.debug_orientation == 1:
                    logging.debug("Flipping")
                else:
                    pass
                x_o = np.flip(x, axis=0)
                y_o = np.flip(y, axis=0)
            else:
                logging.debug("Something went wrong while flipping the array")

        if direction == 2:
            if self.debug_orientation == 1:
                logging.debug("Desired direction: ccw")
            if y[1] < y[0]:
                if self.debug_orientation == 1:
                    logging.debug("Not flipping")
                x_o = x
                y_o = y
            elif y[1] > y[0]:
                if self.debug_orientation == 1:
                    logging.debug("Flipping")
                x_o = np.flip(x, axis=0)
                y_o = np.flip(y, axis=0)
            else:
                logging.warning("Something went wrong while flipping the array")
        return x_o, y_o

    def sort_mahalanobis(self, data, metrics, start):
        dist_m = ss.distance.squareform(ss.distance.pdist(data.T, metrics))
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

    def sort_surface(self, image_slice):
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
        x_mahalanobis, y_mahalanobis = self.sort_mahalanobis(
            xy_sorted.T, "mahalanobis", 0
        )
        x_mahalanobis = np.append(x_mahalanobis, x_mahalanobis[0])
        y_mahalanobis = np.append(y_mahalanobis, y_mahalanobis[0])

        # find the knot points
        tckp, u = splprep(
            [x_mahalanobis, y_mahalanobis],
            s=self.S,
            k=self.K,
            per=True,
            ub=[x_mahalanobis, y_mahalanobis][0],
            ue=[x_mahalanobis, y_mahalanobis][0],
        )

        # evaluate spline, including interpolated points
        xnew, ynew = splev(np.linspace(0, 1, self.INTERP_POINTS_S), tckp)
        xnew = np.append(xnew, xnew[0])
        ynew = np.append(ynew, ynew[0])

        # Sanity check to ensure directionality of sorting in cw- or ccw-direction
        xnew, ynew = self.check_orient(xnew, ynew, direction=1)
        return (
            xy_sorted_closed,
            xnew[1:],
            ynew[1:],
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
            logging.warning(
                f"Error in the shape of the external and internal contours of the slice:\t{slice}"
            )
            sys.exit(90)
        return ext_contour_s, int_contour_s

    def output_sanity_check(self, initial_contour: np.ndarray, contour_s: np.ndarray):
        # check that after csc.CorticalSanityCheck elements of arrays external and
        # internal contours have the same structure and shape as before csc.CorticalSanityCheck
        # sanity check of input data ext_contour_s and int_contour_s
        if np.allclose(initial_contour[0], initial_contour[1], rtol=1e-05, atol=1e-08):
            logging.warning("External contour has a duplicate first point")
            if not np.allclose(contour_s[0], contour_s[1], rtol=1e-05, atol=1e-08):
                logging.warning(
                    "New external contour does not have a duplicate first point"
                )
                contour_s = np.insert(contour_s, 0, contour_s[0], axis=0)
                logging.info("New external contour now has a duplicate first point")
        if np.allclose(initial_contour[-1], initial_contour[-2]):
            logging.warning("External contour has a duplicate last point")
            if not np.allclose(contour_s[-1], contour_s[-2], rtol=1e-05, atol=1e-08):
                logging.warning(
                    "New external contour does not have a duplicate last point"
                )
                contour_s = np.append(contour_s, [contour_s[-1]], axis=0)
                logging.info("New external contour now has a duplicate last point")
        if np.shape(initial_contour) != np.shape(contour_s):
            logging.warning(
                "External contour has a different shape than the initial contour"
            )
        else:
            logging.log(
                logging.INFO,
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

    def volume_splines(self):
        contour_ext_fig = []
        contour_int_fig = []
        contour_ext_fig, contour_int_fig = self.binary_threshold(
            contour_ext_fig, contour_int_fig, img_path=self.img_path
        )

        self.slice_index = np.linspace(
            1, len(contour_ext_fig[0, 0, :]) - 1, self.SLICING_COEFFICIENT, dtype=int
        )

        if self.show_plots is True:
            fig = go.Figure(layout=self.layout)
        else:
            logging.info(f"Volume_splines\t\tshow_plots:\t{self.show_plots}")
            fig = None

        # fmt: off
        if self.phases >= 1:
            img_contours_ext = sitk.GetImageFromArray(contour_ext_fig, isVector=True)
            image_data_ext = np.transpose(sitk.GetArrayViewFromImage(img_contours_ext), [2, 1, 0])
        if self.phases == 2:
            img_contours_int = sitk.GetImageFromArray(contour_int_fig, isVector=True)
            image_data_int = np.transpose(sitk.GetArrayViewFromImage(img_contours_int), [2, 1, 0])

        contour_ext = []
        contour_int = []
        for i, _slice in enumerate(self.slice_index):
            logging.debug(f"Slice:\t{_slice}")
            if self.phases >= 1:
                # TODO: check if ::-1 is still needed (minus sign)
                image_slice_ext = image_data_ext[_slice, :, :][::-1, ::-1]
                original, cortical_ext_x, cortical_ext_y = self.sort_surface(image_slice_ext)
                z = np.ones(len(cortical_ext_x)) * (self.spacing[0] * _slice)
                contour_ext = np.append(contour_ext, np.c_[cortical_ext_x, cortical_ext_y, z])

                if self.phases == 1:
                    if self.show_plots is True:
                        fig = self.plotly_add_traces(
                            fig, original, np.c_[contour_ext[:, 0], contour_ext[:, 1]]
                        )
                    else:
                        fig = None
            else:
                logging.warning(f"Phases is not >= 1: {self.phases}")

            if self.phases == 2:
                # TODO: check if ::-1 is still needed (minus sign)
                image_slice_int = image_data_int[_slice, :, :][::-1, ::-1]
                original, cortical_int_x, cortical_int_y = self.sort_surface(image_slice_int)
                contour_int = np.append(contour_int, np.c_[cortical_int_x, cortical_int_y, z])

                if self.phases == 2:
                    if self.show_plots is True:
                        fig = self.plotly_add_traces(
                            fig, original, np.c_[cortical_int_x, cortical_int_y, z]
                        )
                    else:
                        fig = None
            else:
                logging.warning(f"Phases =/= 2: {self.phases}")
        # fmt: on

        if self.show_plots is True:
            self.plotly_makefig(fig)
        elif self.show_plots is False:
            fig = None

        contour_ext = contour_ext.reshape(-1, 3)
        contour_int = contour_int.reshape(-1, 3)

        return contour_ext, contour_int


def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    logging.getLogger(os.getlogin())
    logging.info("Starting meshing script...")
    start = time.time()

    # fmt: off
    img_basefilename = ["C0002234"]
    cwd = os.getcwd()
    img_basepath = f"{cwd}/01_AIM"
    img_outputpath = f"{cwd}/04_OUTPUT"
    img_path_ext = [str(Path(img_basepath, img_basefilename[i], img_basefilename[i] + "_CORT_MASK_cap.mhd",)) for i in range(len(img_basefilename))]
    filepath_ext = [str(Path(img_basepath, img_basefilename[i], img_basefilename[i])) for i in range(len(img_basefilename))]
    filename = [str(Path(img_outputpath, img_basefilename[i], img_basefilename[i] + "_ext.geo_unrolled")) for i in range(len(img_basefilename))]

    geo_file_path = f"{cwd}/04_OUTPUT/C0002237/fake_example.geo_unrolled"
    mesh_file_path = f"{cwd}/04_OUTPUT/C0002237/C0002237.msh"

    for i in range(len(img_basefilename)):
        Path.mkdir(Path(img_outputpath, img_basefilename[i]), parents=True, exist_ok=True)
    # fmt: on

        cortical_v = OCC_volume(
            img_path_ext[i],
            filepath_ext[i],
            filename[i],
            ASPECT=30,
            SLICE=1,
            UNDERSAMPLING=5,
            SLICING_COEFFICIENT=10,
            INSIDE_VAL=0,
            OUTSIDE_VAL=1,
            LOWER_THRESH=0,
            UPPER_THRESH=0.9,
            S=5,
            K=3,
            INTERP_POINTS=50,
            debug_orientation=0,
            show_plots=False,
            location="cort_ext",
            offset=0,
            thickness_tol=180e-3,
            phases=2,
        )

        cortical_v.plot_mhd_slice()
        cortical_ext, cortical_int = cortical_v.volume_splines()
        # Cortical surface sanity check
        cortex = csc.CorticalSanityCheck(MIN_THICKNESS=cortical_v.MIN_THICKNESS,
                                         ext_contour=cortical_ext,
                                         int_contour=cortical_int,
                                         model=cortical_v.filename,
                                         save_plot=False)

        cortical_ext_split = np.array_split(cortical_ext, len(np.unique(cortical_ext[:, 2])))
        cortical_int_split = np.array_split(cortical_int, len(np.unique(cortical_int[:, 2])))
        cortical_int_sanity = np.zeros(np.shape(cortical_int_split))
        for i, _ in enumerate(cortical_ext_split):
            cortical_int_sanity[i][:, :-1] = cortex.cortical_sanity_check(ext_contour=cortical_ext_split[i], int_contour=cortical_int_split[i], iterator=i, show_plots=False)
            cortical_int_sanity[i][:, -1] = cortical_int_split[i][:, -1]
        cortical_int_sanity = cortical_int_sanity.reshape(-1, 3)

        gmsh.initialize()
        gmsh.clear()

        N_TRANSVERSE = 3
        N_RADIAL = 10
        mesher = Mesher(geo_file_path, mesh_file_path, slicing_coefficient=cortical_v.SLICING_COEFFICIENT, n_transverse=N_TRANSVERSE, n_radial=N_RADIAL)

        cortex_centroid = np.zeros((len(cortical_ext_split), 3))  # center of mass of each slice (x, y, z)
        cortical_int_sanity_split = np.array_split(cortical_int_sanity, len(np.unique(cortical_int_sanity[:, 2])))

        cortical_ext_centroid = np.zeros((np.shape(cortical_ext_split)[0], np.shape(cortical_ext_split)[1], np.shape(cortical_ext_split)[2]))
        cortical_int_centroid = np.zeros((np.shape(cortical_int_split)[0], np.shape(cortical_int_split)[1], np.shape(cortical_int_split)[2]))
        idx_list_ext = np.zeros((len(cortical_ext_split), 4), dtype=int)
        idx_list_int = np.zeros((len(cortical_ext_split), 4), dtype=int)
        intersections_ext = np.zeros((len(cortical_ext_split), 2, 2, 3), dtype=float)
        intersections_int = np.zeros((len(cortical_ext_split), 2, 2, 3), dtype=float)

        for i, _ in enumerate(cortical_ext_split):
            _, idx = np.unique(cortical_ext_split[i].round(decimals=6), axis=0, return_index=True)
            cortical_ext_split[i][np.sort(idx)]
            _, idx = np.unique(cortical_int_sanity_split[i].round(decimals=6), axis=0, return_index=True)
            cortical_int_sanity_split[i][np.sort(idx)]
            cortex_centroid[i][:-1] = mesher.polygon_tensor_of_inertia(cortical_ext_split[i], cortical_int_sanity_split[i])
            cortex_centroid[i][-1] = cortical_ext_split[i][0, -1]
            cortical_ext_centroid[i], idx_list_ext[i], intersections_ext[i] = mesher.insert_tensor_of_inertia(cortical_ext_split[i], cortex_centroid[i][:-1])
            cortical_int_centroid[i], idx_list_int[i], intersections_int[i] = mesher.insert_tensor_of_inertia(cortical_int_sanity_split[i], cortex_centroid[i][:-1])

        cortical_ext_msh = np.reshape(cortical_ext_centroid, (-1, 3))
        cortical_int_msh = np.reshape(cortical_int_centroid, (-1, 3))

        indices_coi_ext, cortical_ext_bspline, intersection_line_tags_ext, cortical_ext_surfs = mesher.gmsh_geometry_formulation(cortical_ext_msh, idx_list_ext)
        indices_coi_int, cortical_int_bspline, intersection_line_tags_int, cortical_int_surfs = mesher.gmsh_geometry_formulation(cortical_int_msh, idx_list_int)
        intersurface_line_tags = mesher.add_interslice_segments(indices_coi_ext, indices_coi_int)
        slices_tags = mesher.add_slice_surfaces(cortical_ext_bspline, cortical_int_bspline, intersurface_line_tags)
        intersurface_surface_tags = mesher.add_intersurface_planes(intersurface_line_tags, intersection_line_tags_ext, intersection_line_tags_int)

        intersection_line_tags = np.append(intersection_line_tags_ext, intersection_line_tags_int)
        cortical_bspline_tags = np.append(cortical_ext_bspline, cortical_int_bspline)
        cortical_surfs = np.concatenate((cortical_ext_surfs, cortical_int_surfs, slices_tags, intersurface_surface_tags), axis=None)

        cort_vol_tags = mesher.add_volume(cortical_ext_surfs, cortical_int_surfs, slices_tags, intersurface_surface_tags)

        # TODO: check if could be implemented when created (relationship with above functions)
        intersurface_line_tags = np.array(intersurface_line_tags, dtype=int).tolist()

        # add here trabecular meshing
        trabecular_volume = TrabecularVolume(geo_file_path, mesh_file_path, slicing_coefficient=cortical_v.SLICING_COEFFICIENT, n_transverse=N_TRANSVERSE, n_radial=N_RADIAL)
        trabecular_volume.set_length_factor(0.4)
        point_tags_r, trab_line_tags_v, trab_line_tags_h, trab_surfs_v, trab_surfs_h, trab_vols = trabecular_volume.get_trabecular_vol(coi_idx=intersections_int)

        # connection between inner trabecular and cortical volumes
        trab_cort_line_tags = mesher.trabecular_cortical_connection(coi_idx=indices_coi_int, trab_point_tags=point_tags_r)
        trab_slice_surf_tags = mesher.trabecular_slices(trab_cort_line_tags=trab_cort_line_tags, trab_line_tags_h=trab_line_tags_h, cort_int_bspline_tags=cortical_int_bspline)

        trab_plane_inertia_tags = trabecular_volume.trabecular_planes_inertia(trab_cort_line_tags, trab_line_tags_v, intersection_line_tags_int)
        cort_trab_vol_tags = mesher.get_trabecular_cortical_volume_mesh(trab_slice_surf_tags, trab_plane_inertia_tags, cortical_int_surfs, trab_surfs_v)

        # meshing
        trab_surfs = list(chain(trab_surfs_v, trab_surfs_h, trab_slice_surf_tags, trab_plane_inertia_tags))
        trabecular_volume.meshing_transfinite(trab_line_tags_v, trab_line_tags_h, trab_surfs, trab_vols, trab_cort_line_tags)

        volume_tags = np.concatenate((cort_vol_tags, cort_trab_vol_tags), axis=None)

        # * physical groups
        mesher.factory.synchronize()
        trab_vol_tags = np.concatenate((trab_vols, cort_trab_vol_tags), axis=None)
        trab_physical_group = mesher.model.addPhysicalGroup(3, trab_vol_tags)
        cort_physical_group = mesher.model.addPhysicalGroup(3, cort_vol_tags)

        mesher.meshing_transfinite(intersection_line_tags, cortical_bspline_tags, cortical_surfs, volume_tags, test_list=intersurface_line_tags)
        mesher.mesh_generate(dim=3)

        gmsh.fltk.run()
        gmsh.finalize()
        end = time.time()
        elapsed = round(end - start, ndigits=3)
        logging.info(f"Elapsed time:  {elapsed} (s)")
        logging.info("Meshing script finished.")


if __name__ == "__main__":
    main()
