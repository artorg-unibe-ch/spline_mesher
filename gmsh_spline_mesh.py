'''
Geometry representation and meshing through spline reconstruction
Author: Simone Poncioni, MSB
Date: 07-09.2022
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import SimpleITK as sitk
from scipy.interpolate import splev, splprep
from pathlib import Path
import matplotlib
import scipy.spatial as ss
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gmsh
import sys
import logging
import cortical_sanity as csc

pio.renderers.default = 'browser'
logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# rcParams
f = 1
plt.rcParams['figure.figsize'] = [8 * f, 8 * f]
plt.rcParams['font.size'] = 15
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


class OCC_volume():
    def __init__(self,
                 img_path, filepath, filename,
                 ASPECT, SLICE, UNDERSAMPLING, SLICING_COEFFICIENT,
                 INSIDE_VAL, OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH,
                 S, K, INTERP_POINTS,
                 debug_orientation,
                 show_plots,
                 location,
                 offset,
                 ext_contour,
                 thickness_tol):
        '''
        Class that imports a voxel-based model and converts it to a geometrical simplified representation
        through the use of splines for each slice in the transverse plane.
        Following spline reconstruction, the model is meshed with GMSH.
        '''
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
        self.THRESHOLD_PARAM = [INSIDE_VAL,
                                OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH]
        self.UNDERSAMPLING = UNDERSAMPLING
        self.SLICING_COEFFICIENT = SLICING_COEFFICIENT
        self.S = S
        self.K = K
        self.INTERP_POINTS_S = INTERP_POINTS
        self.contours_arr = []
        self.height = 1.0
        self.spacing = []
        self.coordsX = []
        self.coordsY = []
        self.xy_sorted_closed = []
        self.x_mahalanobis = []
        self.y_mahalanobis = []
        self.xnew = []
        self.ynew = []

        self.cortex_outer_tags = list()
        self.cortex_inner_tags = list()

        self.MIN_THICKNESS = float(thickness_tol)
        self.ext_contour = ext_contour

        # Figure layout
        self.layout = go.Layout(
            plot_bgcolor="#FFF",  # Sets background color to white
            xaxis=dict(
                title="Medial - Lateral position (mm)",
                linecolor="#BCCCDC",  # Sets color of X-axis line
                showgrid=True  # Removes X-axis grid lines
            ),
            yaxis=dict(
                title="Palmar - Dorsal position (mm)",
                linecolor="#BCCCDC",  # Sets color of Y-axis line
                showgrid=True,  # Removes Y-axis grid lines
            )
        )

    def plot_mhd_slice(self):
        """
        Helper function to plot a slice of the MHD file

        Returns:
            None
        """        
        if self.show_plots is not False:
            img = sitk.PermuteAxes(sitk.ReadImage(self.img_path), [1, 2, 0])
            img_view = sitk.GetArrayViewFromImage(img)

            plt.figure(f'Plot MHD slice n.{self.SLICE}',
                       figsize=(np.shape(img_view)[0]/self.ASPECT, np.shape(img_view)[1]/self.ASPECT))
            plt.imshow(img_view[self.SLICE, :, :], cmap='cividis',
                       interpolation='nearest', aspect='equal')
            plt.title(f'Slice n. {self.SLICE} of masked object', weight='bold')

            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(cax=cax)
            plt.show()
        else:
            print(f'MHD slice, show_plots:\t\t{self.show_plots}')
        return None

    def binary_threshold(self):
        '''
        THRESHOLD_PARAM = [INSIDE_VAL, OUTSIDE_VAL, LOWER_THRESH, UPPER_THRESH]
        '''

        image = sitk.ReadImage(self.img_path)
        # Binary threshold
        btif = sitk.BinaryThresholdImageFilter()
        btif.SetInsideValue(self.THRESHOLD_PARAM[0])
        btif.SetOutsideValue(self.THRESHOLD_PARAM[1])
        btif.SetLowerThreshold(self.THRESHOLD_PARAM[2])
        btif.SetUpperThreshold(self.THRESHOLD_PARAM[3])
        image_thr = btif.Execute(image)
        spacing_image = image.GetSpacing()

        # https://itk.org/pipermail/community/2017-August/013464.html
        img = sitk.JoinSeries([sitk.BinaryContour(
            image_thr[z, :, :], fullyConnected=True) for z in range(image_thr.GetSize()[0])])
        depth = img.GetDepth()
        img = sitk.PermuteAxes(img, [2, 0, 1])
        img.SetSpacing([spacing_image[0], spacing_image[1], spacing_image[2]])

        if self.show_plots is not False:
            plt.figure('Binary contour', figsize=(np.shape(sitk.GetArrayFromImage(image_thr))[
                       1]/self.ASPECT, np.shape(sitk.GetArrayFromImage(image_thr))[2]/self.ASPECT))
            plt.imshow(sitk.GetArrayViewFromImage(img)[
                       :, :, self.SLICE], cmap='gray', interpolation='None', aspect='equal')
            plt.title(
                f'Binary threshold on slice n. {self.SLICE}', weight='bold')
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(cax=cax)
            plt.show()
        else:
            print(f'Binary threshold, show_plots:\t{self.show_plots}')

        # https://stackoverflow.com/questions/25733694/process-image-to-find-external-contour
        self.spacing = img.GetSpacing()
        size = img.GetSize()

        self.contours_arr = []
        for z in range(depth):
            bin_img = sitk.GetArrayViewFromImage(img)[:, :, z]
            if self.location == 'cort_ext':
                contours, hierarchy = cv2.findContours(
                    bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                out = np.zeros_like(bin_img)
                outer_contour = cv2.drawContours(out, contours, -1, 1, 1)
                self.contours_arr.append(outer_contour)
            elif self.location == 'cort_int':
                contours, hierarchy = cv2.findContours(
                    bin_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                inn = np.zeros_like(bin_img)
                inner_contour = cv2.drawContours(inn, contours, 2, 1, 1)
                self.contours_arr.append(inner_contour)
            elif self.location == 'trab_ext':
                contours, hierarchy = cv2.findContours(
                    bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                out = np.zeros_like(bin_img)
                outer_contour = cv2.drawContours(out, contours, -1, 1, 1)
                self.contours_arr.append(outer_contour)
            else:
                raise ValueError(
                    'location argument is set neither to cort_external nor cort_internal')

        img_contours = sitk.GetImageFromArray(self.contours_arr, isVector=True)
        image_data = sitk.GetArrayViewFromImage(img_contours)
        self.height = depth * self.spacing[0]

        if self.show_plots is not False:
            plt.figure('Binary contour - only external',
                       figsize=(np.shape(sitk.GetArrayFromImage(image_thr))[1]/self.ASPECT,
                                np.shape(sitk.GetArrayFromImage(image_thr))[2]/self.ASPECT))
            plt.imshow(image_data[self.SLICE, :, :], cmap='gray',
                       interpolation='None', aspect='equal')
            plt.title(
                f'External contouring on slice n. {self.SLICE}', weight='bold')
            ax = plt.gca()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(cax=cax)
            plt.show()
        else:
            print(f'Binary threshold, show_plots:\t{self.show_plots}')

        coordsX = np.arange(0, size[0] * self.spacing[0], size[0]
                            * self.spacing[0] / float(image_data.shape[2]))
        coordsY = np.arange(0, size[1] * self.spacing[1], size[1]
                            * self.spacing[1] / float(image_data.shape[1]))
        self.coordsX, self.coordsY = np.meshgrid(coordsX, coordsY)

        return None

    def sort_xy(self, x, y):
        # https://stackoverflow.com/questions/58377015/counterclockwise-sorting-of-x-y-data

        x0 = np.mean(x)
        y0 = np.mean(y)
        r = np.sqrt((x-x0)**2 + (y-y0)**2)

        angles = np.where((y-y0) > 0, np.arccos((x-x0)/r),
                          2*np.pi-np.arccos((x-x0)/r))
        mask = np.argsort(angles)

        x_sorted = x[mask]
        y_sorted = y[mask]
        return x_sorted, y_sorted

    def plotly_add_traces(self, fig, xy_sorted_closed, x_mahalanobis, y_mahalanobis, xnew, ynew):
        fig.add_traces([
            go.Scatter(
                mode='lines+markers',
                marker=dict(
                    color='Black',
                    size=3),
                visible=False,
                line=dict(color=px.colors.qualitative.Dark2[0], width=2),
                name="Original",
                x=xy_sorted_closed[:, 0],
                y=xy_sorted_closed[:, 1]),

            go.Scatter(
                visible=False,
                mode='lines',
                line=dict(color=px.colors.qualitative.Dark2[2], width=5),
                name="Mahalanobis sorting",
                x=x_mahalanobis,
                y=y_mahalanobis),

            go.Scatter(
                visible=False,
                line=dict(color=px.colors.qualitative.Dark2[1], width=5),
                name=f"B-spline order {self.K}",
                x=xnew,
                y=ynew),
        ])
        return fig

    def plotly_makefig(self, fig):
        img_contours = sitk.GetImageFromArray(
            self.contours_arr, isVector=True)  # TODO: how to pass this one?
        image_data = sitk.GetArrayViewFromImage(img_contours)
        fig.data[len(image_data)//(2 * self.SLICING_COEFFICIENT)
                 ].visible = True

        # Create and add slider
        steps = []
        for i in range(len(fig.data)):
            if i % 3 == 0:
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)},
                          {'title': f'Slice position {(((i) * self.height) / len(fig.data)):.2f} mm'}],
                    label=i
                )
                # Toggle i'th trace to "visible"
                step["args"][0]["visible"][i] = True
                step["args"][0]["visible"][i+1] = True
                step["args"][0]["visible"][i+2] = True
                steps.append(step)

        sliders = [dict(
            active=10,
            currentvalue={"prefix": "Slice: ", "suffix": " [-]"},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            autosize=False,
            width=1000,
            height=1000)

        fig.add_annotation(text="Slice representation through splines",
                           xref="paper", yref="paper",
                           x=0.1, y=1, showarrow=False,
                           font=dict(size=18, family="stix"))

        fig.update_xaxes(range=[0, 20])
        fig.update_yaxes(range=[0, 35])
        fig.show()
        return fig

    def check_orient(self, x, y, direction=1):
        '''
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
        '''

        if direction == 1:
            if self.debug_orientation == 1:
                print('Desired direction: cw')
            if y[1] > y[0]:
                if self.debug_orientation == 1:
                    print('Not flipping')
                else:
                    pass
                x_o = x
                y_o = y
            elif y[1] < y[0]:
                if self.debug_orientation == 1:
                    print('Flipping')
                else:
                    pass
                x_o = np.flip(x, axis=0)
                y_o = np.flip(y, axis=0)
            else:
                print('Something went wrong while flipping the array')

        if direction == 2:
            if self.debug_orientation == 1:
                print('Desired direction: ccw')
            if y[1] < y[0]:
                if self.debug_orientation == 1:
                    print('Not flipping')
                x_o = x
                y_o = y
            elif y[1] > y[0]:
                if self.debug_orientation == 1:
                    print('Flipping')
                x_o = np.flip(x, axis=0)
                y_o = np.flip(y, axis=0)
            else:
                print('Something went wrong while flipping the array')

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
            # print points_index, target, nneigbour
            sorted_index.append(target)
            target = nneigbour
        sorted_index.append(target)

        x_mahalanobis = data[0][sorted_index]
        y_mahalanobis = data[1][sorted_index]
        return x_mahalanobis, y_mahalanobis

    def sort_surface(self, slices):
        """
        Sort surface points in a clockwise direction and with mahalanobis distance to add robustness


        Args:
            slices (_type_): _description_

        Returns:
            self.xy_sorted_closed (numpy.ndarray): xy array of sorted points
            self.x_mahanalobis (numpy.ndarray): x array of sorted points with mahalanobis distance
            self.y_mahanalobis (numpy.ndarray): y array of sorted points with mahalanobis distance
            self.xnew (numpy.ndarray): x array of sorted points interpolated with bspline and in the same direction
            self.ynew (numpy.ndarray): y array of sorted points interpolated with bspline and in the same direction
        """
        img_contours = sitk.GetImageFromArray(
            self.contours_arr, isVector=True)  # TODO: how to pass this one?
        image_data = sitk.GetArrayViewFromImage(img_contours)
        image_slice = image_data[slices, :, :][::-1, ::-1]

        x = self.coordsX[image_slice == 1][0::self.UNDERSAMPLING]
        y = self.coordsY[image_slice == 1][0::self.UNDERSAMPLING]

        x_s, y_s = self.sort_xy(x, y)
        self.xy_sorted = np.c_[x_s, y_s]
        self.xy_sorted_closed = np.vstack([self.xy_sorted, self.xy_sorted[0]])
        self.x_mahalanobis, self.y_mahalanobis = self.sort_mahalanobis(
            self.xy_sorted.T, 'mahalanobis', 0)
        self.x_mahalanobis = np.append(
            self.x_mahalanobis, self.x_mahalanobis[0])
        self.y_mahalanobis = np.append(
            self.y_mahalanobis, self.y_mahalanobis[0])

        # find the knot points
        tckp, u = splprep([self.x_mahalanobis, self.y_mahalanobis], s=self.S, k=self.K, per=True, ub=[
                          self.x_mahalanobis, self.y_mahalanobis][0], ue=[self.x_mahalanobis, self.y_mahalanobis][0])

        # evaluate spline, including interpolated points
        self.xnew, self.ynew = splev(
            np.linspace(0, 1, self.INTERP_POINTS_S), tckp)
        self.xnew = np.append(self.xnew, self.xnew[0])
        self.ynew = np.append(self.ynew, self.ynew[0])

        # Sanity check to ensure directionality of sorting in cw- or ccw-direction
        self.xnew, self.ynew = self.check_orient(
            self.xnew, self.ynew, direction=1)
        return self.xy_sorted_closed, self.x_mahalanobis, self.y_mahalanobis, self.xnew, self.ynew

    def surfaces_gmsh(self, x, y, z):
        '''
        https://gitlab.onelab.info/gmsh/gmsh/-/issues/456
        https://bbanerjee.github.io/ParSim/fem/meshing/gmsh/quadrlateral-meshing-with-gmsh/
        '''
        gmsh.option.setNumber("General.Terminal", 1)

        points = []
        if self.location == 'cort_ext':
            for i in range(1, len(x)-1):
                point = self.factory.addPoint(x[i], y[i], z, tag=-1)
                points = np.append(points, point)
        elif self.location == 'cort_int':
            for i in range(1, len(x)-1):
                point = self.factory.addPoint(x[i], y[i], z, tag=-1)
                points = np.append(points, point)
        elif self.location == 'trab_ext':
            for i in range(1, len(x)-1):
                point = self.factory.addPoint(x[i], y[i], z, tag=-1)
                points = np.append(points, point)
        else:
            sys.exit(91, 'self.location not defined correctly')

        loop = []
        loop = np.linspace(points[0], points[-1],
                           num=self.INTERP_POINTS_S, dtype='int')
        loop = np.append(loop, loop[0])

        lines = []
        for i in range(1, len(loop)-1):
            line_tag = self.factory.addLine(
                startTag=loop[i], endTag=loop[i+1], tag=-1)
            lines.append(line_tag)

        curve_loop_tag = self.factory.addCurveLoop(lines, tag=-1)
        return lines, points, curve_loop_tag

    def connector_gmsh(self, points, lines_slices, slice_index):
        loop_v = np.linspace(points[0][0], points[-1]
                             [0], num=len(points), dtype='int')
        loop_h = np.linspace(points[0][0], points[0]
                             [-1], num=np.ma.size(points, 1), dtype='int')

        lines_connectors = []
        for i in range(0, len(loop_v)-1):
            for j in range(0, len(loop_h)):
                line = self.factory.addLine(
                    startTag=points[i][j], endTag=points[i+1][j], tag=-1)
                lines_connectors.append(line)

        lines_slices = np.ndarray.astype(lines_slices, dtype='int')
        lines_tag_connectors = np.ndarray.astype(np.array(lines_connectors).reshape(
            [len(slice_index) - 1, self.INTERP_POINTS_S - 1]), dtype='int')

        # append first element of each array
        points = np.c_[points, points[:, 0]]
        # append first element of each array
        lines_tag_connectors = np.c_[
            lines_tag_connectors, lines_tag_connectors[:, 0]]

        surf_ext_tag_l = []
        for i in range(0, len(loop_v)-1):
            for j in range(0, len(loop_h)):
                print(
                    f'Connecting lines: {points[i][j]} {lines_tag_connectors[i][j]}{points[i+1][j]} {lines_tag_connectors[i][j+1]}')
                ll = self.factory.addCurveLoop(
                    [points[i][j],  lines_tag_connectors[i][j],  points[i+1][j],  lines_tag_connectors[i][j+1]], tag=-1)
                surf_ext_tag_l.append(ll)

        return lines_connectors, surf_ext_tag_l

    def add_surface_connectors(self, surf_ext_tags):
        shell_tags = []
        for ll in surf_ext_tags:
            surf_ext_tag = self.factory.addSurfaceFilling(ll, tag=-1)
            shell_tags.append(surf_ext_tag)

        return shell_tags

    def add_volume(self, surf_b, surf_tags, surf_t, loc):
        loop = self.factory.addSurfaceLoop(
            [surf_b] + surf_tags + [surf_t], tag=-1)

        if loc == 'cort_ext':
            tag_v = int(3)
        elif loc == 'cort_int':
            tag_v = int(2)
        elif loc == 'trab':
            tag_v = int(1)
        else:
            print(f'Error in the definition of the volume tag:\t{tag_v}')
            sys.exit(90)

        tag_vol = self.factory.addVolume([loop], tag=-1)
        # self.model.addPhysicalGroup(dim=3, tags=[tag_vol], tag=-1, name='cort_ext')  # TODO: solve physical names issue

        return tag_vol

    def input_sanity_check(self, ext_contour_s:np.ndarray, int_contour_s:np.ndarray):
        """
        Sanity check for the input data before cortical sanity check

        Args:
            ext_contour_s (np.ndarray): array of external contour points
            int_contour_s (np.ndarray): array of internal contour points

        Returns:
            ext_contour_s (np.ndarray): array of external contour points with defined shape and structure (no duplicates, closed contour)
            int_contour_s (np.ndarray): array of internal contour points with defined shape and structure (no duplicates, closed contour)
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
            print(
                f'Error in the shape of the external and internal contours of the slice:\t{slice}')
            sys.exit(90)
        return ext_contour_s, int_contour_s

    def output_sanity_check(self, initial_contour:np.ndarray, contour_s:np.ndarray):
        # check that after csc.CorticalSanityCheck elements of arrays external and internal contours have the same structure and shape as before csc.CorticalSanityCheck
        # sanity check of input data ext_contour_s and int_contour_s
        if np.allclose(initial_contour[0], initial_contour[1], rtol=1e-05, atol=1e-08):
            logging.warning(f'External contour has a duplicate first point')
            if not np.allclose(contour_s[0], contour_s[1], rtol=1e-05, atol=1e-08):
                logging.warning(f'New external contour does not have a duplicate first point')
                contour_s = np.insert(contour_s, 0, contour_s[0], axis=0)
                logging.info(f'New external contour now has a duplicate first point')
        if np.allclose(initial_contour[-1], initial_contour[-2]):
            logging.warning(f'External contour has a duplicate last point')
            if not np.allclose(contour_s[-1], contour_s[-2], rtol=1e-05, atol=1e-08):
                logging.warning(f'New external contour does not have a duplicate last point')
                contour_s = np.append(contour_s, [contour_s[-1]], axis=0)
                logging.info(f'New external contour now has a duplicate last point')
        if np.shape(initial_contour) != np.shape(contour_s):
            logging.warning(f'External contour has a different shape than the initial contour')
        else:
            logging.log(logging.INFO, 'External contour has the same shape as before csc.CorticalSanityCheck')
        print(f'contour_s after output_sanity_check:\n{contour_s}')
        return contour_s

    def offset_tags(self, entities):
        '''
        get all entities of the model and add offset to make unique tags
        '''
        for tag in entities:
            newTag_s = (tag[0], tag[1] + self.offset)
            gmsh.model.setTag(dim=tag[0], tag=tag[1], newTag=newTag_s[1])
        return None

    def volume_splines(self):
        self.binary_threshold()
        img_contours = sitk.GetImageFromArray(self.contours_arr, isVector=True)
        image_data = sitk.GetArrayViewFromImage(img_contours)
        slice_index = np.arange(1, len(image_data), self.SLICING_COEFFICIENT)
        # TODO: Redundancy in slice_index and get_image in OCC_volume.volume_splines()
        # labels: question
        # assignees: @simoneponcioni
        # milestone: 0.1.0     
        get_image = sitk.GetImageFromArray(image_data)
        height = get_image.GetDepth() * self.spacing[0]

        if self.show_plots is not False:
            # Create figure
            fig = go.Figure(layout=self.layout)
        else:
            print(f'Volume_splines, show_plots:\t{self.show_plots}')

        # gmsh initialization
        gmsh.initialize()
        gmsh.model.add(str(self.location))

        connector_arr = []
        lines_slices = []
        surfaces_slices = []
        ext_contour_dstack = []
        for i, slice in enumerate(slice_index):
            z_pos = height * i / (len(slice_index) - 1)
            xy_sorted_closed, x_mahalanobis, y_mahalanobis, xnew, ynew = self.sort_surface(
                slices=slice)
            
            if self.location == 'cort_ext':
                dstack = np.dstack((xnew, ynew))
                ext_contour_dstack.append(dstack)

            if self.location == 'cort_int':
                ext_contour_t = self.ext_contour[i]
                int_contour_t = np.c_[xnew, ynew]

                ext_contour_s, int_contour_s = self.input_sanity_check(ext_contour_t, int_contour_t)

                cortex = csc.CorticalSanityCheck(MIN_THICKNESS=self.MIN_THICKNESS,
                                                     ext_contour=ext_contour_s,
                                                     int_contour=int_contour_s)
                
                
                int_spline_corr = cortex.cortical_sanity_check(ext_contour=ext_contour_s,
                                                               int_contour=int_contour_s)

                # check that after csc.CorticalSanityCheck elements of arrays external and internal contours have the same structure and shape as before csc.CorticalSanityCheck
                int_contour_s = self.output_sanity_check(int_contour_t, int_spline_corr)
                xnew = int_contour_s[:, 0]
                ynew = int_contour_s[:, 1]

            # lines_s, points, curve_loop_tag = self.surfaces_gmsh(x=xnew, y=ynew, z=z_pos)
            lines_s, points, curve_loop_tag = self.surfaces_gmsh(x=xnew, y=ynew, z=z_pos)
            surfaces_slices.append(curve_loop_tag)
            lines_slices = np.append(lines_slices, lines_s)
            connector_arr = np.append(connector_arr, points)
            if self.show_plots is not False:
                fig = self.plotly_add_traces(
                    fig, xy_sorted_closed, x_mahalanobis, y_mahalanobis, xnew, ynew)
            else:
                pass

        if self.show_plots is not False:
            fig = self.plotly_makefig(fig)
        else:
            pass

        # Create gmsh connectors
        connectors_r = np.ndarray.astype(connector_arr.reshape(
            [len(slice_index), self.INTERP_POINTS_S - 1]), dtype='int')
        lines, surf_ext_tags_s = self.connector_gmsh(
            connectors_r, lines_slices, slice_index=slice_index)

        shell_tags = self.add_surface_connectors(surf_ext_tags=surf_ext_tags_s)
        surfaces_first = self.factory.addPlaneSurface(
            [surfaces_slices[0]], tag=-1)
        surfaces_last = self.factory.addPlaneSurface(
            [surfaces_slices[-1]], tag=-1)

        if self.location == 'cort_ext':
            self.cortex_outer_tags = self.add_volume(
                surfaces_first, shell_tags, surfaces_last, self.location)
        elif self.location == 'cort_int':
            self.cortex_inner_tags = self.add_volume(
                surfaces_first, shell_tags, surfaces_last, self.location)

        self.factory.synchronize()
        self.offset_tags(self.factory.getEntities())

        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(self.filepath + self.filename))
        logging.info('GMSH file saved')

        if self.show_plots is not False:
            gmsh.fltk.run()
        gmsh.clear()
        gmsh.finalize()
        print('Exiting GMSH...')

        # reduce nesting level of ext_contour_dstack
        ext_contour_dstack = np.squeeze(ext_contour_dstack)
        return ext_contour_dstack, str(self.filepath + self.filename)

    def cortical_volume(self):
        '''
        Import external cortical volume and substract internal cortical volume with GMSH boolean operations
        '''

        self.factory.cut(objectDimTags=self.cortex_outer_tags,
                         toolDimTags=self.cortex_outer_tags,
                         removeObject=False,
                         removeTool=False)

        # self.outer_cortex_surface, self.inner_cortex_surface)

        if self.show_plots is not False:
            gmsh.fltk.run()
        gmsh.clear()
        gmsh.finalize()
        print('Exiting GMSH...')


class OCC_bool():
    def __init__(self,
                 cort_ext, cort_int, trab_ext,
                 bool_op,
                 filepath,
                 filename,
                 bool_name):

        self.model = gmsh.model
        self.factory = self.model.occ

        self.cortical_ext = str(cort_ext)
        self.cortical_int = str(cort_int)
        self.trabecular = str(trab_ext)
        self.bool_operation = str(bool_op)

        self.filepath = str(filepath)
        self.filename = str(filename)
        self.bool_name = str(bool_name)
        return None

    def volume_difference(self, objectDimTags_s, toolDimTags_s, bool_name):
        gmsh.initialize()
        gmsh.model.occ.add(bool_name)
        gmsh.merge(objectDimTags_s)
        gmsh.merge(toolDimTags_s)

        self.factory.cut(objectDimTags=objectDimTags_s,
                         toolDimTags=toolDimTags_s,
                         tag=-1,
                         removeObject=True, removeTool=True)

        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(self.filepath + self.filename))

        if self.show_plots is not False:
            gmsh.fltk.run()
        gmsh.clear()
        gmsh.finalize()
        print('Exiting GMSH...')


def main():
    img_path_ext = r'/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/01_AIM/C0002231_CORT_MASK_cap01.mhd'
    filepath_ext = r'/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/99_testing_prototyping/'
    img_path_trab = r'/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/01_AIM/C0002231_TRAB_MASK_cap02.mhd'
    filename_ext = Path(img_path_ext).stem + '_ext2' + '.geo_unrolled'
    filename_int = Path(img_path_ext).stem + '_int2' + '.geo_unrolled'
    filename_trab_ext = Path(img_path_ext).stem + '_trab2' + '.geo_unrolled'
    interp_point_s = 100
    slicing_coeff_s = 50
    show_plots_s = False
    thickness_tol_s = 100e-3

    ext_cort_surface = OCC_volume(img_path_ext, filepath_ext, filename_ext,
                                  ASPECT=50, SLICE=5, UNDERSAMPLING=5, SLICING_COEFFICIENT=slicing_coeff_s,
                                  INSIDE_VAL=0, OUTSIDE_VAL=1, LOWER_THRESH=0, UPPER_THRESH=0.9,
                                  S=10, K=3, INTERP_POINTS=interp_point_s,
                                  debug_orientation=0,
                                  show_plots=show_plots_s,
                                  location='cort_ext',
                                  offset=10000,
                                  ext_contour=None,
                                  thickness_tol=thickness_tol_s)
    # ext_cort_surface.plot_mhd_slice()
    cort_ext_arr, cort_ext_vol = ext_cort_surface.volume_splines()

    int_cort_surface = OCC_volume(img_path_ext, filepath_ext, filename_int,
                                  ASPECT=50, SLICE=5, UNDERSAMPLING=5, SLICING_COEFFICIENT=slicing_coeff_s,
                                  INSIDE_VAL=0, OUTSIDE_VAL=1, LOWER_THRESH=0, UPPER_THRESH=0.9,
                                  S=10, K=3, INTERP_POINTS=interp_point_s,
                                  debug_orientation=0,
                                  show_plots=show_plots_s,
                                  location='cort_int',
                                  offset=20000,
                                  ext_contour=cort_ext_arr,
                                  thickness_tol=thickness_tol_s)
    # int_cort_surface.plot_mhd_slice()
    cort_int_arr, cort_int_vol = int_cort_surface.volume_splines()

    # ext_trab_surface = OCC_volume(img_path_trab, filepath_ext, filename_trab_ext,
    #                               ASPECT=50, SLICE=5, UNDERSAMPLING=5, SLICING_COEFFICIENT=slicing_coeff_s,
    #                               INSIDE_VAL=0, OUTSIDE_VAL=1, LOWER_THRESH=0, UPPER_THRESH=0.9,
    #                               S=10, K=3, INTERP_POINTS=interp_point_s,
    #                               debug_orientation=0,
    #                               show_plots=show_plots_s,
    #                               location='trab_ext',
    #                               offset=30000,
    #                               ext_contour=None)
    # # int_cort_surface.plot_mhd_slice()
    # trab_ext_arr, trab_ext_vol = ext_trab_surface.volume_splines()


if __name__ == "__main__":
    logging.info('Starting meshing script...')
    print('Executing gmsh_spline_mesh.py')
    main()
    logging.info('Meshing script finished.')
