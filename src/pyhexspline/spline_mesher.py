import os
from typing import Dict, Optional, Tuple

from numpy import float64, ndarray, uint64
from SimpleITK.SimpleITK import Image

os.environ["NUMEXPR_MAX_THREADS"] = "16"

import logging
import pickle
import sys
import time
from itertools import chain
from pathlib import Path

import gmsh
import numpy as np
import plotly.io as pio
import SimpleITK as sitk

from pyhexspline import cortical_sanity as csc
from pyhexspline.gmsh_mesh_builder import Mesher, TrabecularVolume
from pyhexspline.quad_refinement import QuadRefinement
from pyhexspline.spline_volume import OCC_volume

pio.renderers.default = "browser"
LOGGING_NAME = "MESHING"
# flake8: noqa: E402


class HexMesh:
    def __init__(
        self,
        settings_dict: dict,
        img_dict: dict,
        sitk_image: Optional[Image] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.settings_dict = settings_dict
        self.img_dict = img_dict
        self.logger = logger
        if sitk_image is not None:
            self.sitk_image = sitk_image  # imports the image
        else:
            self.sitk_image = None  # reads the image from img_path_ext

    def mesher(
        self,
    ) -> Tuple[
        Dict[uint64, ndarray],
        Dict[uint64, ndarray],
        int,
        Dict[int, ndarray],
        Dict[int, ndarray],
        ndarray,
        ndarray,
        float64,
        float64,
        Dict[uint64, ndarray],
        Dict[uint64, ndarray],
        ndarray,
    ]:
        """
        Perform the meshing process for cortical and trabecular volumes.

        This function is the entry point for the meshing process, which includes
        reading image data, generating splines, performing sanity checks, and
        creating the mesh using Gmsh. It also handles the configuration of various
        meshing parameters and settings.

        Returns:
            Tuple: A tuple containing the following elements:
                - nodes (Dict[uint64, ndarray]): Dictionary of node tags.
                - elms (Dict[uint64, ndarray]): Dictionary of element tags.
                - nb_nodes (int): Number of nodes in the model.
                - centroids_cort_dict (Dict[int, ndarray]): Dictionary of cortical element centroids.
                - centroids_trab_dict (Dict[int, ndarray]): Dictionary of trabecular element centroids.
                - elm_vol_cort (ndarray): Array of cortical element volumes.
                - elm_vol_trab (ndarray): Array of trabecular element volumes.
                - radius_roi_cort (float64): Radius of the largest cortical ROI in mm.
                - radius_roi_trab (float64): Radius of the largest trabecular ROI in mm.
                - bnds_bot (Dict[uint64, ndarray]): Dictionary of bottom boundary nodes.
                - bnds_top (Dict[uint64, ndarray]): Dictionary of top boundary nodes.
                - reference_point_coord (ndarray): Coordinates of the reference point in mm.

        Raises:
            AssertionError: If the number of volumes and centroids does not match.
            RuntimeError: If negative element volumes are detected, which implies that the simulation would fail.
        """
        logger = logging.getLogger(LOGGING_NAME)
        logger.info("Starting meshing script...")
        start = time.time()

        img_basepath = self.img_dict["img_basepath"]
        img_basefilename = self.img_dict["img_basefilename"]
        str(Path(img_basepath, img_basefilename, img_basefilename))

        if self.sitk_image is None:
            img_path_ext = str(
                Path(
                    img_basepath,
                    img_basefilename,
                    img_basefilename + "_CORT_MASK_cap.mhd",
                )
            )
            sitk_image = None
        else:
            img_path_ext = None
            sitk_image = self.sitk_image

        filepath_ext = str(Path(img_basepath, img_basefilename, img_basefilename))
        geo_unrolled_filename = str(
            Path(
                self.img_dict["outputpath"],
                img_basefilename,
                img_basefilename + ".geo_unrolled",
            )
        )

        mesh_file_path = str(
            Path(
                self.img_dict["meshpath"],
                img_basefilename,
                img_basefilename,
            )
        )

        # Spline_volume settings
        ASPECT = int(self.settings_dict["aspect"])
        SLICE = int(self.settings_dict["_slice"])
        UNDERSAMPLING = int(self.settings_dict["undersampling"])
        SLICING_COEFFICIENT = int(self.settings_dict["slicing_coefficient"])
        INSIDE_VAL = float(self.settings_dict["inside_val"])
        OUTSIDE_VAL = float(self.settings_dict["outside_val"])
        LOWER_THRESH = float(self.settings_dict["lower_thresh"])
        UPPER_THRESH = float(self.settings_dict["upper_thresh"])
        S = int(self.settings_dict["s"])
        K = int(self.settings_dict["k"])
        INTERP_POINTS = int(self.settings_dict["interp_points"])
        DP_SIMPLIFICATION_OUTER = int(self.settings_dict["dp_simplification_outer"])
        DP_SIMPLIFICATION_INNER = int(self.settings_dict["dp_simplification_inner"])
        SHOW_PLOTS = bool(self.settings_dict["show_plots"])
        SHOW_GMSH = bool(self.settings_dict["show_gmsh"])
        WRITE_MESH = bool(self.settings_dict["write_mesh"])
        THICKNESS_TOL = float(self.settings_dict["thickness_tol"])
        PHASES = int(self.settings_dict["phases"])

        # spline_mesher settings
        N_LONGITUDINAL = int(
            self.settings_dict["n_elms_longitudinal"] // SLICING_COEFFICIENT
        )
        N_TRANSVERSE_CORT = int(self.settings_dict["n_elms_transverse_cort"])
        N_TRANSVERSE_TRAB = int(self.settings_dict["n_elms_transverse_trab"])
        N_RADIAL = int(self.settings_dict["n_elms_radial"] // 4)
        ELM_ORDER = int(self.settings_dict["mesh_order"])
        QUAD_REFINEMENT = bool(self.settings_dict["trab_refinement"])
        MESH_ANALYSIS = bool(self.settings_dict["mesh_analysis"])
        ELLIPSOID_FITTING = bool(self.settings_dict["ellipsoid_fitting"])

        # ! obscured from settings by design
        DEBUG_ORIENTATION = 0  # 0: no debug, 1: debug

        cortical_v = OCC_volume(
            sitk_image,
            img_path_ext,
            filepath_ext,
            geo_unrolled_filename,
            ASPECT=ASPECT,
            SLICE=SLICE,
            UNDERSAMPLING=UNDERSAMPLING,
            SLICING_COEFFICIENT=SLICING_COEFFICIENT,
            INSIDE_VAL=INSIDE_VAL,
            OUTSIDE_VAL=OUTSIDE_VAL,
            LOWER_THRESH=LOWER_THRESH,
            UPPER_THRESH=UPPER_THRESH,
            S=S,
            K=K,
            INTERP_POINTS=INTERP_POINTS,
            DP_SIMPLIFICATION_OUTER=DP_SIMPLIFICATION_OUTER,
            DP_SIMPLIFICATION_INNER=DP_SIMPLIFICATION_INNER,
            debug_orientation=DEBUG_ORIENTATION,
            show_plots=SHOW_PLOTS,
            thickness_tol=THICKNESS_TOL,
            phases=PHASES,
        )

        if cortical_v.sitk_image is None:
            img = sitk.PermuteAxes(sitk.ReadImage(cortical_v.img_path), [1, 2, 0])
        else:
            img = self.sitk_image
        if cortical_v.show_plots is True:
            cortical_v.plot_mhd_slice(img)
        else:
            logger.info(f"MHD slice\t\t\tshow_plots:\t{cortical_v.show_plots}")
        image_pad = cortical_v.pad_and_plot(img)
        cortical_ext, cortical_int = cortical_v.volume_splines_optimized(image_pad)

        # Cortical surface sanity check
        cortex = csc.CorticalSanityCheck(
            MIN_THICKNESS=cortical_v.MIN_THICKNESS,
            ext_contour=cortical_ext,
            int_contour=cortical_int,
            model=cortical_v.filename,
            save_plot=False,
            logger=logger,
        )
        z_unique = np.unique(cortical_ext[:, 2])
        cortical_ext_split = np.array_split(cortical_ext, len(z_unique))
        cortical_int_split = np.array_split(cortical_int, len(z_unique))
        cortical_int_sanity = np.zeros(np.shape(cortical_int_split))

        for i, _ in enumerate(cortical_ext_split):
            cortical_int_sanity[i][:, :-1] = cortex.cortical_sanity_check(
                ext_contour=cortical_ext_split[i],
                int_contour=cortical_int_split[i],
                iterator=i,
                show_plots=False,
            )
            cortical_int_sanity[i][:, -1] = cortical_int_split[i][:, -1]
        cortical_int_sanity = cortical_int_sanity.reshape(-1, 3)

        gmsh.initialize()
        gmsh.clear()
        gmsh.option.setNumber("General.NumThreads", 6)
        gmsh.logger.start()

        mesher = Mesher(
            geo_unrolled_filename,
            mesh_file_path,
            slicing_coefficient=cortical_v.SLICING_COEFFICIENT,
            n_longitudinal=N_LONGITUDINAL,
            n_transverse_trab=N_TRANSVERSE_TRAB,
            n_transverse_cort=N_TRANSVERSE_CORT,
            n_radial=N_RADIAL,
            ellipsoid_fitting=ELLIPSOID_FITTING,
        )
        cortex_centroid = np.zeros((len(cortical_ext_split), 3))
        cortical_int_sanity_split = np.array_split(
            cortical_int_sanity, len(np.unique(cortical_int_sanity[:, 2]))
        )
        cortical_ext_centroid = np.zeros(
            (
                np.shape(cortical_ext_split)[0],
                np.shape(cortical_ext_split)[1],
                np.shape(cortical_ext_split)[2],
            )
        )
        cortical_int_centroid = np.zeros(
            (
                np.shape(cortical_int_split)[0],
                np.shape(cortical_int_split)[1],
                np.shape(cortical_int_split)[2],
            )
        )
        idx_list_ext = np.zeros((len(cortical_ext_split), 4), dtype=int)
        idx_list_int = np.zeros((len(cortical_ext_split), 4), dtype=int)
        intersections_ext = np.zeros((len(cortical_ext_split), 2, 2, 3), dtype=float)
        intersections_int = np.zeros((len(cortical_ext_split), 2, 2, 3), dtype=float)

        for i, _ in enumerate(cortical_ext_split):

            cortex_centroid[i][:-1] = mesher.polygon_tensor_of_inertia(
                cortical_ext_split[i],
                cortical_int_sanity_split[i],
                true_coi=False,
            )
            cortex_centroid[i][-1] = cortical_ext_split[i][0, -1]
            (
                cortical_ext_centroid[i],
                idx_list_ext[i],
                intersections_ext_s,
            ) = mesher.insert_tensor_of_inertia(
                cortical_ext_split[i], cortex_centroid[i][:-1]
            )
            intersections_ext[i] = intersections_ext_s
            (
                cortical_int_centroid[i],
                idx_list_int[i],
                intersections_int_s,
            ) = mesher.insert_tensor_of_inertia(
                cortical_int_sanity_split[i], cortex_centroid[i][:-1]
            )
            intersections_int[i] = intersections_int_s

        cortical_ext_msh = np.reshape(cortical_ext_centroid, (-1, 3))
        cortical_int_msh = np.reshape(cortical_int_centroid, (-1, 3))
        (
            indices_coi_ext,
            cortical_ext_bspline,
            intersection_line_tags_ext,
            cortical_ext_surfs,
        ) = mesher.gmsh_geometry_formulation(cortical_ext_msh, idx_list_ext)
        (
            indices_coi_int,
            cortical_int_bspline,
            intersection_line_tags_int,
            cortical_int_surfs,
        ) = mesher.gmsh_geometry_formulation(cortical_int_msh, idx_list_int)
        intersurface_line_tags = mesher.add_interslice_segments(
            indices_coi_ext, indices_coi_int
        )
        slices_tags = mesher.add_slice_surfaces(
            cortical_ext_bspline, cortical_int_bspline, intersurface_line_tags
        )
        intersurface_surface_tags = mesher.add_intersurface_planes(
            intersurface_line_tags,
            intersection_line_tags_ext,
            intersection_line_tags_int,
        )
        intersection_line_tags = np.append(
            intersection_line_tags_ext, intersection_line_tags_int
        )
        cortical_bspline_tags = np.append(cortical_ext_bspline, cortical_int_bspline)
        cort_surfs = np.concatenate(
            (
                cortical_ext_surfs,
                cortical_int_surfs,
                slices_tags,
                intersurface_surface_tags,
            ),
            axis=None,
        )
        cort_vol_tags = mesher.add_volume(
            cortical_ext_surfs,
            cortical_int_surfs,
            slices_tags,
            intersurface_surface_tags,
        )

        intersurface_line_tags = np.array(intersurface_line_tags, dtype=int).tolist()

        # Trabecular meshing
        trabecular_volume = TrabecularVolume(
            geo_unrolled_filename,
            mesh_file_path,
            slicing_coefficient=cortical_v.SLICING_COEFFICIENT,
            n_longitudinal=N_LONGITUDINAL,
            n_transverse_cort=N_TRANSVERSE_CORT,
            n_transverse_trab=N_TRANSVERSE_TRAB,
            n_radial=N_RADIAL,
            QUAD_REFINEMENT=QUAD_REFINEMENT,
            ellipsoid_fitting=ELLIPSOID_FITTING,
        )

        trabecular_volume.set_length_factor(
            float(self.settings_dict["center_square_length_factor"])
        )

        (
            trab_point_tags,
            trab_line_tags_v,
            trab_line_tags_h,
            trab_surfs_v,
            trab_surfs_h,
            trab_vols,
        ) = trabecular_volume.get_trabecular_vol(coi_idx=intersections_int)

        # connection between inner trabecular and cortical volumes
        trab_cort_line_tags = mesher.trabecular_cortical_connection(
            coi_idx=indices_coi_int, trab_point_tags=trab_point_tags
        )

        trab_slice_surf_tags = mesher.trabecular_slices(
            trab_cort_line_tags=trab_cort_line_tags,
            trab_line_tags_h=trab_line_tags_h,
            cort_int_bspline_tags=cortical_int_bspline,
        )

        trab_plane_inertia_tags = trabecular_volume.trabecular_planes_inertia(
            trab_cort_line_tags, trab_line_tags_v, intersection_line_tags_int
        )

        cort_trab_vol_tags = mesher.get_trabecular_cortical_volume_mesh(
            trab_slice_surf_tags,
            trab_plane_inertia_tags,
            cortical_int_surfs,
            trab_surfs_v,
        )

        cort_volume_tags = np.concatenate(
            (cort_vol_tags, cort_trab_vol_tags), axis=None
        )

        trab_refinement = None
        quadref_vols = None
        if trabecular_volume.QUAD_REFINEMENT:
            self.logger.info(
                "Starting quad refinement procedure (this might take some time)"
            )

            # get coords of trab_point_tags
            coords_vertices = []
            for subset in trab_point_tags:
                coords = trabecular_volume.get_vertices_coords(subset)
                coords_vertices.append(coords)

            trab_refinement = QuadRefinement(
                nb_layers=1,
                DIM=2,
                SQUARE_SIZE_0_MM=1,
                MAX_SUBDIVISIONS=3,
                ELMS_THICKNESS=N_LONGITUDINAL,
            )

            (
                quadref_line_tags_intersurf,
                quadref_surfs,
                quadref_vols,
            ) = trab_refinement.exec_quad_refinement(
                trab_point_tags.tolist()
            )  # , coords_vertices
            self.logger.info("Trabecular refinement done")

        # * meshing
        trab_surfs = list(
            chain(
                trab_surfs_v,
                trab_surfs_h,
                trab_slice_surf_tags,
                trab_plane_inertia_tags,
            )
        )

        trab_lines_longitudinal = trab_line_tags_v
        trab_lines_transverse = trab_cort_line_tags
        trab_lines_radial = trab_line_tags_h

        trabecular_volume.meshing_transfinite(
            trab_lines_longitudinal,
            trab_lines_transverse,
            trab_lines_radial,
            trab_surfs,
            trab_vols,
            phase="trab",
        )

        # * physical groups
        mesher.factory.synchronize()
        trab_vol_tags = np.concatenate((trab_vols, cort_trab_vol_tags), axis=None)
        cort_physical_group = mesher.model.addPhysicalGroup(
            3, cort_vol_tags, name="Cortical_Compartment"
        )

        if quadref_vols is not None:
            trab_vol_tags = np.append(trab_vol_tags, quadref_vols[0])
        else:
            pass

        trab_physical_group = mesher.model.addPhysicalGroup(
            3, trab_vol_tags, name="Trabecular_Compartment"
        )

        if trab_refinement is not None and quadref_vols is not None:
            # same as saying if QUAD_REFINEMENT is True
            quadref_physical_group = trab_refinement.model.addPhysicalGroup(
                3, quadref_vols[0]
            )
        print(
            f"Cortical physical group: {cort_physical_group}\nTrabecular physical group: {trab_physical_group}"
        )

        cort_longitudinal_lines = intersection_line_tags
        cort_transverse_lines = intersurface_line_tags
        mesher.meshing_transfinite(
            cort_longitudinal_lines,
            cort_transverse_lines,
            cortical_bspline_tags,
            cort_surfs,
            cort_volume_tags,
            phase="cort",
        )

        mesher.model.geo.mesh.setRecombine(2, -1)

        tot_vol_tags = [cort_vol_tags, trab_vol_tags]
        mesher.mesh_generate(dim=3, element_order=ELM_ORDER, vol_tags=tot_vol_tags)
        mesher.model.mesh.removeDuplicateNodes()
        mesher.model.mesh.removeDuplicateElements()
        mesher.model.occ.synchronize()
        mesher.logger.info("Optimising mesh")
        if ELM_ORDER == 1:
            mesher.model.mesh.optimize(method="UntangleMeshGeometry", force=True)
            pass
        elif ELM_ORDER > 1:
            mesher.model.mesh.optimize(method="HighOrderFastCurving", force=False)

        if MESH_ANALYSIS:
            JAC_FULL = 999.9  # 999.9 if you want to see all the elements
            JAC_NEG = -0.01  # -0.01 if you want to see only negative Jacobians
            mesher.analyse_mesh_quality(hiding_thresh=JAC_FULL)

        nodes = mesher.gmsh_get_nodes()
        elms = mesher.gmsh_get_elms()
        bnds_bot, bnds_top = mesher.gmsh_get_bnds(nodes)
        reference_point_coord = mesher.gmsh_get_reference_point_coord(nodes)

        if SHOW_GMSH:
            gmsh.fltk.run()

        if WRITE_MESH:
            Path(mesher.mesh_file_path).parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(f"{mesher.mesh_file_path}.msh")
            gmsh.write(f"{mesher.mesh_file_path}.inp")
            gmsh.write(f"{mesher.mesh_file_path}.vtk")

        entities_cort = mesher.model.getEntitiesForPhysicalGroup(3, cort_physical_group)
        entities_trab = mesher.model.getEntitiesForPhysicalGroup(3, trab_physical_group)

        centroids_cort = mesher.get_barycenters(tag_s=entities_cort)
        centroids_trab = mesher.get_barycenters(tag_s=entities_trab)

        elm_vol_cort = mesher.get_elm_volume(tag_s=entities_cort)
        elm_vol_trab = mesher.get_elm_volume(tag_s=entities_trab)
        min_elm_vol = min(elm_vol_cort.min(), elm_vol_trab.min())
        if min_elm_vol < 0.0:
            logger.critical(
                f"Negative element volume detected: {min_elm_vol:.3f} (mm^3), exiting..."
            )
        else:
            logger.info(f"Minimum cortical element volume: {np.min(elm_vol_cort):.3f}")
            logger.info(
                f"Minimum trabecular element volume: {np.min(elm_vol_trab):.3f}"
            )

        # get biggest ROI_radius
        radius_roi_cort = mesher.get_radius_longest_edge(tag_s=entities_cort)
        radius_roi_trab = mesher.get_radius_longest_edge(tag_s=entities_trab)

        assert len(elm_vol_cort) + len(elm_vol_trab) == len(centroids_cort) + len(
            centroids_trab
        ), "The number of volumes and centroids does not match."

        # i.e., if cort_physical_group is 1 and trab_physical_group is 2:
        if cort_physical_group < trab_physical_group:
            centroids_c = np.concatenate((centroids_cort, centroids_trab))
        else:
            centroids_c = np.concatenate((centroids_trab, centroids_cort))

        centroids_dict = mesher.get_centroids_dict(centroids_c)
        # split centroids_dict into cort and trab
        centroids_cort_dict, centroids_trab_dict = mesher.split_dict_by_array_len(
            centroids_dict, len(centroids_cort)
        )

        # get number of nodes
        node_tags_cort, _ = mesher.model.mesh.getNodesForPhysicalGroup(
            3, cort_physical_group
        )
        node_tags_trab, _ = mesher.model.mesh.getNodesForPhysicalGroup(
            3, trab_physical_group
        )
        nb_nodes = len(node_tags_cort) + len(node_tags_trab)
        logger.info(f"Number of nodes in model: {nb_nodes}")
        gmsh_log = gmsh.logger.get()
        Path(mesh_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{mesh_file_path}_gmsh.log", "w") as f:
            for line in gmsh_log:
                f.write(line + "\n")
        gmsh.logger.stop()

        gmsh.finalize()
        end = time.time()
        elapsed = round(end - start, ndigits=1)
        logger.info(f"Elapsed time:  {elapsed} (s)")
        logger.info("Meshing script finished.")

        return (
            nodes,
            elms,
            nb_nodes,
            centroids_cort_dict,
            centroids_trab_dict,
            elm_vol_cort,
            elm_vol_trab,
            radius_roi_cort,
            radius_roi_trab,
            bnds_bot,
            bnds_top,
            reference_point_coord,
        )
