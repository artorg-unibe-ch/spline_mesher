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
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
from pyhexspline import cortical_sanity as csc
from pyhexspline.geometry_cleaner import GeometryCleaner
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

    def make_thrusection_compound(self, thrusection_volumes):
        thrusections_compound_surfs = []
        for subvol in thrusection_volumes:
            _, surf_cortvol = gmsh.model.getAdjacencies(3, subvol[0][1])
            thrusections_compound_surfs.append(surf_cortvol)

        for surf in list(zip(*thrusections_compound_surfs)):
            print(surf)
            gmsh.model.mesh.setCompound(2, surf)

        compound_volumes = []
        for subvol in thrusection_volumes:
            compound_volumes.append(subvol[0][1])

        gmsh.model.mesh.setCompound(3, compound_volumes)
        gmsh.model.occ.synchronize()
        return None

    def delete_unused_entities(self, dim, arr: list):
        # make sure that the values inside the list are integers
        gmsh.model.occ.synchronize()
        lines_before = gmsh.model.occ.getEntities(1)
        arr = np.array(arr, dtype=int).tolist()
        surfs_to_delete = []
        if dim == 1:
            for ll in arr:
                surf, _ = gmsh.model.getAdjacencies(1, ll)
                surfs_to_delete.extend(surf)
                gmsh.model.occ.synchronize()
                surfs_to_delete = set(surfs_to_delete)
            for ss in surfs_to_delete:
                gmsh.model.occ.remove([(2, ss)])
            for ll in arr:
                gmsh.model.occ.remove([(1, ll)])
            gmsh.model.occ.synchronize()
            lines_after = gmsh.model.occ.getEntities(1)
            print(f"Deleted {len(lines_before) - len(lines_after)} entities")

        elif dim == 2:
            surfs_before = gmsh.model.occ.getEntities(2)
            for ss in arr:
                gmsh.model.occ.remove([(2, ss)])
            surfs_after = gmsh.model.occ.getEntities(2)
            print(f"Deleted {len(surfs_before) - len(surfs_after)} entities")
        return None

    def get_surfaces_and_lines_from_volumes(self, volume):
        """
        Get surfaces and lines from the given volumes using Gmsh operations.

        Parameters:
        volumes (list): List of volume definitions, each containing dimension and tag.
        gmsh (module): The gmsh module.

        Returns:
        dict: Dictionary containing 'surfaces' and 'lines' for each volume.
        """
        results = {"surfaces": [], "lines": []}
        try:
            dim, tag = volume[0][0], volume[0][1]
        except TypeError:
            dim, tag = volume[0], volume[1]

        # Get the adjacencies for the current volume
        _, surfaces_sections = gmsh.model.getAdjacencies(dim=dim, tag=tag)

        adjacencies = [
            gmsh.model.getAdjacencies(dim=2, tag=surface)
            for surface in surfaces_sections
        ]
        tag_arrays = [adjacency[1] for adjacency in adjacencies]
        lines_sections = sorted(np.unique(np.concatenate(tag_arrays).tolist()))

        results["surfaces"].append(surfaces_sections)
        results["lines"].append(lines_sections)
        return results

    def get_curvature_from_entities(self, lines):
        ATOL = 1e-9
        straight_segments = []
        curved_segments = []

        if isinstance(lines, list):
            # flatten
            lines = list(chain.from_iterable(lines))
        else:
            pass

        for line in lines:
            der = gmsh.model.getDerivative(1, line, [0.0, 0.5])
            if not (
                np.isclose(der[0], der[3], atol=ATOL)
                and np.isclose(der[1], der[4], atol=ATOL)
                and np.isclose(der[2], der[5], atol=ATOL)
            ):
                curved_segments.append(line)
            else:
                straight_segments.append(line)
        assert len(straight_segments) + len(curved_segments) == len(lines)
        return straight_segments, curved_segments

    def get_radial_longitudinal_lines(self, splines):
        # get the point pairs of all the splines (begin, end)
        radial_lines = []
        longitudinal_lines = []

        for spl in splines:
            _, pts = gmsh.model.getAdjacencies(1, spl)
            _, _, z0 = gmsh.model.getValue(0, pts[0], [])
            _, _, z1 = gmsh.model.getValue(0, pts[1], [])
            if z0 != z1:
                longitudinal_lines.append(spl)
            else:
                radial_lines.append(spl)
        assert len(radial_lines) + len(longitudinal_lines) == len(splines)

        # point_coords: dictionary with key = spline tag, value = (x0, y0, z0), (x1, y1, z1)
        # point_coords[spl] = [(x0, y0, z0), (x1, y1, z1)]
        return radial_lines, longitudinal_lines

    # ******

    def split_thrusection_entities(self, thrusections_tags):
        thrusection_entities = {"surfaces": [], "lines": []}
        for _, section in enumerate(thrusections_tags):
            thru_entities = self.get_surfaces_and_lines_from_volumes(section)
            for key, value in thru_entities.items():
                thrusection_entities[key].extend(value)

        lines, splines = self.get_curvature_from_entities(thrusection_entities["lines"])
        # ? lines: cortical elements in transverse direction = 3
        # ? splines: cortical elements in radial and longitudinal directions = 15 / 20
        splines_radial, splines_longitudinal = self.get_radial_longitudinal_lines(
            splines
        )
        return lines, splines_radial, splines_longitudinal

    def add_thrusection(
        self,
        surf_tags,
        MAX_DEGREE=2,
        MAKE_SOLID=True,
        CONTINUITY="G2",
    ):
        gmsh.model.occ.synchronize()
        thrusections_volumes = []
        # Transpose slices_tags to loop over the 'columns' of the list
        if all(isinstance(item, int) for item in surf_tags):
            thrusect = gmsh.model.occ.addThruSections(
                surf_tags,
                maxDegree=MAX_DEGREE,
                makeSolid=MAKE_SOLID,
                continuity=CONTINUITY,
                tag=-1,
            )
            thrusections_volumes.append(thrusect)
        else:
            for subset in list(zip(*surf_tags)):
                # Add ThruSections for cortical slices
                thrusect = gmsh.model.occ.addThruSections(
                    subset,
                    maxDegree=MAX_DEGREE,
                    makeSolid=MAKE_SOLID,
                    continuity=CONTINUITY,
                    tag=-1,
                )
                thrusections_volumes.append(thrusect)
        gmsh.model.occ.synchronize()
        return thrusections_volumes

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
        SHOW_PLOTS = bool(self.settings_dict["show_plots"])
        SHOW_GMSH = bool(self.settings_dict["show_gmsh"])
        WRITE_MESH = bool(self.settings_dict["write_mesh"])
        THICKNESS_TOL = float(self.settings_dict["thickness_tol"])
        PHASES = int(self.settings_dict["phases"])

        # spline_mesher settings
        N_LONGITUDINAL = int(self.settings_dict["n_elms_longitudinal"])
        N_TRANSVERSE_CORT = int(self.settings_dict["n_elms_transverse_cort"])
        N_TRANSVERSE_TRAB = int(self.settings_dict["n_elms_transverse_trab"])
        N_RADIAL = int(self.settings_dict["n_elms_radial"])
        ELM_ORDER = int(self.settings_dict["mesh_order"])
        QUAD_REFINEMENT = bool(self.settings_dict["trab_refinement"])
        MESH_ANALYSIS = bool(self.settings_dict["mesh_analysis"])
        ELLIPSOID_FITTING = bool(self.settings_dict["ellipsoid_fitting"])

        DEBUG_ORIENTATION = (
            0  # 0: no debug, 1: debug # ! obscured from settings by design
        )

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
            debug_orientation=DEBUG_ORIENTATION,
            show_plots=SHOW_PLOTS,
            thickness_tol=THICKNESS_TOL,
            phases=PHASES,
        )

        cortical_v.plot_mhd_slice()
        cortical_ext_split, cortical_int_split, cortical_int_sanity = (
            cortical_v.volume_spline_fast_implementation()
        )

        gmsh.initialize()
        gmsh.clear()
        gmsh.option.setNumber("General.NumThreads", 16)
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
        ) = mesher.gmsh_geometry_formulation(cortical_ext_msh, idx_list_ext)
        (
            indices_coi_int,
            cortical_int_bspline,
        ) = mesher.gmsh_geometry_formulation(cortical_int_msh, idx_list_int)

        intersurface_line_tags = mesher.add_interslice_segments(
            indices_coi_ext, indices_coi_int
        )

        slices_tags = mesher.add_slice_surfaces(
            cortical_ext_bspline, cortical_int_bspline, intersurface_line_tags
        )

        # *ThruSections for Cortical Slices
        cort_slice_thrusections_volumes = self.add_thrusection(
            slices_tags, MAX_DEGREE=2, MAKE_SOLID=True, CONTINUITY="G2"
        )

        cort_lines, cort_splines_radial, cort_splines_longitudinal = (
            self.split_thrusection_entities(cort_slice_thrusections_volumes)
        )

        # make intersurface_line_tags a list of integers
        # intersurface_line_tags = np.array(intersurface_line_tags, dtype=int).tolist()
        # cort_lines.extend(intersurface_line_tags)
        # TODO: delete all entities that are not created by ThruSections
        # get ThruSection entities
        all_line_entities = gmsh.model.getEntities(1)

        # only keep tag
        all_line_entities = [line[1] for line in all_line_entities]
        lines_to_delete = [line for line in all_line_entities if line not in cort_lines]

        print(f"Deleting {len(lines_to_delete)} entities")
        [gmsh.model.occ.remove([(1, line)]) for line in lines_to_delete]

        transverse_lines = []
        radial_lines = []
        radial_lines.extend(cort_splines_radial)
        longitudinal_lines = cort_splines_longitudinal

        # *ThruSections for Cortical-Trabecular Interface
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

        # * ThruSection for Inner Trabecular Volume
        (
            trab_point_tags,
            trab_line_tags_v,
            trab_line_tags_h,
            trab_curveloop_h_list,
        ) = trabecular_volume.get_trabecular_vol(coi_idx=intersections_int)

        trab_inner_surf_thrusection = self.add_thrusection(
            trab_curveloop_h_list,
            MAX_DEGREE=2,
            MAKE_SOLID=True,
            CONTINUITY="G2",
        )

        trab_inner_lines, trab_inner_splines_radial, trab_inner_splines_longitudinal = (
            self.split_thrusection_entities(trab_inner_surf_thrusection)
        )
        # TODO validate this:
        transverse_lines.extend(trab_inner_lines)
        radial_lines.extend(trab_inner_splines_radial)
        longitudinal_lines.extend(trab_inner_splines_longitudinal)

        # connection between inner trabecular and cortical volumes
        trab_cort_line_tags = mesher.trabecular_cortical_connection(
            coi_idx=indices_coi_int, trab_point_tags=trab_point_tags
        )

        trabecular_slice_curve_loop_tags = mesher.trabecular_slices(
            trab_cort_line_tags=trab_cort_line_tags,
            trab_line_tags_h=trab_line_tags_h,
            cort_int_bspline_tags=cortical_int_bspline,
        )

        # add thrusection
        trab_slice_surf_thrusections = self.add_thrusection(
            trabecular_slice_curve_loop_tags,
            MAX_DEGREE=2,
            MAKE_SOLID=True,
            CONTINUITY="G2",
        )
        # split entities
        trab_lines, trab_splines_radial, trab_splines_longitudinal = (
            self.split_thrusection_entities(trab_slice_surf_thrusections)
        )

        # TODO validate this:
        transverse_lines.extend(trab_lines)
        radial_lines.extend(trab_line_tags_h)
        radial_lines.extend(trab_splines_radial)
        longitudinal_lines.extend(trab_cort_line_tags)
        longitudinal_lines.extend(trab_splines_longitudinal)

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
                trab_slice_surf_thrusections,
            )
        )

        trab_lines_longitudinal = trab_line_tags_v
        trab_lines_transverse = np.concatenate(
            (trab_cort_line_tags, cortical_int_bspline), axis=None
        )
        trab_lines_radial = trab_line_tags_h
        radial_lines.extend(trab_lines_radial)

        mesher.factory.synchronize()

        ######################################################################
        # TRANSFINITE CURVES - LONGITUDINAL, CORTICAL, TRANSVERSE, RADIAL

        gmsh.model.occ.synchronize()
        for line in longitudinal_lines:
            mesher.model.mesh.setTransfiniteCurve(line, N_LONGITUDINAL)
        for line in cort_lines:
            mesher.model.mesh.setTransfiniteCurve(line, N_TRANSVERSE_CORT)
        for line in transverse_lines:
            mesher.model.mesh.setTransfiniteCurve(line, N_TRANSVERSE_TRAB)
        for line in radial_lines:
            mesher.model.mesh.setTransfiniteCurve(line, N_RADIAL)

        for surf in gmsh.model.getEntities(2):
            mesher.model.mesh.setTransfiniteSurface(surf[1])
            mesher.model.mesh.setRecombine(2, surf[1])
            mesher.model.mesh.setSmoothing(surf[0], surf[1], 100000)
        for vol in gmsh.model.getEntities(3):
            mesher.model.mesh.setTransfiniteVolume(vol[1])

        gmsh.model.occ.synchronize()
        # make_thrusection_compound(cort_slice_thrusections_volumes)
        gmsh.model.occ.synchronize()
        mesher.mesh_generate(dim=3, element_order=ELM_ORDER)
        ######################################################################

        if SHOW_GMSH:
            gmsh.fltk.run()
        if WRITE_MESH:
            gmsh.write(str(Path(mesh_file_path + ".msh")))

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
