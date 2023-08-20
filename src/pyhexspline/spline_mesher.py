import os

os.environ["NUMEXPR_MAX_THREADS"] = "16"

import logging
import time
from itertools import chain
from pathlib import Path

import gmsh
import numpy as np
import plotly.io as pio
from pyhexspline import cortical_sanity as csc

# from pyhexspline.futils.setup_utils import logging_setup
from pyhexspline.gmsh_mesh_builder import Mesher, TrabecularVolume
from pyhexspline.quad_refinement import QuadRefinement
from pyhexspline.spline_volume import OCC_volume

import pickle

pio.renderers.default = "browser"
LOGGING_NAME = "SIMONE"
# flake8: noqa: E402


class HexMesh:
    def __init__(self, settings_dict: dict, img_dict: dict, sitk_image=None):
        self.settings_dict = settings_dict
        self.img_dict = img_dict
        if sitk_image is not None:
            self.sitk_image = sitk_image  # imports the image
        else:
            self.sitk_image = None  # reads the image from img_path_ext

    def mesher(self):
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
        SLICE = int(self.settings_dict["slice"])
        UNDERSAMPLING = int(self.settings_dict["undersampling"])
        SLICING_COEFFICIENT = int(self.settings_dict["slicing_coefficient"])
        INSIDE_VAL = float(self.settings_dict["inside_val"])
        OUTSIDE_VAL = float(self.settings_dict["outside_val"])
        LOWER_THRESH = float(self.settings_dict["lower_thresh"])
        UPPER_THRESH = float(self.settings_dict["upper_thresh"])
        S = int(self.settings_dict["s"])
        K = int(self.settings_dict["k"])
        INTERP_POINTS = int(self.settings_dict["interp_points"])
        DEBUG_ORIENTATION = int(self.settings_dict["debug_orientation"])
        SHOW_PLOTS = bool(self.settings_dict["show_plots"])
        SHOW_GMSH = bool(self.settings_dict["show_gmsh"])
        WRITE_MESH = bool(self.settings_dict["write_mesh"])
        LOCATION = str(self.settings_dict["location"])
        THICKNESS_TOL = float(self.settings_dict["thickness_tol"])
        PHASES = int(self.settings_dict["phases"])

        # spline_mesher settings
        N_LONGITUDINAL = int(self.settings_dict["n_elms_longitudinal"])
        N_TRANSVERSE_CORT = int(self.settings_dict["n_elms_transverse_cort"])
        N_TRANSVERSE_TRAB = int(self.settings_dict["n_elms_transverse_trab"])
        N_RADIAL = int(self.settings_dict["n_elms_radial"])
        QUAD_REFINEMENT = bool(self.settings_dict["trab_refinement"])
        MESH_ANALYSIS = bool(self.settings_dict["mesh_analysis"])

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
            location=LOCATION,
            thickness_tol=THICKNESS_TOL,
            phases=PHASES,
        )

        cortical_v.plot_mhd_slice()
        cortical_ext, cortical_int = cortical_v.volume_splines()
        # Cortical surface sanity check
        cortex = csc.CorticalSanityCheck(
            MIN_THICKNESS=cortical_v.MIN_THICKNESS,
            ext_contour=cortical_ext,
            int_contour=cortical_int,
            model=cortical_v.filename,
            save_plot=False,
        )
        cortical_ext_split = np.array_split(
            cortical_ext, len(np.unique(cortical_ext[:, 2]))
        )
        cortical_int_split = np.array_split(
            cortical_int, len(np.unique(cortical_int[:, 2]))
        )
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

        mesher = Mesher(
            geo_unrolled_filename,
            mesh_file_path,
            slicing_coefficient=cortical_v.SLICING_COEFFICIENT,
            n_longitudinal=N_LONGITUDINAL,
            n_transverse_trab=N_TRANSVERSE_TRAB,
            n_transverse_cort=N_TRANSVERSE_CORT,
            n_radial=N_RADIAL,
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
            _, idx = np.unique(
                cortical_ext_split[i].round(decimals=6), axis=0, return_index=True
            )
            cortical_ext_split[i][np.sort(idx)]
            _, idx = np.unique(
                cortical_int_sanity_split[i].round(decimals=6),
                axis=0,
                return_index=True,
            )
            cortical_int_sanity_split[i][np.sort(idx)]
            cortex_centroid[i][:-1] = mesher.polygon_tensor_of_inertia(
                cortical_ext_split[i], cortical_int_sanity_split[i]
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

        # TODO: check if could be implemented when created (relationship with above functions)
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
            )

            (
                quadref_line_tags_intersurf,
                quadref_surfs,
                quadref_vols,
            ) = trab_refinement.exec_quad_refinement(
                trab_point_tags.tolist()
            )  # , coords_vertices

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
        trab_physical_group = mesher.model.addPhysicalGroup(
            3, trab_vol_tags, name="Trabecular_Compartment"
        )

        if (
            trab_refinement is not None and quadref_vols is not None
        ):  # same as saying if QUAD_REFINEMENT
            quadref_physical_group = trab_refinement.model.addPhysicalGroup(
                3, quadref_vols[0]
            )
        print(
            f"cortical physical group: {cort_physical_group}\ntrabecular physical group: {trab_physical_group}"
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

        mesher.mesh_generate(dim=3, element_order=1, optimise=True)
        mesher.model.mesh.removeDuplicateNodes()
        mesher.model.occ.synchronize()

        if MESH_ANALYSIS:
            JAC_FULL = 999.9  # 999.9 if you want to see all the elements
            JAC_NEG = -0.01
            mesher.analyse_mesh_quality(hiding_thresh=JAC_FULL)

        nodes = mesher.gmsh_get_nodes()
        elms = mesher.gmsh_get_elms()
        bnds_bot, bnds_top = mesher.gmsh_get_bnds(nodes)
        reference_point_coord = mesher.gmsh_get_reference_point_coord(bnds_top)

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

        gmsh.finalize()
        end = time.time()
        elapsed = round(end - start, ndigits=3)
        logger.info(f"Elapsed time:  {elapsed} (s)")
        logger.info("Meshing script finished.")

        with open(f"{mesh_file_path}_nodes.pickle", "wb") as handle:
            nodes_pkl = pickle.dump(nodes, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{mesh_file_path}_elms.pickle", "wb") as handle:
            elms_pkl = pickle.dump(elms, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{mesh_file_path}_centroids_trab.pickle", "wb") as handle:
            centroids_pkl = pickle.dump(
                centroids_trab, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        with open(f"{mesh_file_path}_centroids_cort.pickle", "wb") as handle:
            centroids_pkl = pickle.dump(
                centroids_cort, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        with open(f"{mesh_file_path}_bnds_bot.pickle", "wb") as handle:
            bnds_bot_pkl = pickle.dump(
                bnds_bot, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        with open(f"{mesh_file_path}_bnds_top.pickle", "wb") as handle:
            bnds_top_pkl = pickle.dump(
                bnds_top, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

        botpath = f"{mesh_file_path}_spline_botnodes.pickle"
        with open(botpath, "wb") as f:
            pickle.dump(bnds_bot, f, protocol=pickle.HIGHEST_PROTOCOL)

        toppath = f"{mesh_file_path}_spline_topnodes.pickle"
        with open(toppath, "wb") as f:
            pickle.dump(bnds_top, f, protocol=pickle.HIGHEST_PROTOCOL)

        cort_dict = f"{mesh_file_path}_spline_centroids_cort_dict.pickle"
        with open(cort_dict, "wb") as f:
            pickle.dump(centroids_cort_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        trab_dict = f"{mesh_file_path}_spline_centroids_trab_dict.pickle"
        with open(trab_dict, "wb") as f:
            pickle.dump(centroids_trab_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return (
            nodes,
            elms,
            centroids_cort,
            centroids_trab,
            bnds_bot,
            bnds_top,
            reference_point_coord,
        )
