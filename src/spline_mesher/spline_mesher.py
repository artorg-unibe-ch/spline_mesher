"""
Geometry representation and meshing through spline reconstruction
Author: Simone Poncioni, MSB
Date: 09.2022 - Ongoing
"""
import os

os.environ["NUMEXPR_MAX_THREADS"] = "16"

import logging
import time
from itertools import chain
from pathlib import Path

import cortical_sanity as csc
import gmsh
import numpy as np
import plotly.io as pio
from futils.setup_utils import logging_setup
from gmsh_mesh_builder import Mesher, TrabecularVolume, QuadRefinement
from spline_volume import OCC_volume

pio.renderers.default = "browser"
LOGGING_NAME = "SIMONE"
QUAD_REFINEMENT = bool(True)
MESH_ANALYSIS = bool(False)
# flake8: noqa: E402


def main():
    logger = logging.getLogger(LOGGING_NAME)
    logger.info("Starting meshing script...")
    start = time.time()

    # fmt: off
    img_basefilename = ["C0002234"]
    cwd = os.getcwd()
    img_basepath = f"{cwd}/01_AIM"
    img_outputpath = f"{cwd}/04_OUTPUT"
    img_path_ext = [str(Path(img_basepath, img_basefilename[i], img_basefilename[i] + "_CORT_MASK_cap.mhd",)) for i in range(len(img_basefilename))]
    filepath_ext = [str(Path(img_basepath, img_basefilename[i], img_basefilename[i])) for i in range(len(img_basefilename))]
    filename = [str(Path(img_outputpath, img_basefilename[i], img_basefilename[i] + "_ext.geo_unrolled")) for i in range(len(img_basefilename))]
    # fmt: on
    geo_file_path = f"{cwd}/04_OUTPUT/C0002237/fake_example.geo_unrolled"
    mesh_file_path = f"{cwd}/04_OUTPUT/C0002237/C0002237.msh"

    for i in range(len(img_basefilename)):
        Path.mkdir(
            Path(img_outputpath, img_basefilename[i]), parents=True, exist_ok=True
        )

        cortical_v = OCC_volume(
            img_path_ext[i],
            filepath_ext[i],
            filename[i],
            ASPECT=30,
            SLICE=1,
            UNDERSAMPLING=1,
            SLICING_COEFFICIENT=5,  # 5, 10, 20 working on 2234
            INSIDE_VAL=0,
            OUTSIDE_VAL=1,
            LOWER_THRESH=0,
            UPPER_THRESH=0.9,
            S=5,
            K=3,
            INTERP_POINTS=200,
            debug_orientation=0,
            show_plots=False,
            location="cort_ext",
            thickness_tol=180e-3,  # 180e-3,
            phases=2,
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

        N_TRANSVERSE = 5
        N_RADIAL = 8
        mesher = Mesher(
            geo_file_path,
            mesh_file_path,
            slicing_coefficient=cortical_v.SLICING_COEFFICIENT,
            n_transverse=N_TRANSVERSE,
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
                intersections_ext[i],
            ) = mesher.insert_tensor_of_inertia(
                cortical_ext_split[i], cortex_centroid[i][:-1]
            )
            (
                cortical_int_centroid[i],
                idx_list_int[i],
                intersections_int[i],
            ) = mesher.insert_tensor_of_inertia(
                cortical_int_sanity_split[i], cortex_centroid[i][:-1]
            )

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
        cortical_surfs = np.concatenate(
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
            geo_file_path,
            mesh_file_path,
            slicing_coefficient=cortical_v.SLICING_COEFFICIENT,
            n_transverse=N_TRANSVERSE,
            n_radial=N_RADIAL,
        )
        trabecular_volume.set_length_factor(0.5)

        if QUAD_REFINEMENT:
            trab_refinement = QuadRefinement(
                nb_layers=1,
                DIM=2,
                SQUARE_SIZE_0_MM=1,
                MAX_SUBDIVISIONS=3,
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
        volume_tags = np.concatenate((cort_vol_tags, cort_trab_vol_tags), axis=None)

        # * quad refinement
        for _, trab_verts in enumerate(trab_point_tags):
            quadref_line_tags, quadref_surf_tags = trab_refinement.quad_refinement(
                vertices_tags=trab_verts
            )
            print(f"Quad refinement, line tags:\n{quadref_line_tags}")
            print(f"Quad refinement, surf tags:\n{quadref_surf_tags}")

        # * meshing
        trab_surfs = list(
            chain(
                trab_surfs_v,
                trab_surfs_h,
                trab_slice_surf_tags,
                trab_plane_inertia_tags,
            )
        )
        trabecular_volume.meshing_transfinite(
            trab_line_tags_v,
            trab_line_tags_h,
            trab_surfs,
            trab_vols,
            trab_cort_line_tags,
        )

        # * physical groups
        mesher.factory.synchronize()
        trab_vol_tags = np.concatenate((trab_vols, cort_trab_vol_tags), axis=None)
        cort_physical_group = mesher.model.addPhysicalGroup(3, cort_vol_tags)
        trab_physical_group = mesher.model.addPhysicalGroup(3, trab_vol_tags)

        print(
            f"cortical physical group: {cort_physical_group}\ntrabecular physical group: {trab_physical_group}"
        )

        mesher.meshing_transfinite(
            intersection_line_tags,
            cortical_bspline_tags,
            cortical_surfs,
            volume_tags,
            test_list=intersurface_line_tags,
        )
        mesher.mesh_generate(dim=3, element_order=1, optimise=True)

        if MESH_ANALYSIS:
            JAC_FULL = 999.9  # 999.9 if you want to see all the elements
            JAC_NEG = -0.01
            mesher.analyse_mesh_quality(hiding_thresh=JAC_FULL)

        gmsh.fltk.run()
        gmsh.finalize()
        end = time.time()
        elapsed = round(end - start, ndigits=3)
        logger.info(f"Elapsed time:  {elapsed} (s)")
        logger.info("Meshing script finished.")


if __name__ == "__main__":
    main()
