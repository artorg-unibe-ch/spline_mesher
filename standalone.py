"""
Geometry representation and meshing through spline reconstruction
Author: Simone Poncioni, MSB
Date: 09.2022 - Ongoing
"""

import os
import logging
from src.pyhexspline.spline_mesher import HexMesh
import hfe_input_transformer as transformer
import coloredlogs
import numpy as np
import gc

# flake8: noqa: E501


def main():
    LOGGING_NAME = "MESHING"
    # configure the logger
    logger = logging.getLogger(LOGGING_NAME)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # configure coloredlogs
    coloredlogs.install(level=logging.INFO, logger=logger)

    img_settings = {
        "img_basefilename": "C0002234",
        "img_basepath": f"{os.getcwd()}/01_AIM",
        "meshpath": f"{os.getcwd()}/03_MESH",
        # "meshpath": "/home/simoneponcioni/Documents/01_PHD/03_Methods/HFE-ACCURATE/03_MESH",  # This is only to test the pipeline
        "outputpath": f"{os.getcwd()}/04_OUTPUT",
    }
    meshing_settings = {
        "aspect": 100,  # aspect ratio of the plots
        "slice": 1,  # slice of the image to be plotted
        "undersampling": 1,  # undersampling factor of the image
        "slicing_coefficient": 5,  # using every nth slice of the image for the spline reconstruction
        "inside_val": int(0),  # threshold value for the inside of the mask
        "outside_val": int(1),  # threshold value for the outside of the mask
        "lower_thresh": float(0),  # lower threshold for the mask
        "upper_thresh": float(119.9),  # upper threshold for the mask
        "s": 5,  # smoothing factor of the spline
        "k": 3,  # degree of the spline
        "interp_points": 100,  # number of points to interpolate the spline
        "thickness_tol": 5e-1,  # minimum cortical thickness tolerance: 3 * XCTII voxel size
        "phases": 2,  # 1: only external contour, 2: external and internal contour
        "center_square_length_factor": 0.5,  # size ratio of the refinement square: 0 < l_f < 1
        # "n_elms_longitudinal": 4,  # number of elements in the longitudinal direction
        # "n_elms_transverse_trab": 6,  # number of elements in the transverse direction for the trabecular compartment
        # "n_elms_transverse_cort": 3,  # number of elements in the transverse direction for the cortical compartment
        # "n_elms_radial": 15,  # number of elements in the radial direction # ! Should be 10 if trab_refinement is True
        "mesh_order": 2,  # set order of the mesh (1: linear, 2: quadratic)
        "show_plots": False,  # show plots during construction
        "show_gmsh": False,  # show gmsh GUI
        "write_mesh": False,  # write mesh to file
        "trab_refinement": False,  # True: refine trabecular mesh at the center
        "mesh_analysis": True,  # True: perform mesh analysis (plot JAC det in GMSH GUI)
    }

    # sitk_image_s = transformer.hfe_input(
    #     path_np_s="/Users/msb/Documents/01_PHD/03_Methods/Meshing/03_MESH/C0001406/C0001406_CORTMASK_array.npy"
    # )

    # sitk_image_s = transformer.hfe_input(
    #     path_np_s="/Users/msb/Documents/01_PHD/03_Methods/HFE-ACCURATE/99_TEMP/material_mapping_testing/synthetic_tests/constant.npy"
    # )

    sitk_image_s = None

    n_elms_longitudinal = np.linspace(2, 2, 5)
    n_elms_transverse_trab = np.linspace(3, 3, 5)
    n_elms_transverse_cort = np.linspace(2, 2, 5)
    n_elms_radial = np.linspace(10, 10, 5)

    for i, _ in enumerate(n_elms_longitudinal):
        meshing_settings.update(
            {
                "n_elms_longitudinal": n_elms_longitudinal[i],
                "n_elms_transverse_trab": n_elms_transverse_trab[i],
                "n_elms_transverse_cort": n_elms_transverse_cort[i],
                "n_elms_radial": n_elms_radial[i],
            }
        )

        mesh = HexMesh(
            meshing_settings,
            img_settings,
            sitk_image=sitk_image_s,
            logger=logging.getLogger(LOGGING_NAME),
        )
        (
            nodes,
            elms,
            centroids_cort,
            centroids_trab,
            elm_vol_cort,
            elm_vol_trab,
            radius_roi_cort,
            radius_roi_trab,
            bnds_bot,
            bnds_top,
            reference_point_coord,
        ) = mesh.mesher()
        print(f"Number of nodes: {len(nodes)}")
        print(f"Number of elements: {len(elms)}")
        print(f"Number of bottom boundary nodes: {len(bnds_bot)}")
        print(f"Number of top boundary nodes: {len(bnds_top)}")
        print(f"Number of centroids in trab physical group: {len(centroids_trab)}")
        print(f"Number of centroids in cort physical group: {len(centroids_cort)}")
        print(f"Radius ROI cort: {radius_roi_cort:.3f} (mm)")
        print(f"Radius ROI trab: {radius_roi_trab:.3f} (mm)")
        print(f"Reference point coordinates: {reference_point_coord} (mm)")
        print("---------------")

        # delete class instance
        del mesh
        gc.collect()


if __name__ == "__main__":
    main()
