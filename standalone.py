"""
Geometry representation and meshing through spline reconstruction
Author: Simone Poncioni, MSB
Date: 09.2022 - Ongoing
"""

import cProfile
import logging
import os

import coloredlogs
import hfe_input_transformer as transformer
import numpy as np
from src.pyhexspline.spline_mesher import HexMesh

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
        "img_basefilename": "C0003102",
        "img_basepath": f"{os.getcwd()}/01_AIM",
        "meshpath": f"{os.getcwd()}/03_MESH",
        # "meshpath": "/home/simoneponcioni/Documents/01_PHD/03_Methods/HFE-ACCURATE/03_MESH",  # This is only to test the pipeline
        "outputpath": f"{os.getcwd()}/04_OUTPUT",
    }
    meshing_settings = {
        "aspect": 100,  # aspect ratio of the plots
        "slice": 1,  # slice of the image to be plotted
        "undersampling": 1,  # undersampling factor of the image
        "slicing_coefficient": 8,  # using every nth slice of the image for the spline reconstruction
        "inside_val": int(0),  # threshold value for the inside of the mask
        "outside_val": int(1),  # threshold value for the outside of the mask
        "lower_thresh": float(0),  # lower threshold for the mask
        "upper_thresh": float(0.9),  # upper threshold for the mask
        "s": 300,  # smoothing factor of the spline
        "k": 3,  # degree of the spline
        "interp_points": 350,  # number of points to interpolate the spline
        "thickness_tol": 5e-1,  # minimum cortical thickness tolerance: 3 * XCTII voxel size
        "phases": 2,  # 1: only external contour, 2: external and internal contour
        "center_square_length_factor": 0.4,  # size ratio of the refinement square: 0 < l_f < 1
        "mesh_order": 1,  # set order of the mesh (1: linear, 2: quadratic)
        "n_elms_longitudinal": 5,  # number of elements in the longitudinal direction
        "n_elms_transverse_trab": 10,  # number of elements in the transverse direction for the trabecular compartment
        "n_elms_transverse_cort": 3,  # number of elements in the transverse direction for the cortical compartment
        "n_elms_radial": 20,  # number of elements in the radial direction # ! Should be 10 if trab_refinement is True
        "ellipsoid_fitting": True,  # True: perform ellipsoid fitting
        "show_plots": False,  # show plots during construction
        "show_gmsh": True,  # show gmsh GUI
        "write_mesh": True,  # write mesh to file
        "trab_refinement": False,  # True: refine trabecular mesh at the center
        "mesh_analysis": True,  # True: perform mesh analysis (plot JAC det in GMSH GUI)
    }

    sitk_image_s = transformer.hfe_input(
        path_np_s="/Users/msb/Documents/01_PHD/03_Methods/Meshing/01_AIM/C0003094/C0003094_CORT_MASK_UNCOMP.npy"
    )

    # sitk_image_s = sitk_image_s[:, :, :-35]
    # print(sitk_image_s.GetSize())

    # sitk_image_s = None

    mesh = HexMesh(
        meshing_settings,
        img_settings,
        sitk_image=sitk_image_s,
        logger=logging.getLogger(LOGGING_NAME),
    )
    (
        nodes,
        elms,
        nb_nodes,
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

    _w_text = 60  # width of the text
    _w_value = 10  # width of the value
    logger.info(f"{'Nodes:':<{_w_text}}{len(nodes):>{_w_value}}")
    logger.info(f"{'Elements:':<{_w_text}}{len(elms):>{_w_value}}")
    logger.info(f"{'Bottom boundary nodes:':<{_w_text}}{len(bnds_bot):>{_w_value}}")
    logger.info(f"{'Top boundary nodes:':<{_w_text}}{len(bnds_top):>{_w_value}}")
    logger.info(
        f"{'Centroids in trab physical group:':<{_w_text}}{len(centroids_trab):>{_w_value}}"
    )
    logger.info(
        f"{'Centroids in cort physical group:':<{_w_text}}{len(centroids_cort):>{_w_value}}"
    )
    logger.info(f"'Radius ROI cort: {radius_roi_cort:.3f} (mm)")
    logger.info(f"'Radius ROI trab: {radius_roi_trab:.3f} (mm)")

    elm_vol = np.concatenate((elm_vol_cort.flatten(), elm_vol_trab.flatten()))
    min_volume = min(elm_vol)
    if min_volume < 0:
        logger.critical(f"Negative volume detected: {min_volume:.3f} (mm^3)")
    else:
        logger.info(f"Minimum element volume: {min_volume:.3f} (mm^3)")

    print("-" * 150)


if __name__ == "__main__":
    main()
