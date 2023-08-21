"""
Geometry representation and meshing through spline reconstruction
Author: Simone Poncioni, MSB
Date: 09.2022 - Ongoing
"""

import os
from src.pyhexspline.spline_mesher import HexMesh
import hfe_input_transformer as transformer


img_settings = {
    "img_basefilename": "C0001406",
    "img_basepath": f"{os.getcwd()}/01_AIM",
    "meshpath": f"{os.getcwd()}/03_MESH",
    "outputpath": f"{os.getcwd()}/04_OUTPUT",
}
meshing_settings = {
    "aspect": 100,  # aspect ratio of the plots
    "slice": 1,  # slice of the image to be plotted
    "undersampling": 1,  # undersampling factor of the image
    "slicing_coefficient": 5,  # 5, 10, 20 working on 2234
    "inside_val": int(0),  # threshold value for the inside of the mask
    "outside_val": int(1),  # threshold value for the outside of the mask
    "lower_thresh": float(0),  # lower threshold for the mask
    "upper_thresh": float(0.9),  # upper threshold for the mask
    "s": 5,  # smoothing factor of the spline
    "k": 3,  # degree of the spline
    "interp_points": 200,  # number of points to interpolate the spline
    "debug_orientation": 0,  # 0: no debug, 1: debug orientation after Mahalanobis sorting # TODO: remove from settings
    "show_plots": False,  # show plots during construction
    "show_gmsh": False,  # show gmsh GUI
    "write_mesh": False,  # write mesh to file
    "location": "cort_ext",
    "thickness_tol": 180e-3,  # minimum cortical thickness tolerance: 3 * XCTII voxel size
    "phases": 2,  # 1: only external contour, 2: external and internal contour
    "trab_refinement": False,  # True: refine trabecular mesh at the center
    "center_square_length_factor": 0.6,  # size ratio of the refinement square: 0 < l_f < 1
    "n_elms_longitudinal": 3,  # number of elements in the longitudinal direction
    "n_elms_transverse_trab": 5,  # number of elements in the transverse direction for the trabecular compartment
    "n_elms_transverse_cort": 5,  # number of elements in the transverse direction for the cortical compartment
    "n_elms_radial": 5,  # number of elements in the radial direction # ! Should be 10 if trab_refinement is True
    "mesh_analysis": False,  # True: perform mesh analysis (plot JAC det in GMSH GUI)
}

sitk_image = transformer.hfe_input(
    path_np_s="99_testing_prototyping/pipeline_implementation_errors/C0001406_CORTMASK_array.npy"
)

mesh = HexMesh(meshing_settings, img_settings, sitk_image=sitk_image)
(
    nodes,
    elms,
    centroids_cort,
    centroids_trab,
    elm_vol_cort,
    elm_vol_trab,
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
print(f"Reference point coordinates: {reference_point_coord}")
print("---------------")
