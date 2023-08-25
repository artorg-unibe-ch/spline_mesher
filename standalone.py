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
    "aspect": 100,
    "slice": 1,
    "undersampling": 1,
    "slicing_coefficient": 10,  # 5, 10, 20 working on 2234
    "inside_val": int(0),
    "outside_val": int(1),
    "lower_thresh": float(0),
    "upper_thresh": float(0.9),
    "s": 5,
    "k": 3,
    "interp_points": 200,
    "debug_orientation": 0,
    "show_plots": False,
    "show_gmsh": True,
    "write_mesh": True,
    "location": "cort_ext",
    "thickness_tol": 180e-3,  # 3 * XCTII voxel size
    "phases": 2,
    "trab_refinement": False,
    "center_square_length_factor": 0.6,  # 0 < l_f < 1
    "n_elms_longitudinal": 2,
    "n_elms_transverse_trab": 10,
    "n_elms_transverse_cort": 3,
    "n_elms_radial": 10,  # should be 10 if trab_refinement is True
    "mesh_analysis": True,
}

sitk_image_s = transformer.hfe_input(
    path_np_s="/Users/msb/Documents/01_PHD/03_Methods/Meshing/03_MESH/C0001406/C0001406_CORTMASK_array.npy"
)

mesh = HexMesh(meshing_settings, img_settings, sitk_image=sitk_image_s)
(
    nodes,
    elms,
    centroids_trab,
    centroids_cort,
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
print("---------------")
