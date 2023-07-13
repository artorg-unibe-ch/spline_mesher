"""
Geometry representation and meshing through spline reconstruction
Author: Simone Poncioni, MSB
Date: 09.2022 - Ongoing
"""

import os
from src.pyhexspline.spline_mesher import HexMesh
import numpy as np
import SimpleITK as sitk


def numpy_reader_to_sitk(path_np: str, spacing: list = [0.0607, 0.0607, 0.0607]):
    """
    Little helper to read numpy array and convert it to sitk image
    Used to get same import conditions as pipeline, where 'dict(bone)' contains a np array of the cortical mask
    """
    cort_mask_np = np.load(path_np)

    sitk_image = sitk.GetImageFromArray(cort_mask_np)
    sitk_image = sitk.PermuteAxes(sitk_image, [2, 1, 0])
    sitk_image = sitk.Flip(sitk_image, [False, True, False])
    sitk_image.SetSpacing(spacing)
    return sitk_image


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
    "slicing_coefficient": 5,  # 5, 10, 20 working on 2234
    "inside_val": int(0),
    "outside_val": int(1),
    "lower_thresh": float(0),
    "upper_thresh": float(0.9),
    "s": 10,
    "k": 3,
    "interp_points": 200,
    "debug_orientation": 0,
    "show_plots": False,
    "show_gmsh": False,
    "write_mesh": True,
    "location": "cort_ext",
    "thickness_tol": 180e-3,  # 3 * XCTII voxel size
    "phases": 2,
    "trab_refinement": False,
    "center_square_length_factor": 0.6,  # less than 1
    "n_elms_longitudinal": 3,
    "n_elms_transverse_trab": 10,
    "n_elms_transverse_cort": 4,
    "n_elms_radial": 10,  # should be 10 if trab_refinement is True
    "mesh_analysis": False,
}

sitk_image = numpy_reader_to_sitk(
    path_np="99_testing_prototyping/pipeline_implementation_errors/C0001406_CORTMASK_array.npy",
    spacing=[0.0607, 0.0607, 0.0607],
)

# save image to mhd format for debugging
sitk.WriteImage(
    sitk_image,
    "99_testing_prototyping/pipeline_implementation_errors/C0001406_CORTMASK.mhd",
)


mesh = HexMesh(meshing_settings, img_settings, sitk_image=sitk_image)
nodes, elms, centroids, bnds_bot, bnds_top, reference_point_coord = mesh.mesher()
print(f"Number of nodes: {len(nodes)}")
print(f"Number of elements: {len(elms)}")
print(f"Number of bottom boundary nodes: {len(bnds_bot)}")
print(f"Number of top boundary nodes: {len(bnds_top)}")
print(f"Number of centroids: {len(centroids)}")
print("---------------")
