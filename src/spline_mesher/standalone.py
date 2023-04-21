import os
from spline_mesher import HexMesh

img_settings = {
    "img_basefilename": "C0002234",
    "img_basepath": f"{os.getcwd()}/01_AIM",
    "img_outputpath": f"{os.getcwd()}/04_OUTPUT",
}
meshing_settings = {
    "aspect": 30,
    "slice": 1,
    "undersampling": 1,
    "slicing_coefficient": 5,  # 5, 10, 20 working on 2234
    "inside_val": 0,
    "outside_val": 1,
    "lower_thresh": 0,
    "upper_thresh": 0.9,
    "s": 10,
    "k": 3,
    "interp_points": 200,
    "debug_orientation": 0,
    "show_plots": False,
    "location": "cort_ext",
    "thickness_tol": 180e-3,  # 3 * XCTII voxel size
    "phases": 2,
}

mesh = HexMesh(meshing_settings, img_settings)
nodes, elms = mesh.mesher()
