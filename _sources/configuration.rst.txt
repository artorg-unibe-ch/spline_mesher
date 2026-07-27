Configuration
=============

This section provides the configuration settings for the spline-mesher project.

Image Settings
--------------

The following settings are used to configure the image paths and filenames:

- **img_basefilename**: Base filename for the images. For example `"C0001234"`.
- **img_basepath**: Base path for the images. Default is the current working directory concatenated with `"/01_AIM"`.
- **meshpath**: Path where the mesh files are stored. Default is the current working directory concatenated with `"/03_MESH"`.
- **outputpath**: Path where the output files are stored. Default is the current working directory concatenated with `"/04_OUTPUT"`.

Meshing Settings
----------------

The following settings are used to configure the meshing process:

- **aspect**: Aspect ratio of the plots. Default is `100`.
- **_slice**: Slice of the image to be plotted. Default is `1`.
- **undersampling**: Undersampling factor of the image. Default is `1`.
- **slicing_coefficient**: Using every nth slice of the image for the spline reconstruction. Default is `20`.
- **inside_val**: Threshold value for the inside of the mask. Default is `0`.
- **outside_val**: Threshold value for the outside of the mask. Default is `1`.
- **lower_thresh**: Lower threshold for the mask. Default is `0.0`.
- **upper_thresh**: Upper threshold for the mask. Default is `0.9`.
- **s**: Smoothing factor of the spline. Default is `500`.
- **k**: Degree of the spline. Default is `3`.
- **interp_points**: Number of points to interpolate the spline. Default is `1000`.
- **dp_simplification_outer**: Ramer-Douglas-Peucker simplification factor for the periosteal contour. Default is `5`.
- **dp_simplification_inner**: Ramer-Douglas-Peucker simplification factor for the endosteal contour. Default is `5`.
- **thickness_tol**: Minimum cortical thickness tolerance. Default is `1`.
- **phases**: Number of phases for contouring. Default is `2` (1: only external contour, 2: external and internal contour).
- **center_square_length_factor**: Size ratio of the refinement square. Default is `0.4`.
- **mesh_order**: Element order for the mesh. Default is `1` (1: linear, 2: quadratic, >2: higher order, not tested).
- **n_elms_longitudinal**: Number of elements in the longitudinal direction. Default is `3`.
- **n_elms_transverse_trab**: Number of elements in the transverse direction for the trabecular compartment. Default is `15`.
- **n_elms_transverse_cort**: Number of elements in the transverse direction for the cortical compartment. Default is `3`.
- **n_elms_radial**: Number of elements in the radial direction. Default is `20`.
  - Note: Should be `10` if `trab_refinement` is `True`.
- **ellipsoid_fitting**: Perform ellipsoid fitting in the inner trabecular compartment. Default is `True`.
- **show_plots**: Show plots during construction. Default is `False`.
- **show_gmsh**: Show GMSH GUI. Default is `False`.
- **write_mesh**: Write mesh to file. Default is `True`.
- **trab_refinement**: Refine trabecular mesh at the center. Default is `False`.
  - Note: Should be set to `False` if `ellipsoid_fitting` is `True`.
- **mesh_analysis**: Perform mesh analysis (plot JAC det in GMSH GUI). Default is `True`.
