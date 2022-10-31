# Geometry representation through spline reconstruction

**Author:** Simone Poncioni, MSB Group

**Date:** 29.07.2022

![Issue creation](https://github.com/simoneponcioni/spline-mesher/actions/workflows/todo_to_issue.yml/badge.svg)

![Python application](https://github.com/simoneponcioni/spline-mesher/actions/workflows/python-app.yml/badge.svg)

## Purpose

Import a voxel-based model and convert it to a geometrical simplified representation through the use of splines for each slice in the transverse plane.

## Methodology

- Import the MetaImage file of the masked model that we want to convert;
- Using SimpleITK, extract the binary contour for each slice (in cortical bone, this contains inner+outer cortex)
- Extract only the outer contour using the OpenCV library and interpolate them over a numpy meshgrid
- First sorting: counterclockwise sorting of points (i, j) wrt CoM of the object
- Second sorting: sorting according to Mahalanobis distance (because it's more independent of the starting points than Euclidean distance)
- Representation of a B-spline of 3rd order over the transverse surface on the points (i, j)

## Notable TODOs

- Faster implementation of sorting algorithm
- Faster implementation of cortical sanity check
- Implement test robustness over different models
- Understand how undersampling of raw data affects spline definition

## MESHING IN GMSH - NOTES AND COMMENTS

- 2D algorithm: Quasi-Structured Quads (experimental)

- 3D algorithm: Delaunay

- 2D recombination algorithm: Simple

- Recombine all triangular meshes: FALSE

- Subdivision algorithm: None

![gitlab issue 1804](https://gitlab.onelab.info/gmsh/gmsh/-/issues/1804)
