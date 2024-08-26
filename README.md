<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

![Issue creation][todo_to_issue]
![Python application][pyapp]

<h3 align="center">Spline-based structured conformal hexahedral meshing</h3>

  <p align="center">
    A Python package for generating GMSH meshes from SCANCO HR-pQCT images.
    <br />
    <a href="https://github.com/simoneponcioni/spline-mesher"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/simoneponcioni/spline-mesher">View Demo</a>
    ·
    <a href="https://github.com/simoneponcioni/spline-mesher/issues">Report Bug</a>
    ·
    <a href="https://github.com/simoneponcioni/spline-mesher/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/artorg-unibe-ch/spline_mesher/blob/master/src/pyhexspline/docs/img/mesh-example01.png" alt="Spline-mesher01" style="width: 45%;">
  <img src="https://github.com/artorg-unibe-ch/spline_mesher/blob/master/src/pyhexspline/docs/img/mesh-example02.png" alt="Spline-mesher02" style="width: 45%;">
</div>




Import a voxel-based model and convert it to a geometrical simplified representation through the use of splines for each slice in the transverse plane.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<p align="center">
  <a href="https://www.python.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" width="100" height="100"></a>
  <a href="https://gmsh.info/"><img src="https://gitlab.onelab.info/uploads/-/system/project/avatar/3/gmsh.png" alt="GMSH" width="100" height="100"></a>
</p>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/simoneponcioni/spline-mesher.git
   ```

2. Install dependencies

   ```sh
   conda create -n meshenv python=3.9 --file requirements.txt
   conda activate meshenv
   python setup.py build_ext --inplace
   python setup.py install
   ```

3. For developers: install the package in editable mode and install requirements for testing

   ```sh
   pip install -e .
   pip install -r requirements-dev.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Standalone execution - rationale

A robust, standalone meshing algorithm designed to employ the cortical mask obtained from the standard image processing technique of the scanner was developed. The algorithm is designed to generate a biphasic, structured, and fully hexahedral mesh of the clinical section. This smooth representation is believed to result in a more precise representation of the mechanical behaviour of the cortical shell than voxel-based isotropic hexahedral meshes. Furthermore, the creation of a structured mesh offers several advantages. Firstly, they offer greater simplicity and efficiency, requiring significantly less memory because connectivity with neighbouring elements is defined implicitly. Secondly, the creation of structured meshes allows topologically identical models to be obtained, enabling easier comparison between different patients measured at the same anatomical site or between multiple measurements in longitudinal studies. Initially, the cortical mask is imported, binarised, and padded to ensure that the periosteal contour does not intersect the image borders. The periosteal and endosteal contours are extracted using the scikit-image package. The contours are sorted in a counter-clockwise direction. The two point clouds are used to build polygons slice-wise, and the geometry of each polygon is simplified using a Douglas-Ramer-Peucker algorithm to remove high-frequency changes in the contours. Successively, a 3rd order periodic BSpline is used in the transverse plane to construct a smooth and accurate representation of the polygons. An internal coordinate system passing through the centroid of the splines is then used to identify all coplanar points in the longitudinal direction to fit 3rd-order BSplines in the plane to get a smoother transition between slices. The final point cloud finally undergoes a sanity check where a minimum cortical thickness of \SI{0.5}{\milli\metre} is set for all periosteal-endosteal point pairs. This guarantees the necessary space to fit at least three elements in the cortical thickness and ensures continuity in extremely thin structures. The points constructing the simplified geometry are imported in the OpenCASCADE kernel built in GMSH, a three-dimensional finite element mesh generator. The mesh geometry is constructed solely using geometric primitives (points, lines, and splines). The geometry is subdivided into simpler six-face subvolumes partitioned through the main axes of inertia of the model, which guarantees the use of the transfinite technique implemented in the mesh generator. The transfinite subvolumes are then used to generate a volumetric structured smooth mesh on the cortical and trabecular compartments. The cortical elements are conforming with the trabecular elements. The generated mesh is subsequently optimised using a Winslow untagler. The mesh size was defined according to a convergence study for stiffness \( \mathrm{S} \) (\SI{}{\newton\per\milli\metre}) and yield force \( \mathrm{F_y}\) (\SI{}{\newton}). The capacity of the mesh to represent the geometry (smoothing and polygon simplification) was determined through the utilisation of the Dice similarity coefficient (DSC), as outlined by Zou et al. A minimum value of \SI{95}{\percent} was considered appropriate. The element quality was assessed by calculating the (signed-) inverse conditioning number ((S-) ICN). The (S-) ICN measure not only describes the quality of the element by quantifying its deviation from the ideal element shape and its distance to degeneracy, but it is also linked to the conditioning of the stiffness matrix. A badly conditioned stiffness matrix can potentially lead to roundoff errors or significantly slow down the convergence speed of the simulation.
  
_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] __v0.0.1__: single execution of radius and tibia mesher
- [x] __v0.0.2__:
  - [x] Faster implementation of sorting algorithm
  - [x] Faster implementation of cortical sanity check
  - [x] Implement test robustness over different models
- [ ] __v1.1.0__: add phase discrimination in order to mesh single phase models (e.g. vertebrae)

See the [open issues](https://github.com/simoneponcioni/spline-mesher/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are __greatly appreciated__.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Simone Poncioni - simone.poncioni@unibe.ch

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-url]: https://github.com/simoneponcioni/spline-mesher/issues

[Python-url]: https://www.python.org/
[GMSH-url]: http://gmsh.info/
[pyapp]: https://github.com/simoneponcioni/spline-mesher/actions/workflows/python-app.yml/badge.svg
[todo_to_issue]: https://github.com/simoneponcioni/spline-mesher/actions/workflows/todo_to_issue.yml/badge.svg
