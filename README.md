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
    <a href="https://artorg-unibe-ch.github.io/spline_mesher/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/artorg-unibe-ch/spline_mesher/blob/master/examples/standalone_execution.ipynb">View Demo</a>
    ·
    <a href="https://github.com/artorg-unibe-ch/spline_mesher/issues">Report Bug</a>
    ·
    <a href="https://github.com/artorg-unibe-ch/spline_mesher/issues">Request Feature</a>
  </p>
</div>

👷🏼 Simone Poncioni<br>🦴 Musculoskeletal Biomechanics Group<br>🎓 ARTORG Center for Biomedical Engineering Research, University of Bern


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


## 📝 Introduction

<p style='text-align: justify;'> A Python package for generating GMSH meshes from SCANCO HR-pQCT images. Import a voxel-based model and convert it to a geometrical simplified representation through the use of splines for each slice in the transverse plane. </p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 💡 Method

- **Structured Mesh**: Provides greater simplicity, efficiency, and lower memory usage due to implicit connectivity with neighboring elements. Ensures topologically identical models for easy comparison between patients or across longitudinal studies.
- **Contour Extraction & Smoothing**: Utilizes scikit-image for periosteal and endosteal contour extraction, followed by Douglas-Ramer-Peucker simplification and 3rd order periodic BSplines for smooth geometry representation.
- **Sanity Check**: Ensures a minimum cortical thickness of 0.5 mm, allowing space for at least three elements and continuity in thin structures.
- **Mesh Generation**: The simplified geometry is imported into GMSH via the OpenCASCADE kernel. Transfinite techniques and the Winslow untangler are used for optimal mesh quality.
- **Quality Assessment**: Mesh quality is evaluated using the Dice similarity coefficient (DSC) with a minimum value of 95%, and the (signed-) inverse conditioning number ((S-) ICN) to ensure matrix conditioning and simulation stability.


_For more examples, please refer to the [Documentation](https://artorg-unibe-ch.github.io/spline_mesher/)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.

## 🔧 Installation

1. Clone the repo

   ```sh
   git clone https://github.com/artorg-unibe-ch/spline_mesher.git
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


<!-- ROADMAP -->
## 🛣️ Roadmap

- [x] __v0.0.1__: single execution of radius and tibia mesher
- [x] __v0.0.2__:
  - [x] Faster implementation of sorting algorithm
  - [x] Faster implementation of cortical sanity check
  - [x] Implement test robustness over different models
- [ ] __v1.1.0__: add phase discrimination in order to mesh single phase models (e.g. vertebrae)

See the [open issues](https://github.com/artorg-unibe-ch/spline_mesher/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## 🤝 Contributing

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
## 📜 License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## 📧 Contact

Simone Poncioni - simone.poncioni@unibe.ch

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[issues-url]: https://github.com/artorg-unibe-ch/spline_mesher/issues

[Python-url]: https://www.python.org/
[GMSH-url]: http://gmsh.info/
[pyapp]: https://github.com/artorg-unibe-ch/spline_mesher/actions/workflows/python-app.yml/badge.svg
[todo_to_issue]: https://github.com/artorg-unibe-ch/spline_mesher/actions/workflows/todo_to_issue.yml/badge.svg

## 🙏 Acknowledgements
This work was funded internally by the ARTORG Center for Biomedical Engineering Research and by the Department of Osteoporosis of the University of Bern. Calculations were performed on <a href="https://www.id.unibe.ch/hpc">UBELIX</a>, the HPC cluster at the University of Bern.

