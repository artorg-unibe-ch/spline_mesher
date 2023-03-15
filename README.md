<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<h3 align="center">Spline-based structured conformal hexahedral meshing</h3>

  <p align="center">
    Add a project description here
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

<img src="https://github.com/simoneponcioni/spline-mesher/blob/master/src/spline_mesher/docs/img/mesh-example.png" alt="Spline-mesher">

Import a voxel-based model and convert it to a geometrical simplified representation through the use of splines for each slice in the transverse plane.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

[<img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" width="100" height="100">][Python-url]
[<img src="https://gitlab.onelab.info/uploads/-/system/project/avatar/3/gmsh.png" alt="Python" width="100" height="100">][GMSH-url]

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
   ```

3. For developers: install the package in editable mode

   ```sh
   pip install -e .
   ```

4. For developers: install requirements for testing

   ```sh
   pip install -r requirements-dev.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Standalone execution

- Import the MetaImage file of the masked model that we want to convert;
- Using SimpleITK extract the binary contour for each slice (in cortical bone, this contains inner+outer cortex)
- Extract only the outer contour using the OpenCV library and interpolate them over a numpy meshgrid
- First sorting: counterclockwise sorting of points (i, j) wrt CoM of the object
- Second sorting: sorting according to Mahalanobis distance (because it's more independent of the starting points than Euclidean distance)
- Representation of a B-spline of 3rd order over the transverse surface on the points (i, j)
- Meshing
- To be continued
  
_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

- [x] __v0.0.1-pre-alpha__: single execution of radius and tibia mesher
- [ ] __v0.0.2__: add phase discrimination in order to mesh single phase models (e.g. vertebrae)
- [ ] __Performance improvements__:
  - [ ] Faster implementation of sorting algorithm
  - [ ] Faster implementation of cortical sanity check
  - [ ] Implement test robustness over different models

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
[issues-shield]: https://img.shields.io/github/issues/simoneponcioni/spline-mesher.svg?style=for-the-badge
[issues-url]: https://github.com/simoneponcioni/spline-mesher/issues
[license-shield]: https://img.shields.io/github/license/simoneponcioni/spline-mesher.svg?style=for-the-badge
[license-url]: https://github.com/simoneponcioni/spline-mesher/blob/master/LICENSE.txt

[Python-url]: https://www.python.org/
[GMSH-url]: http://gmsh.info/
