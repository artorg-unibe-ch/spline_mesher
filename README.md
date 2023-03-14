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

![gitlab issue 1804](https://gitlab.onelab.info/gmsh/gmsh/-/issues/1804)

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

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Spline-based structured conformal hexahedral meshing</h3>

  <p align="center">
    project_description
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

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Import a voxel-based model and convert it to a geometrical simplified representation through the use of splines for each slice in the transverse plane.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

[<img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python" width="100" height="100">][Python-url]
[<img src="https://gitlab.onelab.info/uploads/-/system/project/avatar/3/gmsh.png" alt="Python" width="100" height="100">][GMSH-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

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

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

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
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[product-screenshot]: docs/img/mesh-example.png

[Python-url]: https://www.python.org/
[GMSH-url]: http://gmsh.info/