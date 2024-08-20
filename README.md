# Tadpose: post-SLEAP analysis for tadpoles and other animals
---

A small Python library for convenient post analysis of SLEAP* results. TadPose was developed with the Sweeney lab at ISTA (Austria). To use tadpose the original movie and SLEAP .h5 analysis file is required

## Features
* Alignment of animals into ego-centric views
* Convenient access of original and aligned body-part locations
* Image warping into ego-centric views
* Smoothing spline interpolation of body-part sequences
* Feature computations such as speed, acceleration and angles
* Export of body-parts locations as .csv




## Installation

#### From git master
To install the latest version from repositories master branch

```
pip install git+https://github.com/sommerc/tadpose.git
```

This requires `git` to be installed.

## Examples

Download a SLEAP predicted [demo movie](https://seafile.ist.ac.at/f/a753c77b4243452d8b0c/?dl=1) and unzip.

And follow steps in [example notebook](notebooks/example.ipynb)


