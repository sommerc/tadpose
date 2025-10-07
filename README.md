# TadPose: post-SLEAP analysis for tadpoles, frogs and other animals
---

A Python library for  post-analysis of SLEAP* pose estimation. TadPose was developed with the Sweeney and Novarino lab at ISTA (Austria). To use TadPose, the original movie and SLEAP .h5 analysis file is required.

## Features
* Alignment of animals into ego-centric views
* Convenient access of original and aligned body-part locations
* Stride and gait analysis
* Image warping into ego-centric views
* Feature computations such as speed, acceleration and angles

## Installation

#### From git master
To install the latest version from repositories master branch

```
pip install git+https://github.com/sommerc/tadpose.git
```

This requires `git` to be installed.

## Examples notebooks

#### Mouse gait analysis
For this [gait example](notebooks/example_gait.ipynb), download and extract a SLEAP processed [example video](https://seafile.ist.ac.at/seafhttp/f/e7f0458cbd384cecb371/?op=view).

For details on the stride metrics see [(Sheppard et al., 2022)](https://www.sciencedirect.com/science/article/pii/S221112472101740X)

#### Basic example with frogs
And follow steps in [example notebook](notebooks/example.ipynb)


