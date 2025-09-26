import pathlib
from setuptools import setup, find_packages

# The directory containing this file
_this_dir = pathlib.Path(__file__).parent

# The text of the README file
long_description = (_this_dir / "README.md").read_text()

# Exec version file
exec((_this_dir / "tadpose" / "version.py").read_text())

setup(
    name="tadpose",
    packages=find_packages(),
    version=__version__,  # noqa: F821
    description="TadPose: post-SLEAP analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sommerc/tadpose",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    #    entry_points = {'console_scripts': []},
    author="Christoph Sommer",
    author_email="christoph.sommer23@gmail.com",
    install_requires=[
        "seaborn",
        "matplotlib",
        "tqdm",
        "scikit-image",
        "scikit-learn",
        "scipy",
        "pandas",
        "numpy",
        "h5py",
        "opencv-python-headless",
        "csaps",
        "roifile",
        "tifffile",
        "ipywidgets",
        "PyYAML",
        "moviepy",
    ],
)
