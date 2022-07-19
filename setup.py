#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
import os

package_dirs = {
    'image_processing': os.path.join('src', 'image_processing'),
}

d = generate_distutils_setup(
    packages=package_dirs.keys(),
    package_dir=package_dirs,
)
setup(**d)
