"""
setup.py - a module to allow package installation
"""

from distutils.core import setup


NAME = "qoc2"
VERSION = "0.1"
DEPENDENCIES = [
    "jax",
    "jaxlib",
    "numpy",
    "scipy",
]
DESCRIPTION = "a package for performing quantum optimal control"
AUTHOR = "Yunwei Lu"
AUTHOR_EMAIL = "yunweilu2020@u.northwestern.edu"
PY_MODULE = []
setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
        py_modules=PY_MODULE
)
