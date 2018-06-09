"""Setup to install the zipf_classifier package."""
import os
import re
# To use a consistent encoding
from codecs import open
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


def read(*parts):
    """Return the content of the file at given path."""
    with open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    """Return the version number from the __version__ file."""
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


__version__ = find_version("zipf_classifier", "__version__.py")

setup(
    name='zipf_classifier',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='A package to create a deterministic classifier based on Zipf law',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/LucaCappelletti94/zipf_classifier',

    # Author details
    author='Luca Cappelletti',
    author_email='cappelletti.luca94@gmail.com',

    # Choose your license
    license='MIT',

    include_package_data=True,

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering :: Information Analysis',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3'
    ],
    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['zipf']
)


def status(s):
    """Print the version status."""
    print('\033[1m{0}\033[0m'.format(s))


status('Pushing git tags')
os.system('git tag v{0}'.format(__version__))
os.system('git push --tags')
