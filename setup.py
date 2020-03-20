"""pyspikelib: A set of tools for neuronal spiking data mining"""

import os
import re
import codecs
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


DISTNAME = 'pyspikelib'
DESCRIPTION = 'pyspikelib: A set of tools for neuronal spiking data mining'
MAINTAINER = 'Ivan Lazarevich'
MAINTAINER_EMAIL = 'ivan@lazarevi.ch'
URL = 'https://github.com/vanyalzr/pyspikelib'
DOWNLOAD_URL = 'https://github.com/vanyalzr/pyspikelib'
VERSION = find_version(os.path.join(here, 'pyspikelib/version.py'))
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'


INSTALL_REQUIRES = [
    'quantities==0.12.3',
    'neo==0.7.1',
    'matplotlib==3.1.0',
    'numpy==1.16.4',
    'seaborn==0.9.0',
    'tqdm==4.32.2',
    'pandas==0.25.3',
    'elephant==0.6.2',
    'tsfresh==0.13.0',
    'scikit_learn==0.22.1',
]

EXTRAS_REQUIRE = {'tests': ['pytest'], 'docs': []}

setuptools.setup(
    name=DISTNAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
