"""spikebench: an open spike train time-series classification benchmark"""

import codecs
import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


DISTNAME = 'spikebench'
DESCRIPTION = 'spikebench: an open spike train time-series classification benchmark'
MAINTAINER = 'Ivan Lazarevich'
MAINTAINER_EMAIL = 'ivan@lazarevi.ch'
URL = 'https://github.com/vanyalzr/spikebench'
DOWNLOAD_URL = 'https://github.com/vanyalzr/spikebench'
VERSION = '1.0'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'


INSTALL_REQUIRES = [
    'addict',
    'pathos',
    'quantities',
    'neo',
    'matplotlib',
    'numpy',
    'seaborn',
    'tqdm',
    'pandas',
    'elephant',
    'tsfresh',
    'scikit_learn',
    'psutil',
    'gdown',
    'pyarrow',
    'chika',
]

EXTRAS_REQUIRE = {'tests': ['pytest'], 'data': ['fastparquet']}

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
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
