# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Install s4l package."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='s4l',
    version='1.0',
    description='Code from the "S4L: Self-supervised Semi-supervised Learning" paper',
    author='Google LLC',
    author_email='no-reply@google.com',
    url='http://github.com/google-research/s4l',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
    },
    scripts=[
    ],
    install_requires=[
        'future',
        'numpy',
        'absl-py',
        'tensorflow',
        'tensorflow-hub',
        # For Google Cloud TPUs
        'google-api-python-client',
        'oauth2client',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow self supervised semi supervised s4l learning',
)
