#!/usr/bin/env python

import setuptools

try:
    with open("README.md", "r") as f:
        long_description = f.read()
except:
    long_description=""

setuptools.setup(
        name="efl",
        version="0.0.1a1",
        author="Ian Taylor",
        author_email="ian@iantaylor.xyz",
        description="Data and Models for EFL games and teams",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ianmtaylor1",
        packages=setuptools.find_packages(),
        classifiers=[
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.7"
        ],
        install_requires=[
            'appdirs>=1.4.3',
            'configparser>=3.7.4',
            'pandas>=0.25.0',
            'pystan>=2.19.0.0',
            'requests>=2.22.0',
            'sqlalchemy>=1.3.5',
        ],
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'download-efl-games = efl.data.main:console_download_games',
            ],
        }
)
