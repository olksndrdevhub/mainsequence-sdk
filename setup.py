from setuptools import find_packages
from setuptools import setup
import subprocess
import copy
import pathlib
import os
import toml

BASE_REQUIRED_PACKAGES = ['packages']

def get_install_requirements():
    """
    Gets from pipfile requirements
    Returns
    -------

    """
    try:
        with open ('Pipfile', 'r') as fh:
            pipfile = fh.read()
        pipfile_toml = toml.loads(pipfile)
    except FileNotFoundError:
        return []
    # if the package's key isn't there then just return an empty
    # list
    try:
        required_packages=[]
        for b in BASE_REQUIRED_PACKAGES:
            required_packages.extend(pipfile_toml[b].items())
    except KeyError:
        return []
     # If a version/range is specified in the Pipfile honor it
     # otherwise just list the package
    reqs=[]
    for pkg, ver in required_packages:
        if ver != "*":
            if isinstance(ver,dict):
                if "file" in ver:
                    continue
                req = "{0}{1}".format(pkg, ver['extras'])
            else:
                req="{0}{1}".format(pkg, ver)
        else:
            req=pkg
        reqs.append(req)
    return reqs


#fredapi and quandl neede to create time series from these sources.

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


setup(
  name='mainsequence',
  version=f"1.0.0-{get_git_revision_short_hash()}",
    python_requires='>=3.9.0',
  author='Main Sequence GmbH',
  author_email = 'dev@main-sequence.io',
  install_requires=get_install_requirements(),
    packages=find_packages(include=['mainsequence', 'mainsequence.*']),  # Include only the `mainsequence` package
    description='Main Sequence SDK')