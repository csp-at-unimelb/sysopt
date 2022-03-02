from setuptools import setup, find_packages
from sysopt.version import version

with open("requirements.txt", 'r') as fh:
    requirements = [line.strip() for line in fh.readlines()]

setup(
    name="sysopt",
    version=version,
    author="Peter Cudmore",
    author_email="peter.cudmore@unimelb.edu.au",
    url="https://github.com/csp-at-unimelb/sysopt",
    description="Component-based systems modelling library.",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License v2.0",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries"
    ],
    keywords="modelling, control systems, engineering, optimisation",
    install_requires=requirements
)
