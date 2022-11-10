import platform
import warnings
import setuptools

with open("requirements.txt", 'r') as fh:
    requirements = [line.strip() for line in fh.readlines()]

if platform.machine().startswith("arm"):
    warnings.warn("Please ensure you installed CasADI from source")

with open("version", 'r') as fp:
    major, minor, revision = fp.readline().split('.')
    version = f'{major}.{minor}.{revision}'

setuptools.setup(
    name="sysopt",
    version=version,
    author="Peter Cudmore",
    author_email="peter.cudmore@uqconnect.edu.au",
    url="https://github.com/csp-at-unimelb/sysopt",
    description="Component-based systems modelling library.",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries"
    ],
    packages=['sysopt'],
    package_dir={'sysopt': 'sysopt'},
    install_requires=requirements
)
