import platform
import warnings
import setuptools

with open("requirements.txt", 'r') as fh:
    requirements = [line.strip() for line in fh.readlines()]

if platform.machine().startswith("arm"):
    warnings.warn("Please ensure you installed CasADI from source")

setuptools.setup(
    name="sysopt",
    version="0.0.3",
    author="Peter Cudmore",
    author_email="peter.cudmore@unimelb.edu.au",
    url="https://github.com/csp-at-unimelb/sysopt",
    description="Component-based systems modelling library.",
    classifiers=[
        "Programming Language:: Python:: 3",
        "License:: OSI Approved:: Apache Software License",
        "Operating System:: OS Independent Development",
        "Status:: 3 - Alpha"
        "Topic:: Scientific / Engineering",
        "Topic:: Software Development:: Libraries"
    ],
    packages=['sysopt'],
    package_dir={'sysopt': 'sysopt'},
    install_requires=requirements
)