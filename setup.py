import platform
import warnings
import setuptools


if platform.machine().startswith("arm"):
    warnings.warn("Please ensure you installed CasADI from source")


setuptools.setup()
