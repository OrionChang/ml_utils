from setuptools import setup, find_packages

setup(
    name="ml_utils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
)