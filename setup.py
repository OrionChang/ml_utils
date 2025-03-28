from setuptools import setup, find_packages
import os

setup(
    name="ml_utils",
    version="0.1.0",
    author="Orion Chang",
    author_email="ism.zjx@gmail.com",
    description="ML utilities",
    long_description=open("README.md", "r").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/OrionChang/ml_utils",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "torchinfo>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "coverage>=6.0.0",
        ],
    },
)