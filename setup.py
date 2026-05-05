"""
setup.py — SARDS Package Setup
================================
Makes SARDS installable as a Python package via:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = [
    line.strip()
    for line in (Path(__file__).parent / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]

setup(
    name                          = "sards",
    version                       = "1.0.0",
    author                        = "SARDS Team",
    description                   = (
        "Satellite-Based Activity Risk Detection System — "
        "Urban change monitoring for Dhaka, Bangladesh"
    ),
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    url                           = "https://github.com/yourusername/SARDS",
    packages                      = find_packages(exclude=["tests*", "scripts*"]),
    python_requires               = ">=3.9",
    install_requires              = requirements,
    extras_require = {
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=6.0",
        ],
        "gee": [
            "earthengine-api>=0.1.370",
        ],
    },
    entry_points = {
        "console_scripts": [
            "sards=main:main",
        ],
    },
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords = [
        "satellite", "remote-sensing", "change-detection",
        "urban-analytics", "dhaka", "bangladesh",
        "geospatial", "opencv", "streamlit", "risk-assessment"
    ],
)
