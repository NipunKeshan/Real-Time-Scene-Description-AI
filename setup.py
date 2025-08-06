"""
Setup script for Real-Time AI Scene Description System.
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="real-time-scene-description-ai",
    version="2.0.0",
    author="Nipun Keshan",
    author_email="your-email@domain.com",
    description="Advanced Real-Time AI Scene Description System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NipunKeshan/Real-Time-Scene-Description-AI",
    project_urls={
        "Bug Tracker": "https://github.com/NipunKeshan/Real-Time-Scene-Description-AI/issues",
        "Documentation": "https://github.com/NipunKeshan/Real-Time-Scene-Description-AI/wiki",
        "Source Code": "https://github.com/NipunKeshan/Real-Time-Scene-Description-AI",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.22.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "gunicorn>=21.0.0",
        ],
        "ui": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rtsd=src.main:main",
            "rtsd-cli=src.main:main",
            "rtsd-api=src.api.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.yaml"],
    },
    keywords=[
        "artificial intelligence",
        "computer vision",
        "machine learning",
        "real-time",
        "scene description",
        "image captioning",
        "object detection",
        "video processing",
        "deep learning",
        "pytorch",
        "transformers",
        "streamlit",
        "fastapi",
    ],
    zip_safe=False,
)
