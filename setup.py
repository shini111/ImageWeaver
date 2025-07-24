#!/usr/bin/env python3
"""
ImageWeaver Setup Configuration
AI-Powered Image Placement for Translated Documents
"""

from setuptools import setup, find_packages
import pathlib

# Read README for long description
HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-r')]

setup(
    name="imageweaver",
    version="1.0.0",
    description="AI-Powered Image Placement for Translated Documents",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ImageWeaver",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Text Processing :: Markup :: HTML",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: OS Independent",
    ],
    keywords="html image translation ai llm context-matching document-processing",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "openai": ["openai>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "imageweaver=imageweaver_console:main",
            "imageweaver-gui=imageweaver_gui:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ImageWeaver/issues",
        "Source": "https://github.com/yourusername/ImageWeaver",
        "Documentation": "https://github.com/yourusername/ImageWeaver/wiki",
    },
    include_package_data=True,
    zip_safe=False,
)