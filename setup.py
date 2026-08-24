"""Compatibility shim for offline environments with older setuptools."""

from setuptools import find_packages, setup


setup(
    name="target-space-dream",
    version="0.1.0",
    description="Target-space continuation for high-fidelity function distillation",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=["torch>=2.3"],
    extras_require={"vision": ["torchvision>=0.18"], "test": ["pytest>=8"]},
    entry_points={"console_scripts": ["dream-toy=dream.cli:main"]},
)

