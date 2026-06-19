"""
Setup file for D3+ project.
"""

from setuptools import setup, find_packages

setup(
    name="d3_plus",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "albumentations>=1.3.0",
        "transformers>=4.30.0",
        "timm>=0.9.0",
        "ffmpeg-python>=0.2.0",
        "moviepy>=1.0.3",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-multipart>=0.0.6",
        "aiofiles>=23.2.0",
    ],
    python_requires=">=3.8",
)