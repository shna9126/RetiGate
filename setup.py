from setuptools import setup, find_packages

setup(
    name='retigate',
    version='0.1.0',
    description='Bio-inspired motion-saliency pre-filter for object detection',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'scipy>=1.11.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
    ],
)
