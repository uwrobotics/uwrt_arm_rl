import setuptools
from pathlib import Path

setuptools.setup(
    name='UWRTArmGym',
    version='0.0.1',
    description="An OpenAI Gym Env for UWRT ARM",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include="uwrtarm_gym*"),
    install_requires=['gym', 'pybullet', 'numpy'],  # And any other dependencies foo needs
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6'
)
