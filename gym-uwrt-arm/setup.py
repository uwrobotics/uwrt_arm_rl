import setuptools

setuptools.setup(
    name='gym-uwrt-arm',
    version='0.0.2',
    description="An OpenAI Gym Env for UWRT's ARM",
    install_requires=['gym', 'pybullet', 'numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6'
)
