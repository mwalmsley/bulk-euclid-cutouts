import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gz-euclid",
    version="0.0.1",
    author="Mike Walmsley",
    author_email="walmsleymk1@gmail.com",
    description="GZ Euclid pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwalmsley/gz-euclid-datalab",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA"
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.9",  # zoobot has 3.9,
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'astropy',
        'tqdm',
        'omegaconf'  # for config
    ]
)
