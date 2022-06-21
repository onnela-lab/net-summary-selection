import setuptools
from Cython.Build import cythonize


with open("DESCRIPTION.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cost_based_selection",
    version="0.1",
    author="Louis Raynal, Jukka-Pekka Onnela",
    author_email="onnela@hsph.harvard.edu",
    description="A package implementing various cost-based feature selection methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/onnela-lab/net_summary_selection",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules=cythonize('cost_based_selection/*.pyx', annotate=True),
    install_requires=[
            'cython',
            'joblib>=0.17.0',
            'matplotlib>=3.3.2',
            'networkit>=10',
            'networkx>=2.5',
            'numpy>=1.19.2',
            'pandas>=1.1.3',
            'rpy2>=2.9.3',
            'scipy>=1.5.2',
            'sklearn',
            'tqdm',
            'doit',
    ],
    extras_require={
        'tests': [
            'flake8',
            'pytest',
            'pytest-cov',
        ]
    },
    package_data={'': ['data/*.csv']}
)
