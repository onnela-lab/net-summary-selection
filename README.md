# Selection of summary statistics for network model choice [![CI](https://github.com/onnela-lab/net-summary-selection/actions/workflows/main.yml/badge.svg)](https://github.com/onnela-lab/net-summary-selection/actions/workflows/main.yml)

We here provide a Python package named `cost_based_selection` associated with our manuscript

L. Raynal, T. Hoffmann, and J.-P. Onnela. "Selection of summary statistics for network model choice." *arXiv*, [2101.07766](https://arxiv.org/abs/2101.07766), 2021.

## Table of contents

* [Description](#description)
* [Installation](#installation)
* [Reproducing the results](#reproducing-the-results)

## Description

This project focuses on cost-based selection of features for distinguishing between different (mechanistic network) models. The computational cost of features can vary significantly, and, in computationally demanding settings such as approximate Bayesian computation, selecting low-cost yet informative features is desirable. We consider cost-based adaptations of a range of feature selection methods as well as using pilot simulations based on smaller networks to identify informative features.

## Installation

1. Clone the git repository or download the code as an archive.
2. Set up a new Python virtual environment, e.g., using `pyenv`. This code has been tested on python 3.9.9.
3. Install all python requirements by running `pip install -r requirements.txt`.
4. Ensure an R interpreter is installed (see https://www.r-project.org for installation instructions).
5. Install the [`ranger`](https://www.rdocumentation.org/packages/ranger/versions/0.14.1/topics/ranger) package for random forests, e.g., by running `R -e 'if (!require("ranger")) install.packages("ranger", version="0.14.1", repos="http://cran.r-project.org/")'`.
6. Run `pytest -v` to verify your installation.

## Reproducing the results

You can reproduce all results presented in our paper by running the following commands

```bash
# Ignore figures until we've generated all other results.
doit ignore figures
# Run the analysis (this will take some time ...).
doit -n [number of cores]
# Forget that we ignored figures ...
doit forget figures
# ... and generate them.
doit figures
```
Your results may differ slightly from ours because computation times differ across machines and depend on other processes running on your machine. For reference, the results presented in the manuscript were obtained with `doit -n 6` on a M1 Macbook Pro (2020) with 16GB of memory.
